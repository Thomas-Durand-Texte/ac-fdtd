"""The pressure-only ("scalar", three-level) 2D scheme: the low-memory peer of the first-order one.

(:mod:`ac_fdtd.scheme2d`.)

Why a second scheme
-------------------
:class:`ac_fdtd.scheme2d.AcousticFDTD2D` stores one pressure field and *two* velocity components —
three cell-sized arrays, four in 3D. Eliminating the velocity algebraically leaves the second-order
wave equation on pressure alone,

    p^{n+1} = 2 p^n - p^{n-1} + lam^2 * L(p^n),      lam = c dt / dx,

which stores **two** arrays in any dimension (``p`` and the level before it). That is a third less
memory in 2D and half in 3D, and — because it touches less memory per step on a bandwidth-bound
problem — about two to three times faster. It is **not an approximation**: started from
``p^{-1} = p^0`` (which is the first-order form's ``v^{-1/2} = 0``) it reproduces
:class:`AcousticFDTD2D` to the last bit, interior and boundaries alike. ``tests/test_scalar2d.py``
holds that equivalence against the reference.

The velocity does not vanish from the physics, only from storage. Where it is still needed — a
velocity probe, a locally-reacting wall, a rigid face — it is an O(1) quantity reconstructed from
the two pressure levels, not a grid. See :meth:`add_velocity_receiver`.

What is derived here
-------------------
Each boundary of :mod:`ac_fdtd.scheme2d` becomes a *three-level* relation on pressure:

* **Rigid wall / rigid obstacle** — the Laplacian uses a zero-gradient (Neumann) one-sided stencil
  at the wall and a closed face next to a solid cell. A solid cell, ringed by closed faces, keeps
  ``p = 0``. This is exactly the first-order form's "never write the wall velocity line".
* **Locally-reacting wall**, normalised admittance ``xi`` — derived by carrying the previous step's
  own wall recurrence (the interior identity does *not* hold at the wall, because ``p^n`` there was
  itself produced by the corrected update):

      (1 + a) p^{n+1} = 2 p^n - (1 - a) p^{n-1} + lam^2 * L_N(p^n),     a = c dt xi / (2 dx),

  with ``a`` summed over the walls meeting at a corner — so edges and corners need no special case,
  exactly as in the first-order form. The damping sits on the ``(p^{n+1} - p^{n-1})`` term, which is
  the discrete first time derivative: a radiation term, as the physics requires.
* **Soft volume source** — injected as the *difference* ``S_n - S_{n-1}``, because the eliminated
  divergence already carries ``S_{n-1}``. Injecting ``S_n`` directly is the classic pressure-only
  source error and shows up as a slow drift.

The matched absorbing layer and the air-relaxation states of :class:`AcousticFDTD2D` are *not* here:
they are natural in a first-order system and a pressure-only port of each is separate work. The
open-boundary job is done instead by a CFS-CPML shell (added in a later revision of this module),
which is both a better absorber and the same mechanism as a tuned partial-absorbing wall.
"""

from __future__ import annotations

import numpy as np

from .boundaries2d import WallAdmittances2D, wall_update_coefficients_2d
from .cpml2d import AbsorbingBoundary2D, cpml_axis_profiles
from .grid2d import Grid2D, max_stable_time_step_2d
from .medium import AIR, Medium

__all__ = ["ScalarFDTD2D"]


class ScalarFDTD2D:
    """Pressure-only propagation in a rectangular 2D domain, rigid by default.

    The constructor mirrors :class:`ac_fdtd.scheme2d.AcousticFDTD2D` for the features it shares, so
    the two are drop-in alternatives wherever only pressure is read. The state is the pair
    ``(p^n, p^{n-1})``; at rest the two levels are equal, which is the first-order form's ``v = 0``.

    Args:
        grid: Geometry and spacing.
        medium: Fluid properties. Defaults to air at 20 degrees C.
        courant: Fraction of the stability limit; 1.0 (default) is the limit and least dispersive.
        dtype: Working precision. Double by default — this is a reference, not a fast backend.
        walls: Normalised admittance of each of the four walls. Default is rigid throughout.
        rigid_cells: Boolean mask of grid shape marking solid interior cells (rigid obstacles).
        absorbing_boundary: A CFS-CPML region lining a subset of the walls (open field, or a tuned
            partial absorber). When present, those walls stop being rigid/impedance and the scheme
            runs a first-order CPML update in the boundary shell and the scalar update everywhere
            else; the shell is the only place that stores velocity, so a closed room still holds two
            arrays and an open one holds velocity only in the thin shell. Walls covered by the
            boundary should not also carry an admittance in ``walls`` (the boundary wins).
    """

    def __init__(
        self,
        grid: Grid2D,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
        walls: WallAdmittances2D | None = None,
        rigid_cells: np.ndarray | None = None,
        absorbing_boundary: AbsorbingBoundary2D | None = None,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")

        self.grid = grid
        self.medium = medium
        self.courant = courant
        self.walls = walls or WallAdmittances2D()
        self.dtype = np.dtype(dtype)
        self.dt = courant * max_stable_time_step_2d(grid.dx, medium.sound_speed)

        #: Number of completed steps. The state is (p^{step_index}, p^{step_index - 1}).
        self.step_index = 0

        self.p = np.zeros(grid.shape, dtype=self.dtype)
        self.p_old = np.zeros(grid.shape, dtype=self.dtype)
        self._lambda_squared = self.dtype.type((medium.sound_speed * self.dt / grid.dx) ** 2)
        self._source_coefficient = self.dtype.type(medium.density * medium.sound_speed**2 * self.dt)

        # Locally-reacting wall coefficients, shared verbatim with the first-order form so the two
        # schemes are bit-identical: a = (c dt / 2 dx) * xi, summed over walls at a corner cell.
        indices, coefficient = wall_update_coefficients_2d(grid, medium, self.dt, self.walls)
        self._wall_indices = indices
        self._wall_a = coefficient.astype(self.dtype)

        # Rigid obstacles: open-face masks (1.0 where a face is open, 0.0 where it touches a solid),
        # mirroring AcousticFDTD2D's held-at-zero velocity faces. Built on the x- and y-faces that
        # lie *between* cells; the domain's own walls are handled by the Neumann Laplacian.
        self._rigid_cells: np.ndarray | None = None
        self._x_open: np.ndarray | None = None
        self._y_open: np.ndarray | None = None
        if rigid_cells is not None:
            rigid = np.ascontiguousarray(rigid_cells, dtype=bool)
            if rigid.shape != grid.shape:
                raise ValueError(
                    f"rigid_cells shape {rigid.shape} does not match the grid {grid.shape}"
                )
            self._rigid_cells = rigid
            self._x_open = (~(rigid[:-1, :] | rigid[1:, :])).astype(self.dtype)
            self._y_open = (~(rigid[:, :-1] | rigid[:, 1:])).astype(self.dtype)

        # CFS-CPML open boundary (optional). Builds the hybrid: a p-v + CPML shell around the scalar
        # interior. Velocity and the four memory variables live on the full grid here for a readable
        # reference, but are only updated on the faces a shell cell needs — the interior never gains
        # a velocity. The compiled kernel stores the shell slabs only.
        self.absorbing_boundary = absorbing_boundary
        self._cpml = absorbing_boundary is not None
        if absorbing_boundary is not None:
            self._setup_cpml(absorbing_boundary)

        self._sources: list[tuple[tuple[int, int], np.ndarray]] = []
        self._receivers: list[tuple[tuple[int, int], list[float]]] = []
        self._velocity_receivers: list[list] = []
        self._prescribed_pressure: list[tuple[tuple[int, int], int, np.ndarray]] = []

    def _setup_cpml(self, boundary: AbsorbingBoundary2D) -> None:
        grid = self.grid
        nx, ny = grid.shape
        thickness = boundary.thickness
        if 2 * thickness >= min(nx, ny):
            raise ValueError(
                f"absorbing layer of {thickness} cells does not fit a {grid.shape} grid"
            )

        profiles = cpml_axis_profiles(grid, self.medium, self.dt, boundary)
        cast = self.dtype.type
        self._kx_p, self._bx_p, self._ax_p = (c.astype(self.dtype) for c in profiles[("p", 0)])
        self._ky_p, self._by_p, self._ay_p = (c.astype(self.dtype) for c in profiles[("p", 1)])
        self._kx_v, self._bx_v, self._ax_v = (c.astype(self.dtype) for c in profiles[("v", 0)])
        self._ky_v, self._by_v, self._ay_v = (c.astype(self.dtype) for c in profiles[("v", 1)])

        # The gradients below carry the 1/dx explicitly, so these coefficients do not.
        self._inverse_dx = cast(1.0 / grid.dx)
        self._cv = cast(self.dt / self.medium.density)
        self._cp = cast(self.medium.density * self.medium.sound_speed**2 * self.dt)

        self.vx = np.zeros(grid.velocity_shape(0), dtype=self.dtype)
        self.vy = np.zeros(grid.velocity_shape(1), dtype=self.dtype)
        self._psi_vx = np.zeros(grid.velocity_shape(0), dtype=self.dtype)
        self._psi_vy = np.zeros(grid.velocity_shape(1), dtype=self.dtype)
        self._psi_px = np.zeros(grid.shape, dtype=self.dtype)
        self._psi_py = np.zeros(grid.shape, dtype=self.dtype)

        # shell = cells inside any lined layer; the interior runs the scalar update.
        in_x = np.zeros(nx, dtype=bool)
        in_y = np.zeros(ny, dtype=bool)
        low_x, high_x = boundary.has_axis(0)
        low_y, high_y = boundary.has_axis(1)
        if low_x:
            in_x[:thickness] = True
        if high_x:
            in_x[nx - thickness :] = True
        if low_y:
            in_y[:thickness] = True
        if high_y:
            in_y[ny - thickness :] = True
        shell = in_x[:, None] | in_y[None, :]
        self._shell = shell
        self._interior = ~shell
        # a velocity face is needed when either cell it borders is a shell cell
        self._need_x = np.zeros(grid.velocity_shape(0), dtype=bool)
        self._need_x[1:-1, :] = shell[:-1, :] | shell[1:, :]
        self._need_y = np.zeros(grid.velocity_shape(1), dtype=bool)
        self._need_y[:, 1:-1] = shell[:, :-1] | shell[:, 1:]

        # an impedance wall correction only applies to interior wall cells; a shell corner cell is
        # pure p-v + CPML and ignores any admittance on its other face.
        if self._wall_indices.size:
            keep = self._interior.reshape(-1)[self._wall_indices]
            self._wall_indices = self._wall_indices[keep]
            self._wall_a = self._wall_a[keep]

    # -- time and probes -------------------------------------------------------------------
    @property
    def time(self) -> float:
        """Instant the stored pressure field belongs to, in seconds."""
        return self.step_index * self.dt

    @property
    def sample_rate(self) -> float:
        """Rate the recordings are at, in Hz. Set by the Courant condition, not by choice."""
        return 1.0 / self.dt

    def probe_pressure(self, point: tuple[float, float]) -> float:
        """Pressure at ``point``, at time :attr:`time`, in Pa."""
        return float(self.p[self.grid.cell_index(point)])

    def add_velocity_receiver(self, point: tuple[float, float], axis: int) -> int:
        """Record a velocity component on the lower face of ``point`` every step, and return its
        channel into :attr:`recorded_velocity`.

        The velocity grid is eliminated, so the probe keeps one running value instead — the momentum
        step the scheme would have taken, accumulated at this one face:
        ``v^{n+1/2} = v^{n-1/2} - (dt/(rho dx)) (p^n_here - p^n_lower)``. O(1) per probe, not a
        field. The recorded sample is ``v^{n-1/2}``, half a step behind the pressure at the same
        index — exactly the offset :meth:`AcousticFDTD2D.probe_velocity` documents; average two
        consecutive samples to recentre on a whole step. The lower face of a boundary cell is the
        rigid wall line, which is always zero.
        """
        index = self.grid.cell_index(point)
        self._velocity_receivers.append([index, axis, 0.0, []])
        return len(self._velocity_receivers) - 1

    @property
    def recorded_velocity(self) -> np.ndarray:
        """Everything the velocity receivers have recorded, as ``(channel, sample)``."""
        if not self._velocity_receivers:
            return np.zeros((0, 0))
        return np.array([samples for _, _, _, samples in self._velocity_receivers])

    # -- construction API (the subset shared with AcousticFDTD2D) --------------------------
    def set_initial_pressure(self, field: np.ndarray) -> None:
        """Start from a given pressure at rest — both levels equal, so the velocity is zero."""
        self.p[:] = field
        self.p_old[:] = field

    def set_initial_state(self, pressure: np.ndarray, velocity: list[np.ndarray]) -> None:
        """Seed ``(p^0, p^{-1})`` from a first-order initial pair ``(p^0, v^{-1/2})``.

        The level before the start is the pressure update run backwards one step:
        ``p^{-1} = p^0 + rho c^2 dt * div(v^{-1/2})``. This is what launches a single eigenmode (see
        :func:`ac_fdtd.analytic2d.mode_initial_state`) as a pure sinusoid, exactly as in the
        first-order scheme.
        """
        self.p[:] = pressure
        coefficient = self.medium.density * self.medium.sound_speed**2 * self.dt / self.grid.dx
        divergence = np.zeros(self.grid.shape, dtype=self.dtype)
        divergence += velocity[0][1:, :] - velocity[0][:-1, :]
        divergence += velocity[1][:, 1:] - velocity[1][:, :-1]
        self.p_old[:] = pressure + coefficient * divergence

    def add_volume_source(self, point: tuple[float, float], signal: np.ndarray) -> None:
        """Inject a soft volume-velocity source into the cell containing ``point``.

        Identical semantics and timing to :meth:`AcousticFDTD2D.add_volume_source`: ``signal`` is a
        volume velocity per unit area (1/s), one sample per step, sample ``n`` acting on the step
        ``p^n -> p^{n+1}``. Soft (added, not overwritten), so it does not scatter returning waves.
        """
        index = self.grid.cell_index(point)
        self._sources.append((index, np.asarray(signal, dtype=self.dtype)))

    def add_prescribed_source(
        self,
        point: tuple[float, float],
        signal: np.ndarray,
        start: int = 0,
        kind: str = "pressure",
    ) -> None:
        """Hold one node equal to ``signal`` for a window of steps, then let it go.

        Only ``kind="pressure"`` is supported here; a prescribed *velocity* source needs a stored
        face and is added with the CPML shell in a later revision.
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if kind != "pressure":
            raise ValueError(
                f"the pressure-only scheme supports kind='pressure' only here, got {kind!r}"
            )
        samples = np.asarray(signal, dtype=self.dtype)
        self._prescribed_pressure.append((self.grid.cell_index(point), start, samples))

    def add_pressure_receiver(self, point: tuple[float, float]) -> int:
        """Record the pressure at ``point`` every step from now on, and return its channel."""
        self._receivers.append((self.grid.cell_index(point), []))
        return len(self._receivers) - 1

    @property
    def recorded_pressure(self) -> np.ndarray:
        """Everything the receivers have recorded, as ``(channel, sample)``."""
        if not self._receivers:
            return np.zeros((0, 0))
        return np.array([samples for _, samples in self._receivers])

    # -- the kernel ------------------------------------------------------------------------
    def _laplacian(self, p: np.ndarray) -> np.ndarray:
        """Sum of second differences with zero-gradient (rigid) walls and closed obstacle faces.

        This is exactly the divergence of the velocity the eliminated momentum step would produce:
        a flux across each interior face, masked to zero where a face touches a solid cell, with the
        domain's boundary lines contributing nothing (the Neumann wall).
        """
        lap = np.zeros_like(p)
        flux_x = p[1:, :] - p[:-1, :]
        if self._x_open is not None:
            flux_x = flux_x * self._x_open
        lap[:-1, :] += flux_x
        lap[1:, :] -= flux_x
        flux_y = p[:, 1:] - p[:, :-1]
        if self._y_open is not None:
            flux_y = flux_y * self._y_open
        lap[:, :-1] += flux_y
        lap[:, 1:] -= flux_y
        return lap

    def _apply_impedance_walls(self, p_new: np.ndarray, p: np.ndarray, lap: np.ndarray) -> None:
        """Overwrite the locally-reacting wall cells with their three-level relation.

        ``(1 + a) p^{n+1} = 2 p^n - (1 - a) p^{n-1} + lam^2 L_N(p^n)``. When a CPML boundary is also
        present the indices have been filtered to interior cells, so a shell corner never arrives.
        """
        if not self._wall_indices.size:
            return
        flat_new = p_new.reshape(-1)
        flat_p = p.reshape(-1)
        flat_old = self.p_old.reshape(-1)
        flat_lap = lap.reshape(-1)
        idx = self._wall_indices
        a = self._wall_a
        flat_new[idx] = (
            2.0 * flat_p[idx] - (1.0 - a) * flat_old[idx] + self._lambda_squared * flat_lap[idx]
        ) / (1.0 + a)

    def _apply_cpml(self, p: np.ndarray, p_new: np.ndarray) -> None:
        """Run the first-order CPML update in the shell and overwrite its pressure in ``p_new``.

        The scalar update has already filled ``p_new`` everywhere (the shell included, then
        replaced). Velocity and the memory variables are advanced only on the faces a shell cell
        reads, so the interior keeps no velocity.
        """
        inverse_dx = self._inverse_dx

        gradient_x = (p[1:, :] - p[:-1, :]) * inverse_dx
        mask_x = self._need_x[1:-1, :]
        self._psi_vx[1:-1, :] = np.where(
            mask_x,
            self._bx_v[1:-1, None] * self._psi_vx[1:-1, :] + self._ax_v[1:-1, None] * gradient_x,
            self._psi_vx[1:-1, :],
        )
        update_x = self._cv * (gradient_x / self._kx_v[1:-1, None] + self._psi_vx[1:-1, :])
        self.vx[1:-1, :] = np.where(mask_x, self.vx[1:-1, :] - update_x, self.vx[1:-1, :])

        gradient_y = (p[:, 1:] - p[:, :-1]) * inverse_dx
        mask_y = self._need_y[:, 1:-1]
        self._psi_vy[:, 1:-1] = np.where(
            mask_y,
            self._by_v[None, 1:-1] * self._psi_vy[:, 1:-1] + self._ay_v[None, 1:-1] * gradient_y,
            self._psi_vy[:, 1:-1],
        )
        update_y = self._cv * (gradient_y / self._ky_v[None, 1:-1] + self._psi_vy[:, 1:-1])
        self.vy[:, 1:-1] = np.where(mask_y, self.vy[:, 1:-1] - update_y, self.vy[:, 1:-1])

        divergence_x = (self.vx[1:, :] - self.vx[:-1, :]) * inverse_dx
        self._psi_px = self._bx_p[:, None] * self._psi_px + self._ax_p[:, None] * divergence_x
        divergence_y = (self.vy[:, 1:] - self.vy[:, :-1]) * inverse_dx
        self._psi_py = self._by_p[None, :] * self._psi_py + self._ay_p[None, :] * divergence_y
        p_shell = p - self._cp * (
            divergence_x / self._kx_p[:, None]
            + self._psi_px
            + divergence_y / self._ky_p[None, :]
            + self._psi_py
        )
        p_new[self._shell] = p_shell[self._shell]

    def step(self) -> None:
        """Advance one time step, preserving the ``(p^n, p^{n-1})`` pairing."""
        p = self.p
        self._record_velocity(p)

        lap = self._laplacian(p)
        p_new = 2.0 * p - self.p_old + self._lambda_squared * lap
        self._apply_impedance_walls(p_new, p, lap)
        if self._cpml:
            self._apply_cpml(p, p_new)

        self._inject_sources(p_new)
        self._apply_prescribed_pressure(p_new)
        if self._rigid_cells is not None:
            p_new[self._rigid_cells] = 0.0

        self.p_old = p
        self.p = p_new
        self.step_index += 1
        for index, samples in self._receivers:
            samples.append(self.p[index])

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    def _inject_sources(self, p_new: np.ndarray) -> None:
        step = self.step_index
        for index, signal in self._sources:
            now = signal[step] if step < len(signal) else 0.0
            previous = signal[step - 1] if 0 < step <= len(signal) else 0.0
            delta = now - previous
            if delta != 0.0:
                p_new[index] += self._source_coefficient * delta

    def _record_velocity(self, p: np.ndarray) -> None:
        """Append ``v^{n-1/2}`` for each probe, then advance its running value using ``p = p^n``."""
        if not self._velocity_receivers:
            return
        coefficient = self.dt / (self.medium.density * self.grid.dx)
        for record in self._velocity_receivers:
            index, axis, running, samples = record
            samples.append(running)
            if index[axis] > 0:
                lower = list(index)
                lower[axis] -= 1
                record[2] = running - coefficient * (p[index] - p[tuple(lower)])

    def _apply_prescribed_pressure(self, p_new: np.ndarray) -> None:
        step = self.step_index
        for index, start, signal in self._prescribed_pressure:
            offset = step - start
            if 0 <= offset < signal.size:
                p_new[index] = signal[offset]
