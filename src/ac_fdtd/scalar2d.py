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

The matched absorbing layer of :class:`AcousticFDTD2D` is *not* here: it is natural in a
first-order system. The air-relaxation states are (one cell array per process), driven by the
divergence the scheme rebuilds from its two pressure levels: see
:meth:`ScalarFDTD2D._relaxed_update`. The open-boundary job is done instead by a CFS-CPML shell
(added in a later revision of this module), which is both a better absorber and the same mechanism
as a tuned partial-absorbing wall.
"""

from __future__ import annotations

import math

import numpy as np

from .air import AirAbsorption
from .boundaries2d import WallAdmittances2D, wall_update_coefficients_2d
from .cpml2d import AbsorbingBoundary2D, cpml_axis_profiles
from .grid2d import Grid2D, max_stable_time_step_2d
from .medium import AIR, Medium

__all__ = ["ScalarFDTD2D"]


class _VelocityFace:
    """One driven velocity face: its one stored value, its windows and its wall coefficients.

    ``value`` is ``v^{n-1/2}`` on the face, the one number the pressure-only form must keep to know
    what the first-order scheme's velocity is. ``windows`` are the prescribed samples in insertion
    order; where windows overlap, the one added later wins, as in the reference. ``closed`` is set
    when a rigid obstacle touches an interior face: the reference masks such a face to zero every
    step, so it is silent and stays at zero. ``a_cell`` and ``a_lower`` are the locally-reacting
    wall coefficients of the two cells the face separates (zero off the wall ring; ``a_lower`` is
    unused on a wall line, where ``lower`` is ``None``).
    """

    __slots__ = ("cell", "lower", "closed", "a_cell", "a_lower", "value", "windows")

    def __init__(
        self,
        cell: tuple[int, int],
        lower: tuple[int, int] | None,
        closed: bool,
        a_cell: np.floating,
        a_lower: np.floating,
        zero: np.floating,
    ) -> None:
        self.cell = cell
        self.lower = lower
        self.closed = closed
        self.a_cell = a_cell
        self.a_lower = a_lower
        self.value = zero
        self.windows: list[tuple[int, np.ndarray]] = []


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
        air_absorption: Atmospheric absorption (ISO 9613-1) with the first-order scheme's semantics
            and constants: one cell array per relaxation process, driven by a divergence rebuilt
            from the two pressure levels (see :meth:`_relaxed_update`), so no velocity grid. Not
            combinable with ``absorbing_boundary``: the CPML shell has no relaxation states.
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
        air_absorption: AirAbsorption | None = None,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")
        if air_absorption is not None and absorbing_boundary is not None:
            raise ValueError(
                "air_absorption cannot be combined with absorbing_boundary: the CPML shell has no "
                "relaxation states, so the absorption would silently stop at the shell"
            )

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

        # Air absorption: the first-order scheme's relaxation states, one cell array per process,
        # with the same constants (decay, dt * mean weight, gain), cast the same way. The scheme
        # stays pressure-only: the divergence that drives the states is reconstructed each step
        # from (p^n, p^{n-1}) and the previous step's offset instead of being stored.
        self.air_absorption = air_absorption
        self._relaxation: list[tuple[np.ndarray, np.floating, np.floating, np.floating]] = []
        if air_absorption is not None:
            processes = air_absorption.processes(medium.sound_speed, self.dt)
            relaxed_speed_squared = medium.sound_speed**2 / (
                1.0 + sum(process.strength for process in processes)
            )
            for process in processes:
                decay = math.exp(-self.dt / process.relaxation_time)
                mean_weight = (process.relaxation_time / self.dt) * (1.0 - decay)
                gain = medium.density * relaxed_speed_squared * process.strength / grid.dx
                self._relaxation.append(
                    (
                        np.zeros(grid.shape, dtype=self.dtype),
                        self.dtype.type(decay),
                        self.dtype.type(self.dt * mean_weight),
                        self.dtype.type(gain),
                    )
                )
        # p* = p^n + R^n - alpha D^{n+1/2} is the first-order pressure update with the relaxation
        # folded in: its own rho c^2 dt / dx plus, per process, gain * (dt * mean weight - dt).
        dt = self.dtype.type(self.dt)
        self._pressure_coefficient = self.dtype.type(
            medium.density * medium.sound_speed**2 * self.dt / grid.dx
        )
        self._relaxed_coefficient = self._pressure_coefficient + sum(
            (gain * (weight - dt) for _, _, weight, gain in self._relaxation), self.dtype.type(0.0)
        )
        self._momentum_coefficient = self.dtype.type(self.dt / (medium.density * grid.dx))

        self._sources: list[tuple[tuple[int, int], np.ndarray]] = []
        self._receivers: list[tuple[tuple[int, int], list[float]]] = []
        self._velocity_receivers: list[list] = []
        self._prescribed_pressure: list[tuple[tuple[int, int], int, np.ndarray]] = []
        # Driven velocity faces, keyed by (axis, cell whose lower face it is). Insertion order is
        # the order the corrections are applied in, so the compiled kernel can follow it exactly.
        self._faces: dict[tuple[int, tuple[int, int]], _VelocityFace] = {}

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
        axis: int | None = None,
    ) -> None:
        """Drive one node or one velocity face for a window of steps, then let it go.

        The contract of :meth:`ac_fdtd.scheme2d.AcousticFDTD2D.add_prescribed_source`. A pressure
        source holds the cell at ``signal`` (Pa). A velocity source drives the component ``axis`` on
        the lower face of the cell containing ``point`` (m/s), so a cell on the x_min (or y_min)
        wall drives the wall line itself. A driven face keeps one stored value, folded into the two
        cells it separates; see :meth:`_update_velocity_faces` for the derivation.
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        samples = np.asarray(signal, dtype=self.dtype)
        cell = self.grid.cell_index(point)
        if kind == "pressure":
            if axis is not None:
                raise ValueError("axis is meaningless for a pressure source; leave it None")
            self._prescribed_pressure.append((cell, start, samples))
        elif kind == "velocity":
            if axis not in (0, 1):
                raise ValueError(f"velocity source needs axis in (0, 1), got {axis!r}")
            if self.air_absorption is not None:
                # The relaxed update rebuilds the divergence from the pressure levels, which a
                # driven face breaks; the combination is refused rather than silently wrong.
                raise ValueError("velocity sources cannot be combined with air_absorption here")
            lower = self._lower_cell(cell, axis)
            if self._cpml and (self._shell[cell] or (lower is not None and self._shell[lower])):
                raise ValueError(
                    "a velocity source on a face bordering the absorbing shell is not supported "
                    "(no reference has a CPML plus a driven face)"
                )
            self._velocity_face(axis, cell).windows.append((start, samples))
        else:
            raise ValueError(f"kind must be 'pressure' or 'velocity', got {kind!r}")

    @staticmethod
    def _lower_cell(cell: tuple[int, int], axis: int) -> tuple[int, int] | None:
        """The cell below ``cell`` along ``axis``, or ``None`` on a wall line."""
        if cell[axis] == 0:
            return None
        below = list(cell)
        below[axis] -= 1
        return (below[0], below[1])

    def _wall_coefficient(self, cell: tuple[int, int]) -> np.floating:
        """The wall coefficient ``a`` of ``cell``: zero off the wall ring.

        The wall list is short (the ring), so a linear search is cheaper than any cell-sized table,
        which this scheme does not keep for the faces.
        """
        flat = cell[0] * self.grid.ny + cell[1]
        hits = np.flatnonzero(self._wall_indices == flat)
        return self._wall_a[hits[0]] if hits.size else self.dtype.type(0.0)

    def _velocity_face(self, axis: int, cell: tuple[int, int]) -> _VelocityFace:
        """The driven face below ``cell`` along ``axis``, created at rest on first use."""
        key = (axis, cell)
        face = self._faces.get(key)
        if face is None:
            if self.step_index > 0:
                # A new face starts at v = 0, true only from rest: once the field has moved, the
                # face's v^{n-1/2} is not recoverable from the two pressure levels.
                raise ValueError(
                    "a velocity source on a new face must be added before the first step"
                )
            lower = self._lower_cell(cell, axis)
            closed = False
            a_lower = self.dtype.type(0.0)
            if lower is not None:
                a_lower = self._wall_coefficient(lower)
                if self._rigid_cells is not None:
                    closed = bool(self._rigid_cells[cell] or self._rigid_cells[lower])
            face = _VelocityFace(
                cell,
                lower,
                closed,
                self._wall_coefficient(cell),
                a_lower,
                self.dtype.type(0.0),
            )
            self._faces[key] = face
        return face

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

    def _update_velocity_faces(self, p: np.ndarray, p_new: np.ndarray) -> None:
        """Fold every driven face's momentum step and its prescribed value into ``p_new``.

        The pressure-only update already contains the momentum step of every face, as if nothing
        drove it: ``p^{n+1} = p^n - K div(v_free)``, with ``v_free`` the face's momentum step from
        ``v^{n-1/2}`` and ``p^n``, and ``K = rho c^2 dt / dx``. If the face is instead held at
        ``new``, the cell above it (the one whose lower face it is) gains ``K (new - free)`` and the
        cell below loses as much. A locally-reacting wall cell solves ``(1 + a) p^{n+1} = ...``, so
        its share of the correction is divided by ``1 + a``, the correction entering before that
        solve. A wall line has no momentum step: ``v_free`` is just the stored value, which is held.

        The stored value becomes ``new``, which is what the next step's momentum step starts from
        and what a velocity receiver on this face must read. Faces are visited in insertion order,
        and the compiled kernel applies them in the same order so the two round identically.
        """
        if not self._faces:
            return
        step = self.step_index
        for face in self._faces.values():
            if face.closed:
                face.value = self.dtype.type(0.0)
                continue
            cell = face.cell
            if face.lower is None:
                free = face.value
            else:
                free = face.value - self._momentum_coefficient * (p[cell] - p[face.lower])
            driven = self._driven_value(face, step)
            new = free if driven is None else driven
            delta = new - free
            face.value = new
            amount = self._pressure_coefficient * delta
            p_new[cell] += amount / (1.0 + face.a_cell)
            if face.lower is not None:
                p_new[face.lower] -= amount / (1.0 + face.a_lower)

    @staticmethod
    def _driven_value(face: _VelocityFace, step: int) -> np.floating | None:
        """The prescribed value of ``face`` at ``step``: the last active window, or ``None``."""
        value = None
        for start, samples in face.windows:
            offset = step - start
            if 0 <= offset < samples.size:
                value = samples[offset]
        return value

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
        if self._relaxation:
            p_new = self._relaxed_update(p, lap)
        else:
            p_new = 2.0 * p - self.p_old + self._lambda_squared * lap
            self._apply_impedance_walls(p_new, p, lap)
            self._update_velocity_faces(p, p_new)
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

    def _relaxation_sum(self) -> np.ndarray:
        """``R = sum_k (dt * mean weight)_k psi_k``: the pressure offset the states contribute."""
        offset = np.zeros_like(self.p)
        for state, _, weight, _ in self._relaxation:
            offset += weight * state
        return offset

    def _relaxed_update(self, p: np.ndarray, lap: np.ndarray) -> np.ndarray:
        """``p^{n+1}`` with air relaxation, holding the states one step behind (``psi^{n-1}``).

        The first-order step is, with ``D`` the unscaled divergence sum it feeds the states,
        ``R^n = sum_k m_k psi_k^n`` and ``s^n`` the source increment added after the wall solve:

            (1 + a) (p^{n+1} - s^n) = p^n + R^n - alpha D^{n+1/2} - a p^n
            psi_k^{n+1} = d_k (psi_k^n - g_k D^{n+1/2}) + g_k D^{n+1/2}
            D^{n+1/2} = D^{n-1/2} - (dt / (rho dx)) L(p^n)

        So the previous step's own relation, solved for ``D^{n-1/2}``, rebuilds it from the two
        pressure levels and ``R^{n-1}``; that is why the states lag one step: ``psi^{n-1}`` is
        exactly what the rebuild needs, and it is then advanced to ``psi^n`` before use. Exact in
        exact arithmetic; the operations are reordered relative to the first-order scheme, so it
        agrees to round-off, not to the bit. Solid cells have all faces closed: ``L = 0`` and
        ``p = 0`` there, so their divergence and states stay zero. A prescribed pressure node
        breaks the previous step's relation, so it is not supported together with air.
        """
        alpha = self._relaxed_coefficient
        step = self.step_index
        if step == 0:
            # Nothing to rebuild yet: the states already hold psi^0 = 0, and D^{-1/2} is the
            # start's own divergence, which set_initial_state encodes as p^{-1} = p^0 + K D
            # (zero for a start at rest).
            divergence = (self.p_old - p) / self._pressure_coefficient
        else:
            # Undo the previous step's source increment and its wall solve, then divide alpha out.
            unsourced = p.copy()
            for index, signal in self._sources:
                if step - 1 < len(signal):
                    unsourced[index] -= self._source_coefficient * signal[step - 1]
            previous_offset = self._relaxation_sum()
            divergence = self.p_old + previous_offset - unsourced
            if self._wall_indices.size:
                idx = self._wall_indices
                a = self._wall_a
                flat = divergence.reshape(-1)
                flat[idx] = (
                    (1.0 - a) * self.p_old.reshape(-1)[idx]
                    + previous_offset.reshape(-1)[idx]
                    - (1.0 + a) * unsourced.reshape(-1)[idx]
                )
            divergence /= alpha
            # psi^{n-1} -> psi^n, in the first-order scheme's exact exponential form.
            for state, decay, _, gain in self._relaxation:
                target = gain * divergence
                np.subtract(state, target, out=state)
                state *= decay
                state += target

        # The eliminated momentum step, then the first-order pressure update and wall solve.
        divergence -= self._momentum_coefficient * lap
        p_new = p + self._relaxation_sum() - alpha * divergence
        if self._wall_indices.size:
            idx = self._wall_indices
            a = self._wall_a
            flat_new = p_new.reshape(-1)
            flat_new[idx] = (flat_new[idx] - a * p.reshape(-1)[idx]) / (1.0 + a)
        # The full sample, not the difference: the divergence above carries no source.
        for index, signal in self._sources:
            if step < len(signal):
                p_new[index] += self._source_coefficient * signal[step]
        return p_new

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
            face = self._faces.get((axis, index))
            if face is not None:
                # A driven face has one value, the stored one: the receiver must read what the
                # reference's velocity array holds there, driven or released, not its own estimate.
                samples.append(face.value)
                continue
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
