"""The staggered pressure–velocity leapfrog in 2D: the plane peer of :mod:`ac_fdtd.scheme`.

This is :class:`ac_fdtd.scheme.AcousticFDTD` with one fewer axis. The scheme, the stored-state
invariant ``(p^n, v^{n-1/2})``, the time-centred wall solve, the matched layer and the air
relaxation are all derived and justified in that module and in :mod:`ac_fdtd.boundaries`; the prose
here is kept to what is genuinely different in 2D so the two files can be read against each other:

* two velocity components instead of three, and a cell *area* ``dx**2`` in the energy;
* the Courant limit is ``dx/(c*sqrt(2))`` (see :func:`ac_fdtd.grid2d.max_stable_time_step_2d`);
* everything else is the same relation summed over two axes.

The energy identity is as exact here as in 3D, and :meth:`AcousticFDTD2D.energy` is the same
sharpest-available test: for rigid walls it holds to round-off, and a sign error or an off-by-one
slice shows up as drift.
"""

from __future__ import annotations

import math

import numpy as np

from .air import AirAbsorption
from .boundaries2d import (
    AbsorbingLayer2D,
    WallAdmittances2D,
    edge_slabs_2d,
    layer_factors_2d,
    wall_update_coefficients_2d,
)
from .grid2d import Grid2D, max_stable_time_step_2d
from .medium import AIR, Medium

__all__ = ["AcousticFDTD2D"]

_AXES = (0, 1)


def _upper(axis: int) -> tuple[slice, ...]:
    """Cells from index 1 upwards along ``axis`` — the far side of each interior face."""
    return tuple(slice(1, None) if a == axis else slice(None) for a in _AXES)


def _lower(axis: int) -> tuple[slice, ...]:
    """Cells up to index -1 along ``axis`` — the near side of each interior face."""
    return tuple(slice(None, -1) if a == axis else slice(None) for a in _AXES)


def _interior_faces(axis: int) -> tuple[slice, ...]:
    """Faces of ``axis`` excluding the two boundary lines, which rigid walls hold at zero."""
    return tuple(slice(1, -1) if a == axis else slice(None) for a in _AXES)


class AcousticFDTD2D:
    """Propagation in a rectangular 2D domain, rigid by default.

    Rigid walls are structural: the velocity lines on the walls are never written to, so they stay
    at zero, which is both the fastest way to say ``v.n = 0`` and what makes the energy identity
    exact. Absorbing walls keep that arrangement and add the flux through the wall to the pressure
    update of the boundary cells. See :mod:`ac_fdtd.scheme` and :mod:`ac_fdtd.boundaries` for why
    these forms, and not the obvious ones, are the stable ones.

    Args:
        grid: Geometry and spacing.
        medium: Fluid properties. Defaults to air at 20 °C.
        courant: Fraction of the stability limit to run at; 1.0 (the default) is the limit, which
            is also where numerical dispersion is smallest.
        dtype: Working precision. Double by default — this is a reference, not a fast backend.
        walls: Normalised admittance of each of the four walls. Default is rigid throughout.
        absorbing_layer: Graded matched layer for free-field runs; combines with ``walls``.
        air_absorption: ISO-9613-1 atmospheric absorption, or ``None`` for lossless air.
        rigid_cells: Boolean mask of grid shape marking solid interior cells (rigid obstacles).
            ``None`` (default) is an empty domain. Every velocity face touching a solid cell is held
            at zero (``v.n = 0``), so obstacles are lossless rigid reflectors of any shape and the
            energy identity still holds; the geometry that produces the mask is the caller's job.
    """

    def __init__(
        self,
        grid: Grid2D,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
        walls: WallAdmittances2D | None = None,
        absorbing_layer: AbsorbingLayer2D | None = None,
        air_absorption: AirAbsorption | None = None,
        rigid_cells: np.ndarray | None = None,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")

        self.grid = grid
        self.medium = medium
        self.courant = courant
        self.walls = walls or WallAdmittances2D()
        self.absorbing_layer = absorbing_layer
        self.air_absorption = air_absorption
        self.dtype = np.dtype(dtype)
        self.dt = courant * max_stable_time_step_2d(grid.dx, medium.sound_speed)

        self.p = np.zeros(grid.shape, dtype=self.dtype)
        self.velocity = [np.zeros(grid.velocity_shape(axis), dtype=self.dtype) for axis in _AXES]

        #: Number of completed steps. The state is (p^step_index, v^{step_index - 1/2}).
        self.step_index = 0

        self._velocity_coefficient = self.dtype.type(self.dt / (medium.density * grid.dx))
        self._pressure_coefficient = self.dtype.type(
            medium.density * medium.sound_speed**2 * self.dt / grid.dx
        )
        self._source_coefficient = self.dtype.type(medium.density * medium.sound_speed**2 * self.dt)

        self._divergence = np.empty(grid.shape, dtype=self.dtype)
        self._scratch = np.empty(grid.shape, dtype=self.dtype)

        self._sources: list[tuple[tuple[int, int], np.ndarray]] = []
        self._receivers: list[tuple[tuple[int, int], list[float]]] = []
        self._prescribed_velocity: list[tuple[int, tuple[int, int], int, np.ndarray]] = []
        self._prescribed_pressure: list[tuple[tuple[int, int], int, np.ndarray]] = []

        indices, coefficient = wall_update_coefficients_2d(grid, medium, self.dt, self.walls)
        self._wall_indices = indices
        self._wall_from_updated = (1.0 / (1.0 + coefficient)).astype(self.dtype)
        self._wall_from_previous = (coefficient / (1.0 + coefficient)).astype(self.dtype)

        # Air absorption: one auxiliary field per relaxation process, integrated exactly over the
        # step (the derivation and the reason for the exponential form, not the trapezoidal one,
        # are in ac_fdtd.scheme).
        self._relaxation: list[tuple[np.ndarray, float, float, float]] = []
        self._relaxation_scratch: np.ndarray | None = None
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
            self._relaxation_scratch = np.empty(grid.shape, dtype=self.dtype)
        self._relaxation_target_scale = self.dtype.type(
            self.dt * sum(gain for _, _, _, gain in self._relaxation)
        )

        # Absorbing layer: 1D factors applied only to the slabs where they differ from one.
        self._pressure_damping: list[tuple[tuple[slice, ...], np.ndarray]] = []
        self._velocity_damping: list[list[tuple[tuple[slice, ...], np.ndarray]]] = [
            [] for _ in _AXES
        ]
        if absorbing_layer is not None:
            pressure_factors, velocity_factors = layer_factors_2d(
                grid, medium, self.dt, absorbing_layer
            )
            self._pressure_damping = [
                slab
                for axis in _AXES
                for slab in edge_slabs_2d(pressure_factors[axis].astype(self.dtype), axis)
            ]
            for component in _AXES:
                self._velocity_damping[component] = [
                    slab
                    for axis in _AXES
                    for slab in edge_slabs_2d(
                        velocity_factors[component][axis].astype(self.dtype), axis
                    )
                ]

        # Rigid interior obstacles: a boolean mask of solid cells. A velocity face between a solid
        # cell and any cell is held at zero every step, which is v.n = 0 — the same thing that makes
        # the domain's own walls rigid. The face is zeroed before the pressure update reads it, so
        # the value the update put there never feeds back: it is equivalent to never updating that
        # face, so an obstacle is a lossless reflector and the energy identity is unchanged. With
        # every bounding face at zero a solid cell's divergence is zero, so its pressure stays zero.
        self._rigid_cells: np.ndarray | None = None
        self._vx_open: np.ndarray | None = None
        self._vy_open: np.ndarray | None = None
        if rigid_cells is not None:
            rigid = np.ascontiguousarray(rigid_cells, dtype=bool)
            if rigid.shape != grid.shape:
                raise ValueError(
                    f"rigid_cells shape {rigid.shape} does not match the grid {grid.shape}"
                )
            self._rigid_cells = rigid
            self._vx_open = np.ones(grid.velocity_shape(0), dtype=self.dtype)
            self._vx_open[1:-1, :] = ~(rigid[:-1, :] | rigid[1:, :])
            self._vy_open = np.ones(grid.velocity_shape(1), dtype=self.dtype)
            self._vy_open[:, 1:-1] = ~(rigid[:, :-1] | rigid[:, 1:])

    @property
    def vx(self) -> np.ndarray:
        return self.velocity[0]

    @property
    def vy(self) -> np.ndarray:
        return self.velocity[1]

    @property
    def time(self) -> float:
        """Instant the stored pressure field belongs to, in seconds."""
        return self.step_index * self.dt

    def add_volume_source(self, point: tuple[float, float], signal: np.ndarray) -> None:
        """Inject a soft volume-velocity source into the cell containing ``point``.

        ``signal`` is a volume velocity per unit area (1/s) here, one sample per step; sample
        ``n`` acts on the step from ``p^n`` to ``p^{n+1}``, centred at ``(n+1/2) dt`` — the same
        instant as the velocity field, which keeps it from adding a half-step timing error. Soft:
        it adds to the field rather than overwriting it, so it does not scatter returning waves.
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
        """Hold one node equal to ``signal`` for a window of steps, then let it go.

        The hard counterpart to :meth:`add_volume_source`, with identical semantics and timing to
        :meth:`ac_fdtd.scheme.AcousticFDTD.add_prescribed_source`: inside ``[start, start+len)`` the
        node is overwritten after its update (velocity before the pressure update that reads it,
        pressure last), and outside it the equations take over. A velocity source on a wall line the
        interior update never touches persists as set — the driven-wall / piston case, which is how
        a 2D "loudspeaker" line is built.
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        samples = np.asarray(signal, dtype=self.dtype)
        index = self.grid.cell_index(point)
        if kind == "pressure":
            if axis is not None:
                raise ValueError("axis is meaningless for a pressure source; leave it None")
            self._prescribed_pressure.append((index, start, samples))
        elif kind == "velocity":
            if axis not in (0, 1):
                raise ValueError(f"velocity source needs axis in (0, 1), got {axis!r}")
            self._prescribed_velocity.append((axis, index, start, samples))
        else:
            raise ValueError(f"kind must be 'pressure' or 'velocity', got {kind!r}")

    def add_pressure_receiver(self, point: tuple[float, float]) -> int:
        """Record the pressure at ``point`` every step from now on, and return its channel.

        Sample ``n`` is the pressure at ``(n+1) dt``. Receivers added part-way through a run are
        legitimately shorter than the others; padding them would put recordings on different time
        bases without saying so.
        """
        self._receivers.append((self.grid.cell_index(point), []))
        return len(self._receivers) - 1

    @property
    def recorded_pressure(self) -> np.ndarray:
        """Everything the receivers have recorded, as ``(channel, sample)``."""
        if not self._receivers:
            return np.zeros((0, 0))
        return np.array([samples for _, samples in self._receivers])

    @property
    def sample_rate(self) -> float:
        """Rate the recordings are at, in Hz. Set by the Courant condition, not by choice."""
        return 1.0 / self.dt

    def probe_pressure(self, point: tuple[float, float]) -> float:
        """Pressure at ``point``, at time :attr:`time`, in Pa."""
        return float(self.p[self.grid.cell_index(point)])

    def probe_velocity(self, point: tuple[float, float], axis: int) -> float:
        """One velocity component on the lower face of the cell containing ``point``, in m/s.

        As in 3D this is the field at ``time - dt/2``: velocity is stored half a step behind
        pressure. Averaging with the next step's value recentres it on ``time``.
        """
        return float(self.velocity[axis][self.grid.cell_index(point)])

    def step(self) -> None:
        """Advance one time step, preserving the ``(p^n, v^{n-1/2})`` pairing."""
        self._update_velocity()
        self._damp(self.velocity[0], self._velocity_damping[0])
        self._damp(self.velocity[1], self._velocity_damping[1])
        self._apply_prescribed_velocity()
        if self._rigid_cells is not None:
            self._apply_rigid_velocity()
        self._update_pressure()
        self._damp(self.p, self._pressure_damping)
        self._inject_sources()
        self._apply_prescribed_pressure()
        if self._rigid_cells is not None:
            self.p[self._rigid_cells] = 0.0
        self.step_index += 1
        for index, samples in self._receivers:
            samples.append(self.p[index])

    @staticmethod
    def _damp(field: np.ndarray, slabs: list[tuple[tuple[slice, ...], np.ndarray]]) -> None:
        for index, factor in slabs:
            field[index] *= factor

    def _apply_rigid_velocity(self) -> None:
        """Hold the velocity at zero on every face touching a solid cell (``v.n = 0``)."""
        self.velocity[0] *= self._vx_open
        self.velocity[1] *= self._vy_open

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    def _update_velocity(self) -> None:
        for axis in _AXES:
            self.velocity[axis][_interior_faces(axis)] -= self._velocity_coefficient * (
                self.p[_upper(axis)] - self.p[_lower(axis)]
            )

    def _update_pressure(self) -> None:
        flat = self.p.reshape(-1)
        previous_at_walls = flat[self._wall_indices].copy() if self._wall_indices.size else None

        divergence = self._divergence
        np.subtract(self.velocity[0][_upper(0)], self.velocity[0][_lower(0)], out=divergence)
        component = self.velocity[1]
        np.subtract(component[_upper(1)], component[_lower(1)], out=self._scratch)
        divergence += self._scratch
        self._advance_relaxation(divergence)

        divergence *= self._pressure_coefficient
        self.p -= divergence

        if previous_at_walls is not None:
            flat[self._wall_indices] = (
                flat[self._wall_indices] * self._wall_from_updated
                - previous_at_walls * self._wall_from_previous
            )

    def _advance_relaxation(self, divergence: np.ndarray) -> None:
        """Step each relaxation state and add its contribution; takes the unscaled divergence."""
        target = self._scratch
        offset = self._relaxation_scratch
        for state, decay, mean_weight, gain in self._relaxation:
            np.multiply(divergence, gain, out=target)
            np.subtract(state, target, out=offset)

            np.multiply(offset, decay, out=state)
            state += target

            np.multiply(offset, mean_weight, out=offset)
            self.p += offset

        if self._relaxation:
            np.multiply(divergence, self._relaxation_target_scale, out=target)
            self.p += target

    def _inject_sources(self) -> None:
        for index, signal in self._sources:
            if self.step_index < len(signal):
                self.p[index] += self._source_coefficient * signal[self.step_index]

    def _apply_prescribed_velocity(self) -> None:
        step = self.step_index
        for axis, index, start, signal in self._prescribed_velocity:
            offset = step - start
            if 0 <= offset < signal.size:
                self.velocity[axis][index] = signal[offset]

    def _apply_prescribed_pressure(self) -> None:
        step = self.step_index
        for index, start, signal in self._prescribed_pressure:
            offset = step - start
            if 0 <= offset < signal.size:
                self.p[index] = signal[offset]

    def energy(self) -> float:
        """Total acoustic energy of the stored state, in joules per metre of depth.

        Written for the ``(p^n, v^{n-1/2})`` pairing, with the mixed-time coupling correction kept
        (dropping it is the classic way to get an energy that wobbles at O(dt) and looks physical).
        The cell measure is an area ``dx**2``, so this is an energy per unit out-of-plane length.
        """
        medium = self.medium
        cell_area = self.grid.dx**2

        potential = float(np.dot(self.p.ravel(), self.p.ravel())) / (
            2.0 * medium.density * medium.sound_speed**2
        )

        kinetic = 0.0
        coupling = 0.0
        for axis in _AXES:
            component = self.velocity[axis]
            flat = component.ravel()
            kinetic += 0.5 * medium.density * float(np.dot(flat, flat))
            gradient = (self.p[_upper(axis)] - self.p[_lower(axis)]) / self.grid.dx
            coupling += float(np.sum(component[_interior_faces(axis)] * gradient))

        return cell_area * (potential + kinetic - 0.5 * self.dt * coupling)
