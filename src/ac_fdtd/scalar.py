"""The pressure-only ("scalar", three-level) 3D scheme: the low-memory peer of the first-order box.

(:mod:`ac_fdtd.scheme`.)

This is :class:`ac_fdtd.scalar2d.ScalarFDTD2D` with one more axis, and the reasoning — why the
second-order wave equation on pressure alone is the *same* scheme as the staggered
pressure–velocity leapfrog, in two cell arrays instead of four (half the memory in 3D), with every
boundary re-derived as a three-level relation — is written there and in ``docs/kernel_study``. The
prose here is kept to what three axes actually change:

* four velocity components' worth of storage become none in the interior (the whole point in 3D);
* the Neumann Laplacian, the impedance-wall correction and the CFS-CPML shell all sum over three
  axes and six faces instead of two and four;
* the rigid-obstacle mask is a genuinely 3D feature here — the first-order 3D reference has none.

For any run driven from silence it reproduces :class:`ac_fdtd.scheme.AcousticFDTD` to the last bit
(``tests/test_scalar.py``), which is the sharpest check that the elimination and the boundary
derivations are exact and not merely close.
"""

from __future__ import annotations

import math

import numpy as np

from .air import AirAbsorption
from .boundaries import WallAdmittances, wall_update_coefficients
from .cpml import AbsorbingBoundary, cpml_axis_profiles_3d
from .grid import Grid, max_stable_time_step
from .medium import AIR, Medium

__all__ = ["ScalarFDTD"]

_AXES = (0, 1, 2)


def _upper(axis: int) -> tuple[slice, ...]:
    return tuple(slice(1, None) if a == axis else slice(None) for a in _AXES)


def _lower(axis: int) -> tuple[slice, ...]:
    return tuple(slice(None, -1) if a == axis else slice(None) for a in _AXES)


def _interior_faces(axis: int) -> tuple[slice, ...]:
    return tuple(slice(1, -1) if a == axis else slice(None) for a in _AXES)


def _broadcast(values: np.ndarray, axis: int) -> np.ndarray:
    """Reshape a 1D per-axis profile so it broadcasts along ``axis`` of the 3D field."""
    shape = [1, 1, 1]
    shape[axis] = -1
    return values.reshape(shape)


class ScalarFDTD:
    """Pressure-only propagation in a rectangular box, rigid by default.

    The constructor mirrors :class:`ac_fdtd.scheme.AcousticFDTD` for the features they share, plus a
    rigid-obstacle mask and a CFS-CPML open boundary. The state is the pair ``(p^n, p^{n-1})``.

    Args:
        grid: Geometry and spacing.
        medium: Fluid properties. Defaults to air at 20 degrees C.
        courant: Fraction of the stability limit; 1.0 (default) is the limit and least dispersive.
        dtype: Working precision. Double by default — this is a reference, not a fast backend.
        walls: Normalised admittance of each of the six walls. Default is rigid throughout.
        rigid_cells: Boolean mask of grid shape marking solid interior cells (rigid obstacles).
        absorbing_boundary: A CFS-CPML region lining a subset of the walls (open field, or a tuned
            partial absorber). When present the scheme stores velocity only in the boundary shell;
            a closed room holds two cell arrays and nothing else.
        air_absorption: Atmospheric absorption (ISO 9613-1) with the first-order scheme's semantics
            and constants: one cell array per relaxation process, driven by a divergence rebuilt
            from the two pressure levels (see :meth:`_relaxed_update`), so no velocity grid. Not
            combinable with ``absorbing_boundary``: the CPML shell has no relaxation states.
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
        walls: WallAdmittances | None = None,
        rigid_cells: np.ndarray | None = None,
        absorbing_boundary: AbsorbingBoundary | None = None,
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
        self.walls = walls or WallAdmittances()
        self.dtype = np.dtype(dtype)
        self.dt = courant * max_stable_time_step(grid.dx, medium.sound_speed)
        self.step_index = 0

        self.p = np.zeros(grid.shape, dtype=self.dtype)
        self.p_old = np.zeros(grid.shape, dtype=self.dtype)
        self._lambda_squared = self.dtype.type((medium.sound_speed * self.dt / grid.dx) ** 2)
        self._source_coefficient = self.dtype.type(medium.density * medium.sound_speed**2 * self.dt)

        indices, coefficient = wall_update_coefficients(grid, medium, self.dt, self.walls)
        self._wall_indices = indices
        self._wall_a = coefficient.astype(self.dtype)

        self._rigid_cells: np.ndarray | None = None
        self._open: list[np.ndarray | None] = [None, None, None]
        if rigid_cells is not None:
            rigid = np.ascontiguousarray(rigid_cells, dtype=bool)
            if rigid.shape != grid.shape:
                raise ValueError(
                    f"rigid_cells shape {rigid.shape} does not match the grid {grid.shape}"
                )
            self._rigid_cells = rigid
            for axis in _AXES:
                self._open[axis] = (~(rigid[_lower(axis)] | rigid[_upper(axis)])).astype(self.dtype)

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

        self._sources: list[tuple[tuple[int, ...], np.ndarray]] = []
        self._receivers: list[tuple[tuple[int, ...], list[float]]] = []
        self._velocity_receivers: list[list] = []
        self._prescribed_pressure: list[tuple[tuple[int, ...], int, np.ndarray]] = []

    def _setup_cpml(self, boundary: AbsorbingBoundary) -> None:
        grid = self.grid
        thickness = boundary.thickness
        if 2 * thickness >= min(grid.shape):
            raise ValueError(
                f"absorbing layer of {thickness} cells does not fit a {grid.shape} grid"
            )

        profiles = cpml_axis_profiles_3d(grid, self.medium, self.dt, boundary)
        self._kappa_v, self._b_v, self._a_v = {}, {}, {}
        self._kappa_p, self._b_p, self._a_p = {}, {}, {}
        for axis in _AXES:
            kv, bv, av = (c.astype(self.dtype) for c in profiles[("v", axis)])
            kp, bp, ap = (c.astype(self.dtype) for c in profiles[("p", axis)])
            self._kappa_v[axis], self._b_v[axis], self._a_v[axis] = kv, bv, av
            self._kappa_p[axis], self._b_p[axis], self._a_p[axis] = kp, bp, ap

        self._inverse_dx = self.dtype.type(1.0 / grid.dx)
        self._cv = self.dtype.type(self.dt / self.medium.density)
        self._cp = self.dtype.type(self.medium.density * self.medium.sound_speed**2 * self.dt)

        self._velocity = [np.zeros(grid.velocity_shape(a), dtype=self.dtype) for a in _AXES]
        self._psi_v = [np.zeros(grid.velocity_shape(a), dtype=self.dtype) for a in _AXES]
        self._psi_p = [np.zeros(grid.shape, dtype=self.dtype) for _ in _AXES]

        # shell = cells inside any lined layer
        in_axis = []
        for axis in _AXES:
            marker = np.zeros(grid.shape[axis], dtype=bool)
            low, high = boundary.has_axis(axis)
            if low:
                marker[:thickness] = True
            if high:
                marker[grid.shape[axis] - thickness :] = True
            in_axis.append(marker)
        shell = np.zeros(grid.shape, dtype=bool)
        for axis in _AXES:
            shell |= _broadcast(in_axis[axis], axis)
        self._shell = shell
        self._interior = ~shell

        # a velocity face is needed when either cell it borders is a shell cell
        self._need: list[np.ndarray] = []
        for axis in _AXES:
            need = np.zeros(grid.velocity_shape(axis), dtype=bool)
            need[_interior_faces(axis)] = shell[_lower(axis)] | shell[_upper(axis)]
            self._need.append(need)

        if self._wall_indices.size:
            keep = self._interior.reshape(-1)[self._wall_indices]
            self._wall_indices = self._wall_indices[keep]
            self._wall_a = self._wall_a[keep]

    # -- time and probes -------------------------------------------------------------------
    @property
    def time(self) -> float:
        return self.step_index * self.dt

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt

    def probe_pressure(self, point: tuple[float, float, float]) -> float:
        return float(self.p[self.grid.cell_index(point)])

    def add_velocity_receiver(self, point: tuple[float, float, float], axis: int) -> int:
        """Record ``v^{n-1/2}`` on the lower face of ``point`` along ``axis`` by reconstruction.

        One running value per probe (``v^{n+1/2} = v^{n-1/2} - (dt/(rho dx))(p_here - p_lower)``),
        O(1) not a grid; half a step behind the pressure, exactly as the first-order probe is.
        """
        index = self.grid.cell_index(point)
        self._velocity_receivers.append([index, axis, 0.0, []])
        return len(self._velocity_receivers) - 1

    @property
    def recorded_velocity(self) -> np.ndarray:
        if not self._velocity_receivers:
            return np.zeros((0, 0))
        return np.array([samples for _, _, _, samples in self._velocity_receivers])

    # -- construction API ------------------------------------------------------------------
    def set_initial_pressure(self, field: np.ndarray) -> None:
        self.p[:] = field
        self.p_old[:] = field

    def set_initial_state(self, pressure: np.ndarray, velocity: list[np.ndarray]) -> None:
        """Seed ``(p^0, p^{-1})`` from a first-order pair ``(p^0, v^{-1/2})``: the pressure update
        run backwards one step, ``p^{-1} = p^0 + rho c^2 dt * div(v^{-1/2})``."""
        self.p[:] = pressure
        coefficient = self.medium.density * self.medium.sound_speed**2 * self.dt / self.grid.dx
        divergence = np.zeros(self.grid.shape, dtype=self.dtype)
        for axis in _AXES:
            divergence += velocity[axis][_upper(axis)] - velocity[axis][_lower(axis)]
        self.p_old[:] = pressure + coefficient * divergence

    def add_volume_source(self, point: tuple[float, float, float], signal: np.ndarray) -> None:
        index = self.grid.cell_index(point)
        self._sources.append((index, np.asarray(signal, dtype=self.dtype)))

    def add_prescribed_source(
        self,
        point: tuple[float, float, float],
        signal: np.ndarray,
        start: int = 0,
        kind: str = "pressure",
    ) -> None:
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if kind != "pressure":
            raise ValueError(
                f"the pressure-only scheme supports kind='pressure' only here, got {kind!r}"
            )
        samples = np.asarray(signal, dtype=self.dtype)
        self._prescribed_pressure.append((self.grid.cell_index(point), start, samples))

    def add_pressure_receiver(self, point: tuple[float, float, float]) -> int:
        self._receivers.append((self.grid.cell_index(point), []))
        return len(self._receivers) - 1

    @property
    def recorded_pressure(self) -> np.ndarray:
        if not self._receivers:
            return np.zeros((0, 0))
        return np.array([samples for _, samples in self._receivers])

    # -- the kernel ------------------------------------------------------------------------
    def _laplacian(self, p: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(p)
        for axis in _AXES:
            flux = p[_upper(axis)] - p[_lower(axis)]
            if self._open[axis] is not None:
                flux = flux * self._open[axis]
            lap[_lower(axis)] += flux
            lap[_upper(axis)] -= flux
        return lap

    def _apply_impedance_walls(self, p_new: np.ndarray, p: np.ndarray, lap: np.ndarray) -> None:
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
        inverse_dx = self._inverse_dx
        for axis in _AXES:
            faces = _interior_faces(axis)
            gradient = (p[_upper(axis)] - p[_lower(axis)]) * inverse_dx
            mask = self._need[axis][faces]
            kappa = _broadcast(self._kappa_v[axis][1:-1], axis)
            b = _broadcast(self._b_v[axis][1:-1], axis)
            a = _broadcast(self._a_v[axis][1:-1], axis)
            psi = self._psi_v[axis]
            psi[faces] = np.where(mask, b * psi[faces] + a * gradient, psi[faces])
            update = self._cv * (gradient / kappa + psi[faces])
            self._velocity[axis][faces] = np.where(
                mask, self._velocity[axis][faces] - update, self._velocity[axis][faces]
            )

        divergence_term = np.zeros_like(p)
        for axis in _AXES:
            velocity = self._velocity[axis]
            divergence = (velocity[_upper(axis)] - velocity[_lower(axis)]) * inverse_dx
            b = _broadcast(self._b_p[axis], axis)
            a = _broadcast(self._a_p[axis], axis)
            self._psi_p[axis] = b * self._psi_p[axis] + a * divergence
            kappa = _broadcast(self._kappa_p[axis], axis)
            divergence_term += divergence / kappa + self._psi_p[axis]
        p_shell = p - self._cp * divergence_term
        p_new[self._shell] = p_shell[self._shell]

    def step(self) -> None:
        p = self.p
        self._record_velocity(p)

        lap = self._laplacian(p)
        if self._relaxation:
            p_new = self._relaxed_update(p, lap)
        else:
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
