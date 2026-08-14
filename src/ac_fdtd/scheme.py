"""The staggered pressure–velocity leapfrog, as a readable NumPy reference.

The scheme
----------
Pressure sits at cell centres and each velocity component sits on the faces normal to it
(the standard Yee arrangement). One step is two half-updates::

    v^{n+1/2} = v^{n-1/2} - (dt/rho) * grad(p^n)
    p^{n+1}   = p^n - rho*c^2*dt * div(v^{n+1/2})

**The half step between the two fields is real and is not an implementation detail.** Pressure
is known at whole steps and velocity at half steps, so anything that combines them — intensity,
energy, a velocity probe compared against a pressure probe — has to say which instant it means.
Two consequences are wired into this module rather than left to the caller:

* The stored state is *always* the pair ``(p^n, v^{n-1/2})``: pressure one half step ahead of
  velocity. :meth:`AcousticFDTD.step` preserves that invariant, and every diagnostic below is
  written for that pairing. Getting this wrong does not blow up, it just makes energy drift by
  O(dt) and quietly ruins the very test that was supposed to catch bugs.
* A velocity probe therefore reports a quantity offset by ``-dt/2`` from a pressure probe, and
  :meth:`AcousticFDTD.probe_velocity` documents that instead of hiding it.

Why this scheme rather than a pressure-only one
-----------------------------------------------
The second-order wave equation on pressure alone needs two fields instead of four and, on a
memory-bandwidth-bound 3D problem, that is close to a factor two in run time. It is kept as a
planned alternative backend for exactly that reason. What the p–v form buys is the boundary
treatment: impedance walls, PML layers and the air relaxation states all attach naturally to a
first-order system, and all three are the point of this repository.

Energy
------
For rigid walls the discrete divergence and gradient are exact negative adjoints of each other,
which makes the quadratic form

    H = sum[ p^2 / (2*rho*c^2) ] + sum[ (rho/2) * v^{n-1/2}.v^{n+1/2} ]

conserved to round-off — not approximately, exactly, for as long as the run lasts. Expressed in
the stored state it becomes the form evaluated by :meth:`AcousticFDTD.energy`. It is the
sharpest test in the repository: a sign error, an off-by-one in a slice, or a boundary face
that is updated when it should be held at zero all show up as drift, and none of them
necessarily show up in a picture of the field.
"""

from __future__ import annotations

import math

import numpy as np

from .air import AirAbsorption
from .boundaries import (
    AbsorbingLayer,
    WallAdmittances,
    edge_slabs,
    layer_factors,
    wall_update_coefficients,
)
from .grid import Grid, max_stable_time_step
from .medium import AIR, Medium

__all__ = ["AcousticFDTD"]

_AXES = (0, 1, 2)


def _upper(axis: int) -> tuple[slice, ...]:
    """Cells from index 1 upwards along ``axis`` — the far side of each interior face."""
    return tuple(slice(1, None) if a == axis else slice(None) for a in _AXES)


def _lower(axis: int) -> tuple[slice, ...]:
    """Cells up to index -1 along ``axis`` — the near side of each interior face."""
    return tuple(slice(None, -1) if a == axis else slice(None) for a in _AXES)


def _interior_faces(axis: int) -> tuple[slice, ...]:
    """Faces of ``axis`` excluding the two boundary planes, which rigid walls hold at zero."""
    return tuple(slice(1, -1) if a == axis else slice(None) for a in _AXES)


class AcousticFDTD:
    """Propagation in a rectangular box, rigid by default.

    Rigid walls are enforced structurally rather than by a boundary routine: the velocity
    planes that lie on the walls are never written to, so they stay at the zero they were
    allocated with. That is both the fastest and the least fragile way to say ``v.n = 0``, and
    it is what makes the energy identity hold exactly.

    Absorbing walls keep that arrangement — the wall velocity planes stay zero — and add the
    flux through the wall to the pressure update of the boundary cells instead. See
    :mod:`ac_fdtd.boundaries` for why that form, and not the obvious one, is the stable one.

    Args:
        grid: Geometry and spacing.
        medium: Fluid properties. Defaults to air at 20 °C.
        courant: Fraction of the stability limit to run at. The default of 1.0 is the limit
            itself, which is also where numerical dispersion is smallest; lower values are for
            deliberate experiments, not for safety.
        dtype: Working precision. Double by default — this class is the reference the fast
            backends are checked against, so it should not be the one making approximations.
        walls: Normalised admittance of each of the six walls. Default is rigid throughout.
        absorbing_layer: Graded matched layer lining the whole domain, for free-field runs.
            Combines with ``walls``, though there is rarely a reason to use both.
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
        walls: WallAdmittances | None = None,
        absorbing_layer: AbsorbingLayer | None = None,
        air_absorption: AirAbsorption | None = None,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")

        self.grid = grid
        self.medium = medium
        self.courant = courant
        self.walls = walls or WallAdmittances()
        self.absorbing_layer = absorbing_layer
        self.air_absorption = air_absorption
        self.dtype = np.dtype(dtype)
        self.dt = courant * max_stable_time_step(grid.dx, medium.sound_speed)

        self.p = np.zeros(grid.shape, dtype=self.dtype)
        self.velocity = [np.zeros(grid.velocity_shape(axis), dtype=self.dtype) for axis in _AXES]

        #: Number of completed steps. The state is (p^step_index, v^{step_index - 1/2}).
        self.step_index = 0

        self._velocity_coefficient = self.dtype.type(self.dt / (medium.density * grid.dx))
        self._pressure_coefficient = self.dtype.type(
            medium.density * medium.sound_speed**2 * self.dt / grid.dx
        )
        #: Volume-source term: converts an injected volume velocity per unit volume into the
        #: pressure increment it produces over one step.
        self._source_coefficient = self.dtype.type(medium.density * medium.sound_speed**2 * self.dt)

        # Two scratch buffers keep the pressure update free of hidden temporaries, which at
        # 10^8 cells is the difference between fitting in memory and not.
        self._divergence = np.empty(grid.shape, dtype=self.dtype)
        self._scratch = np.empty(grid.shape, dtype=self.dtype)

        self._sources: list[tuple[tuple[int, int, int], np.ndarray]] = []
        self._receivers: list[tuple[tuple[int, int, int], list[float]]] = []

        # Absorbing walls: the two coefficients of the closed-form solve, on the boundary cells
        # only. Empty arrays when every wall is rigid, and then the whole mechanism costs one
        # `if` per step.
        indices, coefficient = wall_update_coefficients(grid, medium, self.dt, self.walls)
        self._wall_indices = indices
        self._wall_from_updated = (1.0 / (1.0 + coefficient)).astype(self.dtype)
        self._wall_from_previous = (coefficient / (1.0 + coefficient)).astype(self.dtype)

        # Air absorption: one auxiliary field per relaxation process, integrated exactly over
        # the step rather than with a difference formula. The oxygen process has a relaxation
        # time of a few microseconds, which is *shorter than dt* on any ordinary grid; the
        # trapezoidal rule is stable there but rings at the Nyquist frequency, while the
        # exponential form is exact in that limit and simply pins the state to its target.
        self._relaxation: list[tuple[np.ndarray, float, float, float]] = []
        self._relaxation_scratch = None
        if air_absorption is not None:
            processes = air_absorption.processes(medium.sound_speed, self.dt)
            relaxed_speed_squared = medium.sound_speed**2 / (
                1.0 + sum(process.strength for process in processes)
            )
            for process in processes:
                decay = math.exp(-self.dt / process.relaxation_time)
                # Mean of the exact solution over the step, as a weight on the initial offset
                # from the target. Using the endpoint average instead would lag the damping by
                # half a step -- harmless here, but free to get right.
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
        # The part of every process's step-average that is proportional to the divergence is
        # the same shape for all of them, so it is summed once instead of per process.
        self._relaxation_target_scale = self.dtype.type(
            self.dt * sum(gain for _, _, _, gain in self._relaxation)
        )

        # Absorbing layer: 1D factors applied only to the slabs where they differ from one.
        self._pressure_damping = []
        self._velocity_damping = [[] for _ in _AXES]
        if absorbing_layer is not None:
            pressure_factors, velocity_factors = layer_factors(
                grid, medium, self.dt, absorbing_layer
            )
            self._pressure_damping = [
                slab
                for axis in _AXES
                for slab in edge_slabs(pressure_factors[axis].astype(self.dtype), axis)
            ]
            for component in _AXES:
                self._velocity_damping[component] = [
                    slab
                    for axis in _AXES
                    for slab in edge_slabs(
                        velocity_factors[component][axis].astype(self.dtype), axis
                    )
                ]

    @property
    def vx(self) -> np.ndarray:
        return self.velocity[0]

    @property
    def vy(self) -> np.ndarray:
        return self.velocity[1]

    @property
    def vz(self) -> np.ndarray:
        return self.velocity[2]

    @property
    def time(self) -> float:
        """Instant the stored pressure field belongs to, in seconds."""
        return self.step_index * self.dt

    def add_volume_source(self, point: tuple[float, float, float], signal: np.ndarray) -> None:
        """Inject a volume-velocity source into the cell containing ``point``.

        ``signal`` is a volume velocity per unit volume (1/s), one sample per step. Sample
        ``n`` acts on the step from ``p^n`` to ``p^{n+1}``, so it is centred at ``(n+1/2)*dt``
        — the same instant as the velocity field it competes with, which is what keeps the
        source from introducing a half-step timing error of its own.

        This is a *soft* source: it adds to the field rather than overwriting it, so it does
        not act as a scatterer for waves coming back to it. A hard source is easier to reason
        about at the instant of injection and wrong for everything after.
        """
        index = self.grid.cell_index(point)
        self._sources.append((index, np.asarray(signal, dtype=self.dtype)))

    def add_pressure_receiver(self, point: tuple[float, float, float]) -> int:
        """Record the pressure at ``point`` every step from now on, and return its channel.

        Sample ``n`` of the recording is the pressure at ``(n + 1) dt``: the field after the
        first step taken once the receiver existed. Receivers added part-way through a run are
        therefore shorter than the others, which is deliberate — silently padding them would
        put the recordings on different time bases without saying so.
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

    def probe_pressure(self, point: tuple[float, float, float]) -> float:
        """Pressure at ``point``, at time :attr:`time`, in Pa."""
        return float(self.p[self.grid.cell_index(point)])

    def probe_velocity(self, point: tuple[float, float, float], axis: int) -> float:
        """One velocity component near ``point``, in m/s.

        The face returned is the lower face of the cell containing ``point`` on that axis, so
        the sample is half a cell towards the origin.

        Note the instant too: velocity is stored half a step *behind* pressure, so this is the
        field at ``time - dt/2``. Averaging with the value from the next step recentres it on
        ``time``, which is what an intensity calculation needs.
        """
        return float(self.velocity[axis][self.grid.cell_index(point)])

    def step(self) -> None:
        """Advance one time step, preserving the ``(p^n, v^{n-1/2})`` pairing."""
        self._update_velocity()
        self._damp(self.velocity[0], self._velocity_damping[0])
        self._damp(self.velocity[1], self._velocity_damping[1])
        self._damp(self.velocity[2], self._velocity_damping[2])
        self._update_pressure()
        self._damp(self.p, self._pressure_damping)
        self._inject_sources()
        self.step_index += 1
        for index, samples in self._receivers:
            samples.append(self.p[index])

    @staticmethod
    def _damp(field: np.ndarray, slabs) -> None:
        for index, factor in slabs:
            field[index] *= factor

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
        for axis in (1, 2):
            component = self.velocity[axis]
            np.subtract(component[_upper(axis)], component[_lower(axis)], out=self._scratch)
            divergence += self._scratch
        self._advance_relaxation(divergence)

        divergence *= self._pressure_coefficient
        self.p -= divergence

        if previous_at_walls is not None:
            # The closed-form solve of the time-centred wall condition. The wall faces of the
            # velocity arrays stay at zero throughout, so the flux through the wall enters here
            # and here only — counting it twice is the one way to get this wrong.
            flat[self._wall_indices] = (
                flat[self._wall_indices] * self._wall_from_updated
                - previous_at_walls * self._wall_from_previous
            )

    def _advance_relaxation(self, divergence: np.ndarray) -> None:
        """Step each relaxation state and add its contribution to the pressure.

        Takes the *unscaled* divergence sum, before the pressure coefficient is folded in, and
        must therefore run before that multiplication rather than after it.
        """
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
            # The remaining piece of the step-average, dt * target, folded in once.
            np.multiply(divergence, self._relaxation_target_scale, out=target)
            self.p += target

    def _inject_sources(self) -> None:
        for index, signal in self._sources:
            if self.step_index < len(signal):
                self.p[index] += self._source_coefficient * signal[self.step_index]

    def energy(self) -> float:
        """Total acoustic energy of the stored state, in joules.

        Written for the ``(p^n, v^{n-1/2})`` pairing: substituting the velocity update into
        ``v^{n-1/2}.v^{n+1/2}`` turns the mixed-time product into terms of the stored state
        plus a coupling correction. Dropping that correction is the classic way to get an
        energy that wobbles at O(dt) and looks like a physics result.
        """
        medium = self.medium
        cell_volume = self.grid.dx**3

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

        return cell_volume * (potential + kinetic - 0.5 * self.dt * coupling)
