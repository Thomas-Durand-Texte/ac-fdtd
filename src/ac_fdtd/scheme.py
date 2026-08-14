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

import numpy as np

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
    """Lossless propagation in a rigid rectangular box.

    Rigid walls are enforced structurally rather than by a boundary routine: the velocity
    planes that lie on the walls are never written to, so they stay at the zero they were
    allocated with. That is both the fastest and the least fragile way to say ``v.n = 0``, and
    it is what makes the energy identity hold exactly.

    Args:
        grid: Geometry and spacing.
        medium: Fluid properties. Defaults to air at 20 °C.
        courant: Fraction of the stability limit to run at. The default of 1.0 is the limit
            itself, which is also where numerical dispersion is smallest; lower values are for
            deliberate experiments, not for safety.
        dtype: Working precision. Double by default — this class is the reference the fast
            backends are checked against, so it should not be the one making approximations.
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")

        self.grid = grid
        self.medium = medium
        self.courant = courant
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
        self._update_pressure()
        self._inject_sources()
        self.step_index += 1

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    def _update_velocity(self) -> None:
        for axis in _AXES:
            self.velocity[axis][_interior_faces(axis)] -= self._velocity_coefficient * (
                self.p[_upper(axis)] - self.p[_lower(axis)]
            )

    def _update_pressure(self) -> None:
        divergence = self._divergence
        np.subtract(self.velocity[0][_upper(0)], self.velocity[0][_lower(0)], out=divergence)
        for axis in (1, 2):
            component = self.velocity[axis]
            np.subtract(component[_upper(axis)], component[_lower(axis)], out=self._scratch)
            divergence += self._scratch
        divergence *= self._pressure_coefficient
        self.p -= divergence

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
