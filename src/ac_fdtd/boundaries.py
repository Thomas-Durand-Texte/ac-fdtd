"""What happens at the edges of the grid: absorbing walls, and a layer that fakes free field.

Two mechanisms, for two different jobs.

Locally reacting walls
----------------------
A real wall admittance imposes ``v.n = Y p`` at the surface. The obvious implementation — set
the wall velocity from the neighbouring cell pressure — is a trap, and it is the same trap the
rest of this codebase is built to avoid: velocity lives half a step away from pressure, so
``v^{n+1/2} = Y p^n`` is off by ``dt/2`` and leaks or injects energy depending on the sign.
Boundary instability in FDTD is nearly always this.

The fix is to centre the condition in time, ``v^{n+1/2} = Y (p^{n+1} + p^n) / 2``, which looks
implicit because ``p^{n+1}`` is what we are computing. It is implicit — but only within a single
cell, so it can be solved in closed form. Substituting into the pressure update of the boundary
cell gives

    p^{n+1} = ((1 - a) p^n - dt rho c^2 div_interior) / (1 + a),      a = c dt xi / (2 dx)

with ``xi = rho c Y`` the normalised admittance. Every absorbing face touching a cell adds its
own ``a``, so edges and corners — where two or three walls meet, and where naive schemes come
apart — need no special case at all. The resulting update is unconditionally dissipative for
any ``xi >= 0``: it can only remove energy, never add it, whatever the field does.

``xi = 0`` is a rigid wall and ``xi = 1`` is the impedance-matched wall, which is exactly the
first-order absorbing boundary condition: perfect at normal incidence, and progressively worse
as the angle opens up, because a locally reacting surface cannot know which way the wave is
travelling.

Absorbing layer
---------------
Which is why free-field simulations get a graded layer instead. Damping the two equations by
the *same* ``sigma``,

    dv/dt + sigma v = -(1/rho) grad p,    dp/dt + sigma p = -rho c^2 div v

leaves the local impedance at ``rho c``, so the layer attenuates without reflecting off its own
gradient, at any angle. ``sigma`` is graded as a polynomial from zero at the inner face to a
value chosen for a target round-trip reflection, and it is separable per axis — which means it
costs three 1D profiles and no extra field, applied only to the slabs where it differs from one.

**This is a sponge layer, not a true PML.** A split-field PML is exactly reflectionless at all
angles in the continuum; this is only approximately so at grazing incidence. It is here because
it costs nothing in memory and its reflection is *measured* (see ``scripts/validate_boundaries.py``)
rather than assumed — if the measurement is not good enough for a given job, that is the
argument for paying for a PML, and the number to beat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .grid import Grid
from .medium import Medium

__all__ = [
    "AbsorbingLayer",
    "WallAdmittances",
    "admittance_from_absorption",
    "edge_slabs",
    "layer_factors",
    "random_incidence_absorption",
    "reflection_coefficient",
    "wall_update_coefficients",
]

_AXES_ = (0, 1, 2)

#: Order of the faces everywhere in this module: (axis, side), side 0 = low, 1 = high.
FACES = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))


def admittance_from_absorption(absorption: float) -> float:
    """Normalised admittance of a wall with a given normal-incidence absorption coefficient.

    ``alpha = 1 - |R|^2`` with a real, non-negative reflection coefficient, so
    ``xi = (1 - R) / (1 + R)``. Real and positive is an assumption, not a fact: real materials
    have a phase, which is what makes a frequency-dependent (filter) boundary the eventual
    upgrade. For a single number out of an absorption table this is the right reading of it.
    """
    if not 0.0 <= absorption <= 1.0:
        raise ValueError(f"absorption must be in [0, 1], got {absorption}")
    reflection = math.sqrt(1.0 - absorption)
    return (1.0 - reflection) / (1.0 + reflection)


def reflection_coefficient(normalised_admittance: float, angle: float = 0.0) -> float:
    """Pressure reflection coefficient of that wall, at an angle from the normal in radians."""
    xi = normalised_admittance
    cosine = math.cos(angle)
    return (cosine - xi) / (cosine + xi)


def random_incidence_absorption(normalised_admittance: float, n_angles: int = 1024) -> float:
    """Absorption coefficient averaged over all angles of incidence, by the Paris formula.

    ``alpha_random = integral of alpha(theta) sin(2 theta) d theta`` over a quarter turn.

    **This is the number the Sabine and Eyring formulae want, and it is not the number in an
    absorption table.** A locally reacting surface absorbs considerably more at oblique
    incidence than head-on — for a wall quoted at 0.15 normal-incidence, the random-incidence
    figure is 0.25, and the predicted reverberation time differs by a factor of 1.7. Comparing a
    simulated decay against Sabine fed with the normal-incidence coefficient makes a correct
    simulation look twice as absorbent as it should be.

    The integral is done numerically rather than by one of the closed-form approximations in
    circulation, because those hold in the small-admittance limit and it is not obvious from a
    call site whether that limit applies.
    """
    angles = (np.arange(n_angles) + 0.5) * (math.pi / 2.0) / n_angles
    cosines = np.cos(angles)
    reflection = (cosines - normalised_admittance) / (cosines + normalised_admittance)
    absorption = 1.0 - reflection**2
    return float(np.trapezoid(absorption * np.sin(2.0 * angles), angles))


@dataclass(frozen=True)
class WallAdmittances:
    """Normalised admittance ``xi = rho c Y`` of each of the six walls.

    ``0`` is rigid, ``1`` is impedance-matched. Values are dimensionless so that a wall means
    the same thing in air and in water.
    """

    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0

    def __post_init__(self) -> None:
        for face, value in zip(FACES, self.values, strict=True):
            if value < 0.0:
                raise ValueError(
                    f"wall {face} has negative admittance {value}: that is a wall that "
                    "generates sound, and it will make the run diverge"
                )

    @property
    def values(self) -> tuple[float, ...]:
        """Admittances in :data:`FACES` order."""
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    @property
    def is_rigid(self) -> bool:
        return all(value == 0.0 for value in self.values)

    @classmethod
    def uniform(cls, normalised_admittance: float) -> WallAdmittances:
        return cls(*([normalised_admittance] * 6))

    @classmethod
    def from_absorption(cls, absorption: float) -> WallAdmittances:
        """All six walls with the same normal-incidence absorption coefficient."""
        return cls.uniform(admittance_from_absorption(absorption))

    @classmethod
    def matched(cls) -> WallAdmittances:
        """All six walls impedance-matched: the first-order absorbing boundary condition."""
        return cls.uniform(1.0)


def wall_update_coefficients(
    grid: Grid, medium: Medium, dt: float, walls: WallAdmittances
) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices of the cells touching an absorbing wall, and their coefficient ``a``.

    Returned sparse rather than as a grid-sized array: the boundary is a surface, and at the
    sizes this solver is meant for, a full field of mostly zeros is gigabytes of nothing.
    Cells on an edge or corner appear once, with the contributions of their two or three faces
    summed — which is all the special-casing that geometry needs.
    """
    shape = grid.shape
    strides = (shape[1] * shape[2], shape[2], 1)
    scale = medium.sound_speed * dt / (2.0 * grid.dx)

    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for (axis, side), admittance in zip(FACES, walls.values, strict=True):
        if admittance == 0.0:
            continue
        coordinates = [np.arange(count) for count in shape]
        coordinates[axis] = np.array([0 if side == 0 else shape[axis] - 1])

        flat = np.zeros(1, dtype=np.int64)
        for a, coordinate in enumerate(coordinates):
            broadcast = [1, 1, 1]
            broadcast[a] = -1
            flat = flat + coordinate.reshape(broadcast) * strides[a]
        flat = flat.ravel()

        indices.append(flat)
        values.append(np.full(flat.size, scale * admittance))

    if not indices:
        return np.empty(0, dtype=np.int64), np.empty(0)

    flat = np.concatenate(indices)
    coefficient = np.concatenate(values)
    unique, inverse = np.unique(flat, return_inverse=True)
    summed = np.zeros(unique.size)
    np.add.at(summed, inverse, coefficient)
    return unique, summed


@dataclass(frozen=True)
class AbsorbingLayer:
    """A graded matched layer lining every wall, to approximate free field.

    Args:
        thickness: Depth of the layer in cells. Fewer than about ten cells cannot be graded
            gently enough to be quiet, however large the damping is made.
        target_reflection: Round-trip amplitude reflection the grading is designed for, from
            the standard analytic estimate. It sets how hard the layer damps. It is not a
            promise: the measured performance is around -100 dB at normal incidence and
            -17 to -33 dB in a real 3D field, and neither number moves much with this
            parameter, because what limits the layer is the angle of incidence.
        order: Polynomial order of the grading. Three is the usual choice.
        faces: Which walls to line, as ``(axis, side)`` pairs. All six by default; a subset is
            what a half-space or a duct needs.
    """

    thickness: int = 16
    target_reflection: float = 1e-5
    order: float = 3.0
    faces: tuple[tuple[int, int], ...] = FACES

    def __post_init__(self) -> None:
        if self.thickness < 1:
            raise ValueError(f"thickness must be at least one cell, got {self.thickness}")
        if not 0.0 < self.target_reflection < 1.0:
            raise ValueError(f"target_reflection must be in (0, 1), got {self.target_reflection}")

    def max_damping(self, dx: float, sound_speed: float) -> float:
        """Peak ``sigma`` at the outer face, in 1/s."""
        depth = self.thickness * dx
        return -(self.order + 1.0) * sound_speed * math.log(self.target_reflection) / (2.0 * depth)


def layer_factors(
    grid: Grid, medium: Medium, dt: float, layer: AbsorbingLayer
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """Per-axis damping factors for the pressure grid and for each velocity component.

    Returns ``(pressure_factors, velocity_factors)`` where ``pressure_factors[axis]`` is a 1D
    array over that axis and ``velocity_factors[component][axis]`` likewise — the component's
    own axis being sampled on faces rather than centres, because a matched layer requires each
    field to see the ``sigma`` at *its own* position, not its neighbour's.
    """
    sigma_max = layer.max_damping(grid.dx, medium.sound_speed)
    depth = layer.thickness * grid.dx

    def factors(positions: np.ndarray, extent: float, axis: int) -> np.ndarray:
        distance = np.full(positions.shape, np.inf)
        if (axis, 0) in layer.faces:
            distance = np.minimum(distance, positions)
        if (axis, 1) in layer.faces:
            distance = np.minimum(distance, extent - positions)
        inside = np.clip(1.0 - distance / depth, 0.0, 1.0)
        return np.exp(-sigma_max * inside**layer.order * dt)

    pressure_factors = []
    for axis in range(3):
        extent = grid.lengths[axis]
        pressure_factors.append(factors(grid.cell_centres(axis), extent, axis))

    velocity_factors = []
    for component in range(3):
        per_axis = []
        for axis in range(3):
            extent = grid.lengths[axis]
            if axis == component:
                positions = np.arange(grid.shape[axis] + 1) * grid.dx
            else:
                positions = grid.cell_centres(axis)
            per_axis.append(factors(positions, extent, axis))
        velocity_factors.append(per_axis)

    return pressure_factors, velocity_factors


def edge_slabs(factor: np.ndarray, axis: int) -> list[tuple[tuple[slice, ...], np.ndarray]]:
    """Split a 1D damping profile into the slabs where it actually damps.

    The profile is one everywhere except within the absorbing layer at each end, so multiplying
    the whole field by it would be three full passes over memory to change a shell. Each
    returned pair is an index into the field and the factor to multiply that slab by, already
    shaped to broadcast.
    """
    damping = factor < 1.0
    if not damping.any():
        return []

    def broadcast(values: np.ndarray) -> np.ndarray:
        shape = [1, 1, 1]
        shape[axis] = -1
        return values.reshape(shape)

    def index(span: slice) -> tuple[slice, ...]:
        return tuple(span if a == axis else slice(None) for a in _AXES_)

    if damping.all():
        return [(index(slice(None)), broadcast(factor))]

    count = factor.size
    leading = int(np.argmin(damping)) if damping[0] else 0
    trailing = int(np.argmin(damping[::-1])) if damping[-1] else 0

    slabs = []
    if leading:
        slabs.append((index(slice(0, leading)), broadcast(factor[:leading])))
    if trailing:
        slabs.append((index(slice(count - trailing, None)), broadcast(factor[count - trailing :])))
    return slabs
