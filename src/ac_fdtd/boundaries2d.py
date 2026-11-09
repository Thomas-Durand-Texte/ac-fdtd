"""2D edges: absorbing walls and a matched layer — the plane peer of :mod:`ac_fdtd.boundaries`.

The physics is identical to 3D and is derived there; this module only re-expresses the two
grid-dependent constructions — the per-cell wall coefficients and the graded layer factors — over
two axes and four walls instead of three and six. The scalar relations that do not depend on the
grid (``admittance_from_absorption``, ``reflection_coefficient``, ``random_incidence_absorption``)
are the same numbers in any dimension and are imported from :mod:`ac_fdtd.boundaries` rather than
copied.

Both mechanisms keep the defining property of the 3D versions: the wall velocity lines stay at
zero and the flux through the wall enters only the pressure update of the boundary cells, so edges
where two walls meet need no special case and the closed-form solve is unconditionally dissipative
for any non-negative admittance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .boundaries import (  # noqa: F401  (re-exported: dimension-independent, one definition)
    admittance_from_absorption,
    random_incidence_absorption,
    reflection_coefficient,
)
from .grid2d import Grid2D
from .medium import Medium

__all__ = [
    "AbsorbingLayer2D",
    "WallAdmittances2D",
    "admittance_from_absorption",
    "edge_slabs_2d",
    "layer_factors_2d",
    "random_incidence_absorption",
    "reflection_coefficient",
    "wall_update_coefficients_2d",
]

_AXES_ = (0, 1)

#: Order of the walls everywhere here: (axis, side), side 0 = low, 1 = high.
FACES = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class WallAdmittances2D:
    """Normalised admittance ``xi = rho c Y`` of each of the four walls.

    ``0`` is rigid, ``1`` is impedance-matched. Dimensionless, so a wall means the same thing in
    air and in water.
    """

    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0

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
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def is_rigid(self) -> bool:
        return all(value == 0.0 for value in self.values)

    @classmethod
    def uniform(cls, normalised_admittance: float) -> WallAdmittances2D:
        return cls(*([normalised_admittance] * 4))

    @classmethod
    def from_absorption(cls, absorption: float) -> WallAdmittances2D:
        """All four walls with the same normal-incidence absorption coefficient."""
        return cls.uniform(admittance_from_absorption(absorption))

    @classmethod
    def matched(cls) -> WallAdmittances2D:
        """All four walls impedance-matched: the first-order absorbing boundary condition."""
        return cls.uniform(1.0)


def wall_update_coefficients_2d(
    grid: Grid2D, medium: Medium, dt: float, walls: WallAdmittances2D
) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices of the cells touching an absorbing wall, and their coefficient ``a``.

    Returned sparse — the boundary is a line, not an area — with corner cells appearing once and
    carrying the summed contribution of the two walls that meet there, which is all the
    special-casing a 2D corner needs.
    """
    shape = grid.shape
    strides = (shape[1], 1)
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
            broadcast = [1, 1]
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
class AbsorbingLayer2D:
    """A graded matched layer lining the walls, to approximate free field in the plane.

    Args:
        thickness: Depth in cells. Below about ten cells it cannot be graded gently enough to be
            quiet, however large the damping.
        target_reflection: Round-trip amplitude reflection the grading is designed for; sets how
            hard the layer damps. A design figure, not a promise — what limits the layer is the
            angle of incidence, as in 3D.
        order: Polynomial order of the grading. Three is the usual choice.
        faces: Which walls to line, as ``(axis, side)`` pairs. All four by default; a subset is
            what a half-plane or a duct needs.
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


def layer_factors_2d(
    grid: Grid2D, medium: Medium, dt: float, layer: AbsorbingLayer2D
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """Per-axis damping factors for the pressure grid and for each velocity component.

    Returns ``(pressure_factors, velocity_factors)`` with ``pressure_factors[axis]`` a 1D array
    over that axis and ``velocity_factors[component][axis]`` likewise — the component's own axis
    sampled on faces rather than centres, because a matched layer needs each field to see the
    ``sigma`` at its own position.
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
    for axis in range(2):
        extent = grid.lengths[axis]
        pressure_factors.append(factors(grid.cell_centres(axis), extent, axis))

    velocity_factors = []
    for component in range(2):
        per_axis = []
        for axis in range(2):
            extent = grid.lengths[axis]
            if axis == component:
                positions = np.arange(grid.shape[axis] + 1) * grid.dx
            else:
                positions = grid.cell_centres(axis)
            per_axis.append(factors(positions, extent, axis))
        velocity_factors.append(per_axis)

    return pressure_factors, velocity_factors


def edge_slabs_2d(factor: np.ndarray, axis: int) -> list[tuple[tuple[slice, ...], np.ndarray]]:
    """Split a 1D damping profile into the slabs where it actually damps.

    The profile is one except inside the layer at each end, so multiplying the whole field by it
    would be two full passes over memory to change a shell. Each returned pair is an index into the
    field and the factor to multiply that slab by, already shaped to broadcast.
    """
    damping = factor < 1.0
    if not damping.any():
        return []

    def broadcast(values: np.ndarray) -> np.ndarray:
        shape = [1, 1]
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
