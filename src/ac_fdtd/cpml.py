"""A CFS-CPML open boundary for the 3D pressure-only scheme: the box peer of :mod:`ac_fdtd.cpml2d`.

The mechanism, the grading and the reasoning are identical and written up there and in
``docs/kernel_study``; this module only re-expresses the profiles over three axes and six faces.
:class:`ac_fdtd.scalar.ScalarFDTD` runs the first-order CPML update in the boundary shell and the
scalar update everywhere else, so velocity and the memory fields live only in the shell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .boundaries import FACES
from .grid import Grid
from .medium import Medium

__all__ = ["AbsorbingBoundary", "cpml_axis_profiles_3d"]


@dataclass(frozen=True)
class AbsorbingBoundary:
    """A CFS-CPML absorbing region lining a subset of the six walls of a box.

    The fields mean exactly what :class:`ac_fdtd.cpml2d.AbsorbingBoundary2D` documents; ``faces`` is
    over the six ``(axis, side)`` pairs. A closed room passes ``None`` and stores no velocity.
    """

    thickness: int = 24
    target_reflection: float = 1e-6
    kappa_max: float = 5.0
    alpha_max: float = 0.0
    order: float = 3.0
    faces: tuple[tuple[int, int], ...] = FACES

    def __post_init__(self) -> None:
        if self.thickness < 1:
            raise ValueError(f"thickness must be at least one cell, got {self.thickness}")
        if not 0.0 < self.target_reflection < 1.0:
            raise ValueError(f"target_reflection must be in (0, 1), got {self.target_reflection}")
        if self.kappa_max < 1.0:
            raise ValueError(f"kappa_max must be >= 1, got {self.kappa_max}")
        if self.alpha_max < 0.0:
            raise ValueError(f"alpha_max must be non-negative, got {self.alpha_max}")

    @classmethod
    def open_field(
        cls,
        thickness: int = 24,
        characteristic_frequency: float = 1000.0,
        faces: tuple[tuple[int, int], ...] = FACES,
    ) -> AbsorbingBoundary:
        """A deep layer tuned for a near-anechoic open boundary at all angles."""
        return cls(
            thickness=thickness,
            target_reflection=1e-6,
            kappa_max=5.0,
            alpha_max=2.0 * math.pi * characteristic_frequency,
            faces=faces,
        )

    @classmethod
    def partial_skin(
        cls,
        absorption: float,
        thickness: int = 8,
        characteristic_frequency: float = 1000.0,
        faces: tuple[tuple[int, int], ...] = FACES,
    ) -> AbsorbingBoundary:
        """A shallow skin tuned to absorb a target fraction, angle-robustly (see the 2D peer)."""
        if not 0.0 < absorption < 1.0:
            raise ValueError(f"absorption must be in (0, 1), got {absorption}")
        return cls(
            thickness=thickness,
            target_reflection=1.0 - absorption,
            kappa_max=1.0,
            alpha_max=2.0 * math.pi * characteristic_frequency,
            order=2.0,
            faces=faces,
        )

    def has_axis(self, axis: int) -> tuple[bool, bool]:
        """Whether this boundary lines the low and high side of ``axis``."""
        return ((axis, 0) in self.faces, (axis, 1) in self.faces)


def cpml_axis_profiles_3d(
    grid: Grid, medium: Medium, dt: float, boundary: AbsorbingBoundary
) -> dict:
    """Per-axis ``(kappa, b, a)`` triples; keys ``("p", axis)`` at cell centres, ``("v", axis)`` at
    faces (one longer). Identical construction to the 2D module, over three axes."""
    thickness = boundary.thickness
    layer_depth = thickness * grid.dx
    sigma_max = (
        -(boundary.order + 1.0)
        * medium.sound_speed
        * math.log(boundary.target_reflection)
        / (2.0 * layer_depth)
    )

    def triple(n: int, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        low, high = boundary.has_axis(axis)
        position = np.arange(n)
        depth = np.zeros(n)
        if low:
            depth = np.maximum(depth, (thickness - position) / thickness)
        if high:
            depth = np.maximum(depth, (position - (n - 1 - thickness)) / thickness)
        depth = np.clip(depth, 0.0, 1.0)
        sigma = sigma_max * depth**boundary.order
        kappa = 1.0 + (boundary.kappa_max - 1.0) * depth**boundary.order
        alpha = boundary.alpha_max * (1.0 - depth)
        b = np.exp(-(sigma / kappa + alpha) * dt)
        denom = kappa * (sigma + kappa * alpha)
        a = np.zeros_like(denom)
        active = denom > 1e-30
        a[active] = sigma[active] / denom[active] * (b[active] - 1.0)
        return kappa, b, a

    profiles: dict = {}
    for axis in range(3):
        profiles[("p", axis)] = triple(grid.shape[axis], axis)
        profiles[("v", axis)] = triple(grid.shape[axis] + 1, axis)
    return profiles
