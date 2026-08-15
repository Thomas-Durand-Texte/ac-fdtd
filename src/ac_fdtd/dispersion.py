"""What the grid does to the speed of sound, as a function of direction and resolution.

The scheme's dispersion relation is exact and closed-form, so none of this needs a simulation:

    sin(omega dt / 2) = (c dt / dx) sqrt( sum_axes sin^2(k_axis dx / 2) )

Dividing the resulting ``omega`` by ``k`` gives the phase velocity the grid actually propagates
at, and the ratio to ``c`` is the error. Expressed in points per wavelength it depends on
nothing else, which is what makes it a design rule rather than a property of one run.

The one result worth knowing before reading any of it: **at the stability limit the body
diagonal is exact.** Not approximately — the arcsine and its argument cancel identically when
the Courant number is one, and a wave travelling along ``(1,1,1)`` propagates at exactly ``c``
for every resolution. The axes are the worst case, and everything else falls between. That is
also the reason the default Courant number is 1: running below the limit is not a safety
margin, it breaks the cancellation and makes the error worse in every direction at once.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "phase_velocity_ratio",
    "worst_direction_error",
    "required_points_per_wavelength",
]

_AXIS = np.array([1.0, 0.0, 0.0])
_FACE_DIAGONAL = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
_BODY_DIAGONAL = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)

#: The three directions worth naming: worst case, an intermediate, and the exact one.
NAMED_DIRECTIONS = {
    "axis (1,0,0)": _AXIS,
    "face diagonal (1,1,0)": _FACE_DIAGONAL,
    "body diagonal (1,1,1)": _BODY_DIAGONAL,
}


def phase_velocity_ratio(
    points_per_wavelength: np.ndarray | float,
    direction: np.ndarray,
    courant: float = 1.0,
) -> np.ndarray:
    """Numerical phase velocity divided by ``c``, for one direction of propagation.

    Args:
        points_per_wavelength: Wavelength in cells. Below two the wave is not representable.
        direction: Propagation direction; normalised internally.
        courant: Fraction of the stability limit, so ``c dt / dx = courant / sqrt(3)``.

    Returns:
        ``c_numerical / c``. Below one everywhere except along the body diagonal at
        ``courant = 1``, where it is exactly one.
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    ratio = courant / math.sqrt(3.0)

    resolution = np.asarray(points_per_wavelength, dtype=float)
    half_angle = np.pi / resolution[..., None] * direction
    spatial = np.sqrt(np.sum(np.sin(half_angle) ** 2, axis=-1))

    argument = ratio * spatial
    if np.any(argument > 1.0):
        raise ValueError(
            "the dispersion relation has no real solution here: this combination of "
            "resolution and Courant number is unstable"
        )
    return resolution / (np.pi * ratio) * np.arcsin(argument)


def _direction_grid(n_angles: int = 90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directions over one octant, which is all the symmetry leaves."""
    azimuth = np.linspace(0.0, np.pi / 2.0, n_angles)
    elevation = np.linspace(0.0, np.pi / 2.0, n_angles)
    grid_azimuth, grid_elevation = np.meshgrid(azimuth, elevation, indexing="ij")
    directions = np.stack(
        (
            np.cos(grid_elevation) * np.cos(grid_azimuth),
            np.cos(grid_elevation) * np.sin(grid_azimuth),
            np.sin(grid_elevation),
        ),
        axis=-1,
    )
    return grid_azimuth, grid_elevation, directions


def direction_map(
    points_per_wavelength: float, courant: float = 1.0, n_angles: int = 90
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase velocity error over an octant of directions, as ``(azimuth, elevation, error)``."""
    azimuth, elevation, directions = _direction_grid(n_angles)
    ratio = courant / math.sqrt(3.0)
    half_angle = np.pi / points_per_wavelength * directions
    spatial = np.sqrt(np.sum(np.sin(half_angle) ** 2, axis=-1))
    error = points_per_wavelength / (np.pi * ratio) * np.arcsin(ratio * spatial) - 1.0
    return azimuth, elevation, error


def worst_direction_error(
    points_per_wavelength: float, courant: float = 1.0, n_angles: int = 90
) -> float:
    """Largest phase velocity error over all directions, as a fraction of ``c``."""
    _, _, error = direction_map(points_per_wavelength, courant, n_angles)
    return float(np.max(np.abs(error)))


def required_points_per_wavelength(
    target_error: float, courant: float = 1.0, tolerance: float = 1e-4
) -> float:
    """Resolution needed to keep the phase velocity error below ``target_error`` everywhere.

    Bisection rather than the ``1/n^2`` asymptote, which is only accurate where the answer is
    already large — and the interesting requests are the coarse ones.
    """
    if not 0.0 < target_error < 1.0:
        raise ValueError(f"target_error must be in (0, 1), got {target_error}")

    low, high = 2.5, 4.0
    while worst_direction_error(high, courant) > target_error:
        high *= 2.0
        if high > 1e5:
            raise ValueError(f"no resolution reaches an error of {target_error}")

    while high - low > tolerance * low:
        middle = 0.5 * (low + high)
        if worst_direction_error(middle, courant) > target_error:
            low = middle
        else:
            high = middle
    return high
