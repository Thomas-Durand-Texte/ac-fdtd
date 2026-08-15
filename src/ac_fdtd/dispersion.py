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

Fourth order in space
---------------------
The same relation covers a wider spatial stencil by substituting its symbol, which is what
``order=4`` does. The result is worth stating carefully, because the name promises more than it
delivers: **a fourth-order spatial operator with a leapfrog time step is still second order
overall.** The error still falls as the square of the resolution — it is the constant that
improves, by about a factor of 2.8 — because what is left is the time discretisation, and the
exact cancellation along the body diagonal is gone too, since the wider stencil forces a
smaller Courant number (0.495 rather than 0.577).

What it does buy is real anyway: 6.3 points per wavelength instead of 10.5 for one percent,
which is 4.7 times fewer cells, and about 6.8 times fewer cell-updates once the shorter time
step is paid for.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "max_courant",
    "phase_velocity_ratio",
    "worst_direction_error",
    "required_points_per_wavelength",
]

#: Spatial orders this module knows the symbol of.
SUPPORTED_ORDERS = (2, 4)

_AXIS = np.array([1.0, 0.0, 0.0])
_FACE_DIAGONAL = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
_BODY_DIAGONAL = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)

#: The three directions worth naming: worst case, an intermediate, and the exact one.
NAMED_DIRECTIONS = {
    "axis (1,0,0)": _AXIS,
    "face diagonal (1,1,0)": _FACE_DIAGONAL,
    "body diagonal (1,1,1)": _BODY_DIAGONAL,
}


def _symbol(theta: np.ndarray, order: int) -> np.ndarray:
    """``dx`` times the staggered difference operator's Fourier symbol, as a magnitude.

    Second order is ``(p_i - p_{i-1}) / dx``; fourth order is the standard staggered
    ``[27 (p_i - p_{i-1}) - (p_{i+1} - p_{i-2})] / (24 dx)``.
    """
    if order == 2:
        return 2.0 * np.sin(theta / 2.0)
    if order == 4:
        return (27.0 * np.sin(theta / 2.0) - np.sin(1.5 * theta)) / 12.0
    raise ValueError(f"order must be one of {SUPPORTED_ORDERS}, got {order}")


def max_courant(order: int = 2) -> float:
    """Largest stable ``c dt / dx`` for this spatial order, in three dimensions.

    A wider stencil has a larger symbol at the grid's shortest wavelength, so it buys accuracy
    partly with a shorter time step: 0.577 at second order, 0.495 at fourth.
    """
    peak = float(np.max(_symbol(np.linspace(0.0, np.pi, 20001), order)))
    return 2.0 / (math.sqrt(3.0) * peak)


def phase_velocity_ratio(
    points_per_wavelength: np.ndarray | float,
    direction: np.ndarray,
    courant: float = 1.0,
    order: int = 2,
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
    ratio = courant * max_courant(order)

    resolution = np.asarray(points_per_wavelength, dtype=float)
    angle = 2.0 * np.pi / resolution[..., None] * direction
    spatial = 0.5 * np.sqrt(np.sum(_symbol(angle, order) ** 2, axis=-1))

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
    points_per_wavelength: float, courant: float = 1.0, n_angles: int = 90, order: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase velocity error over an octant of directions, as ``(azimuth, elevation, error)``."""
    azimuth, elevation, directions = _direction_grid(n_angles)
    ratio = courant * max_courant(order)
    angle = 2.0 * np.pi / points_per_wavelength * directions
    spatial = 0.5 * np.sqrt(np.sum(_symbol(angle, order) ** 2, axis=-1))
    error = points_per_wavelength / (np.pi * ratio) * np.arcsin(ratio * spatial) - 1.0
    return azimuth, elevation, error


def worst_direction_error(
    points_per_wavelength: float, courant: float = 1.0, n_angles: int = 90, order: int = 2
) -> float:
    """Largest phase velocity error over all directions, as a fraction of ``c``."""
    _, _, error = direction_map(points_per_wavelength, courant, n_angles, order)
    return float(np.max(np.abs(error)))


def relative_cost(target_error: float, order: int = 4) -> float:
    """Cell-updates this order needs for a target error, relative to second order.

    Below one means cheaper. Counts both effects: the coarser grid a wider stencil allows
    (cells go as the cube of the resolution) and the shorter time step it forces.
    """
    second = required_points_per_wavelength(target_error, order=2)
    other = required_points_per_wavelength(target_error, order=order)
    return (other / second) ** 4 * (max_courant(2) / max_courant(order))


def required_points_per_wavelength(
    target_error: float, courant: float = 1.0, tolerance: float = 1e-4, order: int = 2
) -> float:
    """Resolution needed to keep the phase velocity error below ``target_error`` everywhere.

    Bisection rather than the ``1/n^2`` asymptote, which is only accurate where the answer is
    already large — and the interesting requests are the coarse ones.
    """
    if not 0.0 < target_error < 1.0:
        raise ValueError(f"target_error must be in (0, 1), got {target_error}")

    low, high = 2.5, 4.0
    while worst_direction_error(high, courant, order=order) > target_error:
        high *= 2.0
        if high > 1e5:
            raise ValueError(f"no resolution reaches an error of {target_error}")

    while high - low > tolerance * low:
        middle = 0.5 * (low + high)
        if worst_direction_error(middle, courant, order=order) > target_error:
            low = middle
        else:
            high = middle
    return high
