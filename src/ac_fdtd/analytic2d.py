"""Closed-form rigid-rectangle solutions, the 2D peer of :mod:`ac_fdtd.analytic`.

The same two distinct exact answers matter here as in 3D — the physical eigenfrequency of the
continuous rectangle and the eigenfrequency the discrete scheme actually has — and their gap is the
discretisation error, which should shrink as ``dx^2``. The cosine mode shapes are exact
eigenvectors of the discrete rigid-wall Laplacian, so :func:`mode_initial_state` launches a single
mode that stays a pure sinusoid at :func:`discrete_mode_frequency` forever; comparing against that
is far stronger than peak-picking a spectrum. See the 3D module for the full reasoning.
"""

from __future__ import annotations

import math

import numpy as np

from .grid2d import Grid2D
from .medium import Medium

__all__ = [
    "discrete_mode_frequency",
    "mode_frequency",
    "mode_initial_state",
    "mode_pressure_field",
    "wavenumbers",
]


def mode_frequency(
    lengths: tuple[float, float], mode: tuple[int, int], sound_speed: float
) -> float:
    """Eigenfrequency of mode ``(l, m)`` of a rigid rectangle, in Hz.

    ``f = (c/2) * sqrt((l/Lx)^2 + (m/Ly)^2)``, from the *simulated* dimensions (``Grid2D.lengths``).
    """
    return (
        0.5
        * sound_speed
        * math.sqrt(sum((order / length) ** 2 for order, length in zip(mode, lengths, strict=True)))
    )


def wavenumbers(lengths: tuple[float, float], mode: tuple[int, int]) -> tuple[float, float]:
    """Modal wavenumbers ``k = order*pi/length``, in rad/m."""
    return tuple(order * math.pi / length for order, length in zip(mode, lengths, strict=True))


def discrete_mode_frequency(
    grid: Grid2D, mode: tuple[int, int], medium: Medium, dt: float
) -> float:
    """Eigenfrequency the discrete scheme gives that mode, in Hz.

    From the staggered-leapfrog dispersion relation
    ``sin(omega*dt/2) = (c*dt/dx) * sqrt( sum_axes sin^2(k_axis*dx/2) )``. The right-hand side
    reaches 1 at the stability limit for the shortest representable mode; beyond that the arcsine
    has no real solution, which is the instability seen from the other side.
    """
    dx = grid.dx
    spatial = math.sqrt(sum(math.sin(k * dx / 2.0) ** 2 for k in wavenumbers(grid.lengths, mode)))
    argument = medium.sound_speed * dt / dx * spatial
    if argument > 1.0:
        raise ValueError(
            f"mode {mode} is not representable: dispersion argument {argument:.6f} > 1, "
            "which means this time step is unstable for this grid"
        )
    return math.asin(argument) / (math.pi * dt)


def mode_pressure_field(grid: Grid2D, mode: tuple[int, int]) -> np.ndarray:
    """Unit-amplitude pressure field of one rigid-rectangle mode, sampled at cell centres."""
    field = np.ones(grid.shape)
    for axis, (order, length) in enumerate(zip(mode, grid.lengths, strict=True)):
        profile = np.cos(order * math.pi * grid.cell_centres(axis) / length)
        field = field * profile.reshape([-1 if a == axis else 1 for a in range(2)])
    return field


def mode_initial_state(
    grid: Grid2D, mode: tuple[int, int], medium: Medium, dt: float, amplitude: float = 1.0
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Initial ``(p^0, v^{-1/2})`` that launches exactly one mode and nothing else.

    Zero velocity would be wrong — it is not the value the staggered scheme wants half a step
    before a pressure maximum, and the difference excites a counter-propagating solution. Requiring
    ``p^n = A*cos(omega*n*dt)*Phi`` pins the velocity to ``v^{-1/2} = (dt/(2*rho)) * grad(p^0)``.
    """
    pressure = amplitude * mode_pressure_field(grid, mode)
    coefficient = dt / (2.0 * medium.density * grid.dx)

    velocity = []
    for axis in range(2):
        component = np.zeros(grid.velocity_shape(axis))
        interior = tuple(slice(1, -1) if a == axis else slice(None) for a in range(2))
        upper = tuple(slice(1, None) if a == axis else slice(None) for a in range(2))
        lower = tuple(slice(None, -1) if a == axis else slice(None) for a in range(2))
        component[interior] = coefficient * (pressure[upper] - pressure[lower])
        velocity.append(component)

    return pressure, velocity
