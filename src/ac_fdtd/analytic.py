"""Closed-form solutions for the rigid rectangular box, in both continuous and discrete form.

A rigid box is the one 3D case where everything is known exactly, which makes it the backbone
of the validation. What matters here is that there are *two* exact answers and they are not the
same number:

* :func:`mode_frequency` — the physical eigenfrequency of the continuous room.
* :func:`discrete_mode_frequency` — the eigenfrequency the discrete scheme actually has, from
  its dispersion relation.

The gap between them is the discretisation error, and it is the thing worth measuring: it
should shrink as ``dx^2``. Conflating the two turns a scheme that is behaving perfectly into an
apparent 1 % error, or worse, hides a real bug behind a tolerance loose enough to swallow the
discretisation error as well.

The cosine mode shapes are exact eigenvectors of the *discrete* rigid-wall Laplacian, not just
of the continuous one. That is what :func:`mode_initial_state` exploits: started from one mode
with a consistently staggered velocity field, the simulation stays a pure sinusoid at
:func:`discrete_mode_frequency` forever, to round-off. Comparing against that is a far stronger
test than picking peaks out of a spectrum, and it costs a few thousand steps instead of a
second of simulated time.
"""

from __future__ import annotations

import math

import numpy as np

from .grid import Grid
from .medium import Medium

__all__ = [
    "mode_frequency",
    "wavenumbers",
    "discrete_mode_frequency",
    "mode_pressure_field",
    "mode_initial_state",
]


def mode_frequency(
    lengths: tuple[float, float, float],
    mode: tuple[int, int, int],
    sound_speed: float,
) -> float:
    """Eigenfrequency of mode ``(l, m, n)`` of a rigid box, in Hz.

    ``f = (c/2) * sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)``. Use the *simulated* room dimensions
    (:attr:`Grid.lengths`), not the requested ones.
    """
    return (
        0.5
        * sound_speed
        * math.sqrt(sum((order / length) ** 2 for order, length in zip(mode, lengths, strict=True)))
    )


def wavenumbers(
    lengths: tuple[float, float, float], mode: tuple[int, int, int]
) -> tuple[float, float, float]:
    """Modal wavenumbers ``k = order*pi/length``, in rad/m."""
    return tuple(order * math.pi / length for order, length in zip(mode, lengths, strict=True))


def discrete_mode_frequency(
    grid: Grid, mode: tuple[int, int, int], medium: Medium, dt: float
) -> float:
    """Eigenfrequency the discrete scheme gives that mode, in Hz.

    From the dispersion relation of the staggered leapfrog,

        sin(omega*dt/2) = (c*dt/dx) * sqrt( sum_axes sin^2(k_axis*dx/2) )

    The right-hand side reaches 1 exactly when the scheme is at its stability limit and the
    mode is the shortest the grid can hold; beyond that the arcsine has no real solution, which
    *is* the instability, seen from the other side.
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


def mode_pressure_field(grid: Grid, mode: tuple[int, int, int]) -> np.ndarray:
    """Unit-amplitude pressure field of one rigid-box mode, sampled at cell centres."""
    field = np.ones(grid.shape)
    for axis, (order, length) in enumerate(zip(mode, grid.lengths, strict=True)):
        profile = np.cos(order * math.pi * grid.cell_centres(axis) / length)
        field = field * profile.reshape([-1 if a == axis else 1 for a in range(3)])
    return field


def mode_initial_state(
    grid: Grid, mode: tuple[int, int, int], medium: Medium, dt: float, amplitude: float = 1.0
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Initial ``(p^0, v^{-1/2})`` that launches exactly one mode and nothing else.

    Setting the velocity to zero would be the obvious thing to do and would be wrong: zero is
    not the value the staggered scheme wants half a step *before* a pressure maximum, and the
    difference excites a second, counter-propagating solution. Requiring
    ``p^n = A*cos(omega*n*dt)*Phi`` for all ``n`` pins the velocity down to

        v^{-1/2} = (dt / (2*rho)) * grad(p^0)

    which is what this returns. The factor of a half is the half step, made concrete.
    """
    pressure = amplitude * mode_pressure_field(grid, mode)
    coefficient = dt / (2.0 * medium.density * grid.dx)

    velocity = []
    for axis in range(3):
        component = np.zeros(grid.velocity_shape(axis))
        interior = tuple(slice(1, -1) if a == axis else slice(None) for a in range(3))
        upper = tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
        lower = tuple(slice(None, -1) if a == axis else slice(None) for a in range(3))
        component[interior] = coefficient * (pressure[upper] - pressure[lower])
        velocity.append(component)

    return pressure, velocity
