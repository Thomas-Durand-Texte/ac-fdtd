"""The Cartesian grid: spacing, cell counts, and the time step the spacing forces on us.

Two things here are easy to get wrong and are therefore made explicit.

**The room you asked for is not quite the room you get.** One uniform spacing cannot divide
three arbitrary lengths exactly, so each axis is rounded to a whole number of cells and the
simulated room is ``n * dx`` on that axis. The difference is at most half a cell, but it moves
the modal frequencies by a comparable relative amount — which is the same order as the error we
are trying to measure in the validation. So :attr:`Grid.lengths` reports the room that is
actually simulated, and :attr:`Grid.length_error` reports how far that is from the request.
Every analytical comparison must use the former.

**The time step follows from the spacing, not from the frequency of interest.** The 2D
prototype this repo grew out of used ``dt = 1e-2 / (1.2 * f_max)``, which is only stable
because ``dx = 1e-2`` happened to be baked into the numerator. Here it comes from the
Courant condition, so changing either ``dx`` or ``c`` cannot silently produce an unstable run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Grid", "max_stable_time_step"]

#: Spatial dimensionality, spelled out because it appears in the stability limit as sqrt(3)
#: and reads as a magic number otherwise.
_N_DIMENSIONS = 3


def max_stable_time_step(dx: float, sound_speed: float) -> float:
    """Largest time step the explicit staggered scheme is stable at, in seconds.

    The Courant limit in 3D is ``c*dt*sqrt(3)/dx <= 1``. Note that running *at* the limit is
    not a compromise but the preferred operating point: it is where the scheme's numerical
    dispersion is smallest. Running below it buys no accuracy and costs proportionally more
    steps.
    """
    return dx / (sound_speed * math.sqrt(_N_DIMENSIONS))


@dataclass(frozen=True)
class Grid:
    """A uniform Cartesian grid of cubic cells.

    Pressure lives at cell centres, so cell ``(i, j, k)`` is centred at
    ``((i+0.5)*dx, (j+0.5)*dx, (k+0.5)*dx)``. Normal velocity lives on the faces between them,
    which is what puts the rigid-wall condition exactly on the boundary planes ``x = 0`` and
    ``x = nx*dx`` rather than half a cell inside.
    """

    dx: float
    nx: int
    ny: int
    nz: int
    requested_lengths: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.dx <= 0.0:
            raise ValueError(f"dx must be positive, got {self.dx}")
        if min(self.nx, self.ny, self.nz) < 1:
            raise ValueError(f"cell counts must be >= 1, got {self.shape}")

    @classmethod
    def from_lengths(cls, lengths: tuple[float, float, float], dx: float) -> Grid:
        """Grid covering a room of the given dimensions, rounding each axis to whole cells."""
        counts = tuple(max(1, round(length / dx)) for length in lengths)
        return cls(dx=dx, nx=counts[0], ny=counts[1], nz=counts[2], requested_lengths=lengths)

    @classmethod
    def from_max_frequency(
        cls,
        lengths: tuple[float, float, float],
        max_frequency: float,
        sound_speed: float,
        points_per_wavelength: float = 10.0,
    ) -> Grid:
        """Grid resolving up to ``max_frequency`` at a given number of points per wavelength.

        Ten points per wavelength is the conventional default for a second-order scheme; it is
        a starting point, not a guarantee. What it costs in phase error depends on the
        direction of propagation, and the honest number for a given target comes from measuring
        the dispersion rather than from this default.
        """
        if max_frequency <= 0.0:
            raise ValueError(f"max_frequency must be positive, got {max_frequency}")
        if points_per_wavelength <= 0.0:
            raise ValueError(f"points_per_wavelength must be positive, got {points_per_wavelength}")
        dx = sound_speed / (points_per_wavelength * max_frequency)
        return cls.from_lengths(lengths, dx)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the pressure field."""
        return (self.nx, self.ny, self.nz)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def lengths(self) -> tuple[float, float, float]:
        """Dimensions of the room actually simulated, in metres."""
        return (self.nx * self.dx, self.ny * self.dx, self.nz * self.dx)

    @property
    def length_error(self) -> tuple[float, float, float]:
        """Signed difference ``simulated - requested`` per axis, in metres.

        Zero on every axis when no request was recorded.
        """
        if self.requested_lengths is None:
            return (0.0, 0.0, 0.0)
        return tuple(
            simulated - requested
            for simulated, requested in zip(self.lengths, self.requested_lengths, strict=True)
        )

    def velocity_shape(self, axis: int) -> tuple[int, int, int]:
        """Shape of the velocity component normal to ``axis``: one extra face on that axis."""
        shape = list(self.shape)
        shape[axis] += 1
        return tuple(shape)

    def cell_index(self, point: tuple[float, float, float]) -> tuple[int, int, int]:
        """Index of the cell containing ``point``, clipped to the grid.

        Clipping rather than raising is deliberate: a source nominally *on* a wall is a normal
        request, and ``x = Lx`` would otherwise fall one cell outside.
        """
        return tuple(
            int(min(max(math.floor(coordinate / self.dx), 0), count - 1))
            for coordinate, count in zip(point, self.shape, strict=True)
        )

    def cell_centres(self, axis: int) -> np.ndarray:
        """Coordinates of the cell centres along one axis, in metres."""
        return (np.arange(self.shape[axis]) + 0.5) * self.dx

    def memory_bytes(self, itemsize: int = 8, extra_fields: int = 0) -> int:
        """Footprint of the state — pressure plus three velocity components — in bytes.

        ``extra_fields`` counts additional cell-sized arrays, such as the relaxation states an
        air absorption model carries. The velocity components each hold one extra plane; that
        is a sub-percent correction at useful sizes, but it is included because the only time
        anyone reads this number is when deciding whether a run fits in memory.
        """
        velocity_cells = sum(math.prod(self.velocity_shape(axis)) for axis in range(3))
        cells = self.n_cells * (1 + extra_fields) + velocity_cells
        return cells * itemsize
