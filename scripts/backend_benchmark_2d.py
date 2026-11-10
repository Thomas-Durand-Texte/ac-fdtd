"""Measure 2D throughput, in billions of cell-updates per second, for the NumPy and C backends.

The 2D counterpart of ``backend_benchmark.py``. It runs a square grid of a few sizes through each
available backend and reports GCU/s (cells times steps over wall time). These are the numbers the
front end's backend selector interpolates (ac-fdtd-studio, ``engine/selection.py``), so re-run this
rather than editing a table by hand, and quote one run — there is ~10 % run-to-run variance.

There is no Torch 2D kernel yet, so only NumPy and C appear.

Usage:  uv run python scripts/backend_benchmark_2d.py
"""

from __future__ import annotations

import time

import numpy as np

from ac_fdtd import c_backend
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scheme2d import AcousticFDTD2D

#: Square grids to measure, by side length in cells.
SIDES = (300, 700, 1400, 2400)
#: Aim for roughly this many cell-updates per measurement, so small grids run more steps.
TARGET_UPDATES = 3e8


def _gcups(make, side: int, steps: int) -> float:
    grid = Grid2D(dx=0.01, nx=side, ny=side)
    solver = make(grid)
    solver.run(3)  # warm up: build the library / thread pool, touch the arrays
    start = time.perf_counter()
    solver.run(steps)
    elapsed = time.perf_counter() - start
    return (grid.n_cells * steps) / elapsed / 1e9


def _c_backend(dtype: np.dtype):
    def make(grid: Grid2D):
        return c_backend.CAcousticFDTD2D(grid, dtype=dtype)

    return make


def main() -> None:
    have_c = c_backend.is_available()
    print(f"{'side':>6} {'cells':>12} {'numpy':>9} {'c_fp64':>9} {'c_fp32':>9}")
    for side in SIDES:
        cells = side * side
        steps = max(20, int(TARGET_UPDATES / cells))
        numpy_gcups = _gcups(AcousticFDTD2D, side, steps)
        if have_c:
            c64 = f"{_gcups(_c_backend(np.float64), side, steps):>9.3f}"
            c32 = f"{_gcups(_c_backend(np.float32), side, steps):>9.3f}"
        else:
            c64 = c32 = f"{'--':>9}"
        print(f"{side:>6} {cells:>12,} {numpy_gcups:>9.3f} {c64} {c32}")


if __name__ == "__main__":
    main()
