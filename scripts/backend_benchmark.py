"""M7 — which backend, at which size, and what does a real room actually cost?

Three questions:

1. **Throughput against grid size**, for every backend. Reported as cell-updates per second,
   because wall clock hides the size dependence and the size dependence is the whole point:
   the array backends are launch-bound when the grid is small and bandwidth-bound when it is
   large, and they cross over.
2. **How close each one gets to the hardware.** This scheme moves about one byte per
   arithmetic operation, so the ceiling is memory bandwidth, measured here directly with a
   fused add rather than taken from a specification sheet. Each backend's traffic is counted
   from its own implementation: forty-one array passes per cell per step for NumPy and PyTorch,
   which express a step as whole-array operations, against forty-eight bytes for the fused C
   loop.
3. **What that means for a room.** The decision table at the end: for a given bandwidth, how
   many cells, how much memory, and how long one second of impulse response takes on each
   backend.

Run: ``uv run python scripts/backend_benchmark.py``  (a few minutes)
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotstyle  # noqa: E402

from ac_fdtd import AIR, AcousticFDTD, Grid, WallAdmittances  # noqa: E402
from ac_fdtd.dispersion import required_points_per_wavelength  # noqa: E402

FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m7_backends.svg"

SIZES = (64, 96, 128, 192, 256, 320, 384)
#: Steps are chosen per size to time about this much work, so a small grid is not timed over
#: three milliseconds and reported to two decimal places.
TARGET_CELL_UPDATES = 4e8
MIN_STEPS = 10
WARMUP = 3
WALLS = WallAdmittances.uniform(0.1)

#: Array passes per cell per step, counted from the NumPy and PyTorch implementations: 21 in
#: the velocity update (three axes, each a difference into a temporary, a scale, and an
#: in-place subtract) and 20 in the pressure update.
ARRAY_PASSES = 41
#: Bytes per cell per step for the fused C kernel: 28 in the velocity sweep (p read once, the
#: three velocity components read and written) and 20 in the pressure sweep.
FUSED_BYTES_PER_ELEMENT = 12

ROOM = (5.0, 4.0, 3.0)
BANDWIDTHS = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
MEMORY_CEILING = 64 * 1024**3
#: Fields resident during a room run: pressure, three velocities, three relaxation states.
FIELDS_IN_MEMORY = 7


def _torch():
    try:
        import torch

        from ac_fdtd import torch_backend

        return torch, torch_backend
    except ImportError:
        return None, None


def _c_backend():
    try:
        from ac_fdtd import c_backend

        return c_backend if c_backend.is_available() else None
    except ImportError:
        return None


def stream_ceilings() -> dict[str, float]:
    """Measured bandwidth of a fused add, per device — the number to be judged against.

    A specification sheet quotes what the memory controller can do; this is what a kernel that
    does nothing but move data achieves, which is the fairer comparison. Best of several trials
    rather than the mean: a ceiling is what the hardware *can* do, and every source of noise
    here subtracts from it.
    """
    torch, backend = _torch()
    ceilings = {}
    if torch is None:
        return ceilings
    for device in backend.available_devices():
        left = torch.randn(320, 320, 320, device=device)
        right = torch.randn(320, 320, 320, device=device)
        best = 0.0
        for _ in range(5):
            for _ in range(2):
                left.add_(right)
            backend.synchronize(device)
            started = time.perf_counter()
            for _ in range(10):
                left.add_(right)
            backend.synchronize(device)
            elapsed = (time.perf_counter() - started) / 10
            best = max(best, 3 * left.numel() * 4 / elapsed / 1e9)
        ceilings[device] = best
        del left, right
        gc.collect()
    return ceilings


def build_cases():
    """Every backend available here, with the bytes each moves per cell per step."""
    torch, torch_backend = _torch()
    c_backend = _c_backend()

    cases = [
        {
            "name": "NumPy fp64",
            "family": "NumPy",
            "bytes": ARRAY_PASSES * 8,
            "ceiling": "cpu",
            "make": lambda grid: AcousticFDTD(grid, walls=WALLS),
        }
    ]
    if torch is not None:
        cases.append(
            {
                "name": "Torch CPU fp32",
                "family": "Torch",
                "bytes": ARRAY_PASSES * 4,
                "ceiling": "cpu",
                "make": lambda grid: torch_backend.TorchAcousticFDTD(
                    grid, device="cpu", dtype=torch.float32, walls=WALLS
                ),
            }
        )
        if "mps" in torch_backend.available_devices():
            cases.append(
                {
                    "name": "Torch MPS fp32",
                    "family": "Torch",
                    "bytes": ARRAY_PASSES * 4,
                    "ceiling": "mps",
                    "make": lambda grid: torch_backend.TorchAcousticFDTD(
                        grid, device="mps", dtype=torch.float32, walls=WALLS
                    ),
                }
            )
    if c_backend is not None:
        for label, dtype in (("C fp64", np.float64), ("C fp32", np.float32)):
            cases.append(
                {
                    "name": label,
                    "family": "C",
                    "bytes": FUSED_BYTES_PER_ELEMENT * np.dtype(dtype).itemsize,
                    "ceiling": "cpu",
                    "make": (
                        lambda grid, dtype=dtype: c_backend.CAcousticFDTD(
                            grid, dtype=dtype, walls=WALLS
                        )
                    ),
                }
            )
    return cases


def measure(case, size: int) -> float:
    """Cell-updates per second, after a warm-up and with the device flushed before stopping."""
    grid = Grid(dx=0.01, nx=size, ny=size, nz=size)
    steps = max(MIN_STEPS, int(TARGET_CELL_UPDATES / grid.n_cells))
    solver = case["make"](grid)
    solver.run(WARMUP)
    if hasattr(solver, "synchronize"):
        solver.synchronize()

    started = time.perf_counter()
    solver.run(steps)
    if hasattr(solver, "synchronize"):
        solver.synchronize()
    elapsed = time.perf_counter() - started

    del solver
    gc.collect()
    torch, _ = _torch()
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return grid.n_cells * steps / elapsed


def room_cost(max_frequency: float, throughput: float, points_per_wavelength: float):
    """Cells, memory and seconds of compute for one second of impulse response."""
    grid = Grid.from_max_frequency(
        ROOM, max_frequency, AIR.sound_speed, points_per_wavelength=points_per_wavelength
    )
    dt = grid.dx / (AIR.sound_speed * np.sqrt(3.0))
    steps = 1.0 / dt
    return {
        "grid": grid,
        "cells": grid.n_cells,
        "steps": steps,
        "seconds": grid.n_cells * steps / throughput,
    }


def main() -> None:
    plotstyle.apply()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    ceilings = stream_ceilings()
    cases = build_cases()

    results = {case["name"]: [] for case in cases}
    for size in SIZES:
        for case in cases:
            results[case["name"]].append(measure(case, size))
        print(
            f"  {size}^3 done ({size**3 / 1e6:.1f} M cells)",
            flush=True,
        )

    cells = np.array([size**3 for size in SIZES], dtype=float)
    styles = {
        "NumPy": {"color": plotstyle.SERIES[0]},
        "Torch": {"color": plotstyle.SERIES[1]},
        "C": {"color": plotstyle.SERIES[2]},
    }
    dashes = {
        "NumPy fp64": (1, 0),
        "Torch CPU fp32": (1, 0),
        "Torch MPS fp32": (5, 2),
        "C fp64": (5, 2),
        "C fp32": (1, 0),
    }
    markers = {
        "NumPy fp64": "o",
        "Torch CPU fp32": "o",
        "Torch MPS fp32": "s",
        "C fp64": "^",
        "C fp32": "o",
    }

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # --- throughput --------------------------------------------------------------------
    ax = axes[0]
    for case in cases:
        name = case["name"]
        ax.loglog(
            cells / 1e6,
            np.array(results[name]) / 1e9,
            marker=markers[name],
            dashes=dashes[name],
            color=styles[case["family"]]["color"],
            label=name,
        )
    ax.set_xlabel("cells  [millions]")
    ax.set_ylabel("throughput  [Gcell-updates/s]")
    ax.set_title("The fused C loop leads at every size")
    ax.legend(loc="lower right", fontsize=8)

    # --- roofline ----------------------------------------------------------------------
    #
    # Plotted as bandwidth rather than as a percentage of the ceiling. The traffic model is a
    # count of array passes, good to perhaps twenty percent, and dividing one approximation by
    # another produced a backend at "108 % of the hardware" -- a number that says more about
    # the model than about the run.
    ax = axes[1]
    for case in cases:
        name = case["name"]
        ax.semilogx(
            cells / 1e6,
            np.array(results[name]) * case["bytes"] / 1e9,
            marker=markers[name],
            dashes=dashes[name],
            color=styles[case["family"]]["color"],
            label=name,
        )
    for device, value in ceilings.items():
        ax.axhline(value, color=plotstyle.MUTED, lw=1.2, ls="--")
        ax.text(0.32, value * 1.03, f"{device} stream ceiling", fontsize=8, color=plotstyle.MUTED)
    ax.set_ylim(0, 1.15 * max(ceilings.values(), default=1.0))
    ax.set_xlabel("cells  [millions]")
    ax.set_ylabel("memory traffic achieved  [GB/s]")
    ax.set_title("The same bandwidth; different amounts of it needed")

    # --- what a room costs -------------------------------------------------------------
    ax = axes[2]
    resolution = required_points_per_wavelength(0.01)
    frequencies = np.geomspace(BANDWIDTHS[0], BANDWIDTHS[-1], 40)
    for case in cases:
        name = case["name"]
        # Interpolated in log-cells so the small-grid penalty is not quietly extrapolated away.
        peak = np.interp(
            np.log10([room_cost(f, 1.0, resolution)["cells"] for f in frequencies]),
            np.log10(cells),
            results[name],
        )
        seconds = [
            room_cost(f, throughput, resolution)["seconds"]
            for f, throughput in zip(frequencies, peak, strict=True)
        ]
        ax.loglog(
            frequencies,
            np.array(seconds) / 3600.0,
            dashes=dashes[name],
            color=styles[case["family"]]["color"],
            label=name,
        )
    ax.axhline(1.0, color=plotstyle.MUTED, lw=1.2, ls="--")
    ax.text(520, 1.15, "one hour", fontsize=8, color=plotstyle.MUTED)
    ax.set_xticks(list(BANDWIDTHS))
    ax.set_xticklabels([f"{value:.0f}" for value in BANDWIDTHS])
    # A log axis keeps labelling its minor ticks otherwise, on top of the ones just set.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("bandwidth resolved  [Hz]")
    ax.set_ylabel("hours per second of response")
    ax.set_title(
        f"{ROOM[0]:.0f}x{ROOM[1]:.0f}x{ROOM[2]:.0f} m room, {resolution:.1f} points per wavelength"
    )
    ax.legend(loc="upper left", fontsize=8)

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")

    print("\nmeasured stream bandwidth (fused add, 16.8 M float32):")
    for device, value in ceilings.items():
        print(f"  {device:>4s}: {value:6.0f} GB/s")

    print("\nthroughput, Gcell-updates/s\n")
    print("  cells    " + " ".join(f"{case['name']:>15s}" for case in cases))
    for index, size in enumerate(SIZES):
        row = " ".join(f"{results[case['name']][index] / 1e9:15.2f}" for case in cases)
        print(f"  {size:3d}^3 {size**3 / 1e6:6.1f}M {row}")

    print("\nmemory traffic at the largest grid (model good to about 20 %)\n")
    for case in cases:
        achieved = results[case["name"]][-1] * case["bytes"] / 1e9
        ceiling = ceilings.get(case["ceiling"])
        if ceiling:
            print(
                f"  {case['name']:>15s}: {achieved:6.0f} GB/s against a {ceiling:.0f} GB/s "
                f"ceiling, moving {case['bytes']:3d} bytes per cell per step"
            )

    print(
        f"\ndecision table — {ROOM[0]:.0f}x{ROOM[1]:.0f}x{ROOM[2]:.0f} m room, "
        f"{resolution:.1f} points per wavelength, one second of response.\n"
        f"Above {cells[-1] / 1e6:.0f} M cells the throughput is held at its measured plateau, "
        f"which the last three sizes support but which memory pressure could still spoil.\n"
    )
    header = "  band       cells    memory " + " ".join(f"{case['name']:>15s}" for case in cases)
    print(header)
    for frequency in BANDWIDTHS:
        reference = room_cost(frequency, 1.0, resolution)
        memory = reference["grid"].memory_bytes(itemsize=4, extra_fields=FIELDS_IN_MEMORY - 4)
        times = []
        for case in cases:
            throughput = float(
                np.interp(np.log10(reference["cells"]), np.log10(cells), results[case["name"]])
            )
            seconds = reference["cells"] * reference["steps"] / throughput
            times.append(f"{seconds:14.0f}s" if seconds < 3600 else f"{seconds / 3600:13.1f}h")
        flag = "  << over 64 GB" if memory > MEMORY_CEILING else ""
        print(
            f"  {frequency:5.0f} Hz {reference['cells'] / 1e6:8.1f}M "
            f"{memory / 1024**3:6.1f}GB " + " ".join(times) + flag
        )

    print(f"\nwrote {FIGURE_PATH}  ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
