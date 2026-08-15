"""A picture of what the solver is actually doing: a pulse crossing a room and coming back.

Not a validation — everything here is checked quantitatively elsewhere. This exists because a
field of numbers is hard to have intuitions about, and four snapshots of a horizontal slice
show the things the tables cannot: the wavefront curvature, the first reflections arriving from
each wall, and the interference pattern that is the beginning of the reverberant field.

Run: ``uv run python scripts/example_wavefront.py``  (seconds)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotstyle  # noqa: E402

from ac_fdtd import (  # noqa: E402
    AIR,
    AcousticFDTD,
    Grid,
    WallAdmittances,
    band_limited_impulse,
)

FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "wavefront.svg"

ROOM = (4.0, 3.0, 2.5)
MAX_FREQUENCY = 800.0
SOURCE = (0.8, 0.7, 1.2)
ABSORPTION = 0.1
SNAPSHOT_TIMES = (4e-3, 9e-3, 16e-3, 28e-3)

#: Two hues from opposite ends of the palette, meeting at a neutral: pressure is signed, and a
#: sequential ramp would make rarefaction look like silence.
SIGNED = LinearSegmentedColormap.from_list(
    "signed", [plotstyle.SERIES[0], "#a8c4de", "#f2f2f0", "#eec39a", plotstyle.SERIES[1]]
)


def main() -> None:
    plotstyle.apply()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    grid = Grid.from_max_frequency(ROOM, MAX_FREQUENCY, AIR.sound_speed)
    solver = AcousticFDTD(grid, walls=WallAdmittances.from_absorption(ABSORPTION))

    total_steps = int(max(SNAPSHOT_TIMES) / solver.dt) + 1
    solver.add_volume_source(SOURCE, band_limited_impulse(total_steps, solver.dt, MAX_FREQUENCY))

    height = grid.cell_index(SOURCE)[2]
    snapshots = []
    for instant in SNAPSHOT_TIMES:
        solver.run(int(instant / solver.dt) - solver.step_index)
        snapshots.append(solver.p[:, :, height].copy())

    # One scale for every panel, so the decay between them is visible rather than normalised
    # away. Set from the second snapshot: the first is still near the source and would flatten
    # everything after it.
    scale = np.max(np.abs(snapshots[1]))

    figure, axes = plt.subplots(1, len(snapshots), figsize=(14.5, 3.6))
    for ax, instant, field in zip(axes, SNAPSHOT_TIMES, snapshots, strict=True):
        ax.imshow(
            field.T,
            origin="lower",
            cmap=SIGNED,
            vmin=-scale,
            vmax=scale,
            extent=(0, grid.lengths[0], 0, grid.lengths[1]),
            interpolation="bilinear",
        )
        ax.plot(*SOURCE[:2], "o", color=plotstyle.INK, markersize=5)
        clipping = np.max(np.abs(field)) / scale
        note = f", clipped {clipping:.0f}x" if clipping > 1.5 else ""
        ax.set_title(f"{1e3 * instant:.0f} ms — {AIR.sound_speed * instant:.1f} m travelled{note}")
        ax.set_xlabel("x  [m]")
        ax.grid(False)
    axes[0].set_ylabel("y  [m]")

    figure.suptitle(
        f"{ROOM[0]:.0f} x {ROOM[1]:.0f} x {ROOM[2]:.1f} m room, walls at "
        f"{ABSORPTION:.2f} absorption, horizontal slice through the source",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")

    print(
        f"grid {grid.shape} at {grid.dx * 1e3:.0f} mm, "
        f"{total_steps} steps of {solver.dt * 1e6:.1f} us"
    )
    print(
        "peak pressure in each snapshot, relative to the second: "
        + ", ".join(f"{np.max(np.abs(field)) / scale:.2f}" for field in snapshots)
    )
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
