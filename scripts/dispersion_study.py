"""M7 — how fine does the grid have to be, and why "ten points per wavelength"?

This is the cheapest study in the repository and the one with the most leverage. Halving the
required resolution divides the cell count by eight and the step count by two: a factor of
sixteen in run time, which is more than the difference between the fastest and slowest backend
measured. Choosing the resolution well is worth more than choosing the hardware well.

None of it needs a simulation. The scheme's dispersion relation is closed-form, so the phase
velocity error is an exact function of resolution, direction and Courant number.

Run: ``uv run python scripts/dispersion_study.py``  (seconds)
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

from ac_fdtd.dispersion import (  # noqa: E402
    NAMED_DIRECTIONS,
    direction_map,
    phase_velocity_ratio,
    required_points_per_wavelength,
    worst_direction_error,
)

FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m7_dispersion.svg"

RESOLUTIONS = np.geomspace(4.0, 100.0, 200)
MAP_RESOLUTION = 10.0
COURANT_NUMBERS = (1.0, 0.9, 0.5)
TARGET_ERRORS = np.geomspace(1e-4, 0.1, 60)
REPORTED_TARGETS = (0.05, 0.02, 0.01, 0.005, 0.001)

#: One hue, light to dark: this panel shows a magnitude, not a category or a polarity.
MAGNITUDE = LinearSegmentedColormap.from_list(
    "magnitude", ["#f4f7fb", "#9dc0e0", plotstyle.SERIES[0], "#123a63"]
)


def main() -> None:
    plotstyle.apply()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # --- error against resolution ------------------------------------------------------
    ax = axes[0]
    for (name, direction), colour in zip(NAMED_DIRECTIONS.items(), plotstyle.SERIES, strict=True):
        error = np.abs(phase_velocity_ratio(RESOLUTIONS, direction) - 1.0)
        if np.max(error) < 1e-12:
            # The body diagonal is exact, and a log axis cannot draw zero. Drawn along the
            # bottom of the frame instead, and labelled so the placement is not mistaken for
            # a measurement.
            ax.axhline(1.3e-5, color=colour, lw=2.0, label=f"{name} — exactly zero")
            continue
        ax.loglog(RESOLUTIONS, 100 * error, color=colour, label=name, zorder=3)

    reference = 100 * np.abs(phase_velocity_ratio(np.array([10.0]), [1, 0, 0])[0] - 1.0)
    ax.loglog(
        RESOLUTIONS,
        reference * (10.0 / RESOLUTIONS) ** 2,
        color=plotstyle.INK,
        lw=1.2,
        dashes=(6, 4),
        zorder=5,
        label="$n^{-2}$, through the axis curve",
    )
    ax.set_ylim(1e-5, 20)
    ax.set_xlabel("points per wavelength")
    ax.set_ylabel("phase velocity error  [%]")
    ax.set_title("The body diagonal is exact at the stability limit")
    ax.legend(loc="lower left")

    # --- anisotropy --------------------------------------------------------------------
    ax = axes[1]
    azimuth, elevation, error = direction_map(MAP_RESOLUTION)
    mesh = ax.pcolormesh(
        np.degrees(azimuth),
        np.degrees(elevation),
        100 * np.abs(error),
        cmap=MAGNITUDE,
        shading="gouraud",
    )
    figure.colorbar(mesh, ax=ax, label="phase velocity error  [%]")
    ax.plot([0, 90, 0], [0, 0, 90], "o", color=plotstyle.SERIES[1], markersize=7)
    ax.plot(
        [45],
        [np.degrees(np.arctan(1 / np.sqrt(2)))],
        "D",
        color="white",
        markeredgecolor=plotstyle.INK,
        markersize=8,
    )
    ax.text(46, 37, "body diagonal", fontsize=8, color=plotstyle.INK)
    ax.text(4, 4, "axes", fontsize=8, color=plotstyle.SERIES[1])
    ax.set_xlabel("azimuth  [deg]")
    ax.set_ylabel("elevation  [deg]")
    ax.set_title(f"Anisotropy at {MAP_RESOLUTION:.0f} points per wavelength")
    ax.grid(False)

    # --- the design rule ---------------------------------------------------------------
    ax = axes[2]
    for courant, colour in zip(COURANT_NUMBERS, plotstyle.SERIES, strict=True):
        needed = [required_points_per_wavelength(t, courant) for t in TARGET_ERRORS]
        ax.loglog(100 * TARGET_ERRORS, needed, color=colour, label=f"Courant {courant}")
    ax.plot(
        [100 * 0.01],
        [required_points_per_wavelength(0.01)],
        "o",
        color=plotstyle.SERIES[0],
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2,
    )
    ax.annotate(
        "1 % costs 10.5 points",
        xy=(1.0, required_points_per_wavelength(0.01)),
        xytext=(1.4, 6.0),
        fontsize=8,
        color=plotstyle.INK,
        arrowprops={"arrowstyle": "-", "color": plotstyle.MUTED, "lw": 1},
    )
    ax.set_xlabel("phase velocity error target  [%]")
    ax.set_ylabel("points per wavelength needed")
    ax.set_title("Running below the limit costs resolution")
    ax.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")

    print("phase velocity error by direction, Courant 1.0\n")
    header = "  ppw " + " ".join(f"{name:>22s}" for name in NAMED_DIRECTIONS)
    print(header + "        worst")
    for resolution in (5, 8, 10, 15, 20, 30, 60):
        row = [
            100 * (phase_velocity_ratio(np.array([resolution]), direction)[0] - 1.0)
            for direction in NAMED_DIRECTIONS.values()
        ]
        print(
            f"  {resolution:3d} "
            + " ".join(f"{value:+21.4f}%" for value in row)
            + f"  {100 * worst_direction_error(resolution):8.4f}%"
        )

    print("\nresolution needed for a target error, by Courant number\n")
    print("  target      1.0     0.9     0.5   cells vs Courant 1.0")
    for target in REPORTED_TARGETS:
        needed = [required_points_per_wavelength(target, c) for c in COURANT_NUMBERS]
        print(
            f"  {100 * target:5.2f} % "
            + " ".join(f"{value:7.1f}" for value in needed)
            + f"   {(needed[2] / needed[0]) ** 3:6.2f}x at 0.5"
        )

    print(f"\nwrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
