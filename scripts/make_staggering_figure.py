"""A picture of the one convention the whole solver rests on: the staggered grid.

Pressure and velocity are offset from each other in *both* space and time, and the two offsets
do different jobs. The spatial one puts each velocity component on the faces it flows through,
which is what makes the wall condition land exactly on the boundary plane. The temporal one is
what makes the leapfrog second-order accurate — and it is the one that quietly biases results
when it is forgotten, because nothing about a half-step error announces itself.

Run: ``uv run python scripts/make_staggering_figure.py``  (instant)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotstyle  # noqa: E402

FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "staggering.svg"

PRESSURE = plotstyle.SERIES[0]
VELOCITY = plotstyle.SERIES[1]


def draw_space(ax) -> None:
    """One cell, with pressure at its centre and each velocity component on its own faces."""
    for x in (0, 1, 2):
        ax.plot([x, x], [0, 2], color=plotstyle.GRID, lw=1.2, zorder=0)
        ax.plot([0, 2], [x, x], color=plotstyle.GRID, lw=1.2, zorder=0)

    for i in (0, 1):
        for j in (0, 1):
            ax.plot(i + 0.5, j + 0.5, "o", color=PRESSURE, markersize=9, zorder=3)
    ax.text(1.5, 1.62, "$p_{i,j}$", color=PRESSURE, ha="center", fontsize=11)

    # x-velocity on the vertical faces, y-velocity on the horizontal ones.
    for x in (0, 1, 2):
        for y in (0.5, 1.5):
            ax.annotate(
                "",
                xy=(x + 0.22, y),
                xytext=(x - 0.22, y),
                arrowprops={"arrowstyle": "-|>", "color": VELOCITY, "lw": 1.8},
            )
    for y in (0, 1, 2):
        for x in (0.5, 1.5):
            ax.annotate(
                "",
                xy=(x, y + 0.22),
                xytext=(x, y - 0.22),
                arrowprops={"arrowstyle": "-|>", "color": VELOCITY, "lw": 1.8},
            )
    ax.text(2.30, 1.5, "$v_{x,\\,i+1/2}$", color=VELOCITY, ha="left", va="center", fontsize=11)
    ax.text(1.5, 2.22, "$v_{y,\\,j+1/2}$", color=VELOCITY, ha="center", fontsize=11)

    ax.text(
        1.0,
        -0.45,
        "Each velocity component sits on the faces it flows through,\n"
        "so a rigid wall is a face held at zero — exactly on the boundary.",
        ha="center",
        va="top",
        fontsize=9,
        color=plotstyle.MUTED,
    )
    ax.set_xlim(-0.6, 3.0)
    ax.set_ylim(-1.15, 2.6)
    ax.set_title("In space: pressure at centres, velocity on faces")
    ax.axis("off")


def draw_time(ax) -> None:
    """The leapfrog: each field is updated from the other one, half a step away."""
    pressure_y, velocity_y = 1.0, 0.0
    instants = [0, 1, 2]

    ax.annotate(
        "",
        xy=(2.75, pressure_y),
        xytext=(-0.55, pressure_y),
        arrowprops={"arrowstyle": "-|>", "color": plotstyle.MUTED, "lw": 1.2},
    )
    ax.annotate(
        "",
        xy=(2.75, velocity_y),
        xytext=(-0.55, velocity_y),
        arrowprops={"arrowstyle": "-|>", "color": plotstyle.MUTED, "lw": 1.2},
    )
    ax.text(2.82, pressure_y, "pressure", color=PRESSURE, va="center", fontsize=10)
    ax.text(2.82, velocity_y, "velocity", color=VELOCITY, va="center", fontsize=10)

    for n in instants:
        ax.plot(n, pressure_y, "o", color=PRESSURE, markersize=10, zorder=3)
        ax.text(
            n,
            pressure_y + 0.22,
            f"$p^{{n{n - 1:+d}}}$".replace("+0", ""),
            color=PRESSURE,
            ha="center",
            fontsize=11,
        )
    for n in instants:
        ax.plot(n - 0.5, velocity_y, "o", color=VELOCITY, markersize=10, zorder=3)

    ax.text(-0.5, velocity_y - 0.30, "$v^{n-3/2}$", color=VELOCITY, ha="center", fontsize=11)
    ax.text(0.5, velocity_y - 0.30, "$v^{n-1/2}$", color=VELOCITY, ha="center", fontsize=11)
    ax.text(1.5, velocity_y - 0.30, "$v^{n+1/2}$", color=VELOCITY, ha="center", fontsize=11)

    # One full step: pressure drives velocity forward, then velocity drives pressure.
    ax.annotate(
        "",
        xy=(1.5, velocity_y + 0.13),
        xytext=(1.0, pressure_y - 0.13),
        arrowprops={"arrowstyle": "-|>", "color": plotstyle.INK, "lw": 1.6},
    )
    ax.annotate(
        "",
        xy=(2.0, pressure_y - 0.13),
        xytext=(1.5, velocity_y + 0.13),
        arrowprops={"arrowstyle": "-|>", "color": plotstyle.INK, "lw": 1.6},
    )
    ax.text(
        1.06,
        0.52,
        "$-\\,\\dfrac{dt}{\\rho}\\,\\nabla p^{\\,n}$",
        fontsize=10,
        color=plotstyle.INK,
        ha="right",
        va="center",
    )
    ax.text(
        1.94,
        0.52,
        "$-\\,\\rho c^{2} dt\\,\\nabla\\!\\cdot v^{\\,n+1/2}$",
        fontsize=10,
        color=plotstyle.INK,
        ha="left",
        va="center",
    )

    # The pair the solver actually holds. Ringed rather than joined by a line, which would
    # cross the update label below it.
    for x, y in ((1.0, pressure_y), (0.5, velocity_y)):
        ax.plot(
            x,
            y,
            "o",
            markersize=19,
            markerfacecolor="none",
            markeredgecolor=plotstyle.SERIES[2],
            markeredgewidth=1.8,
            zorder=4,
        )
    ax.text(
        -0.75,
        pressure_y + 0.62,
        "the stored state is always this pair,\nhalf a step apart:  $(p^{\\,n},\\ v^{\\,n-1/2})$",
        color=plotstyle.SERIES[2],
        ha="left",
        va="center",
        fontsize=10,
    )

    ax.text(
        1.1,
        -0.95,
        "The two fields are never known at the same instant. Anything combining them —\n"
        "energy, intensity, a boundary condition — has to say which one it means.",
        ha="center",
        va="top",
        fontsize=9,
        color=plotstyle.MUTED,
    )
    ax.set_xlim(-0.9, 3.9)
    ax.set_ylim(-1.65, 1.75)
    ax.set_title("In time: half a step apart, forever")
    ax.axis("off")


def main() -> None:
    plotstyle.apply()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": (1, 1.3)})
    draw_space(axes[0])
    draw_time(axes[1])
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
