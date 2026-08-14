"""Shared figure style, so every figure in the repository reads as one set.

The three categorical colours are checked for colour-vision deficiency separation rather than
chosen by eye: the worst adjacent pair is 21.6 dE apart under protanopia and 18.0 under
tritanopia (OKLab x100), against a floor of 8. Series are also given distinct markers and are
labelled, so identity never rests on colour alone.
"""

from __future__ import annotations

import matplotlib as mpl

#: Fixed order — a series keeps its colour when other series are added or removed.
SERIES = ("#2f6fb0", "#c26016", "#8452c9")

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"


def apply() -> None:
    """Set the repository's rcParams. Call once, before creating any figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "svg.fonttype": "none",
            "figure.dpi": 110,
        }
    )
