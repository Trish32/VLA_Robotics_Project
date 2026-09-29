"""Shared visual language for the figures on the repository front page.

One place for the palette and the type scale so four panels read as one set rather than
four charts that happened to be made the same week. Everything here is chosen against a
white page, because that is what GitHub renders a README on in light mode, and the
figures carry their own background so dark mode does not show black text on nothing.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

#: Ink on paper, not pure black on pure white: #14181f reads softer at figure scale.
INK = "#14181f"
MUTED = "#5b6472"
FAINT = "#aab2be"
PAPER = "#ffffff"
PANEL = "#f6f7f9"

#: One accent per meaning, used identically across every panel.
GOOD = "#1f8a4c"          # the policy, and anything that works
BAD = "#c2410c"           # the failing arm: greedy / oracle selection
COOL = "#2d6cdf"          # the model's own prediction
WARM = "#b45309"           # ground truth it is compared against
CUBE = "#2b2f3a"
BOWL = "#d97706"
BALL = "#3f9c46"


def use() -> None:
    """Apply the shared rcParams. Call once per figure, before creating axes."""
    matplotlib.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": FAINT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })


def caption(ax, text: str, *, y: float = -0.19, width: int = 118) -> None:
    """One line under a panel, in the muted ink. Captions carry the numbers.

    Wrapped here rather than with matplotlib's `wrap=True`, which measures against the
    figure and not the axes, so a caption under a wide panel runs off the page while
    still reporting that it wrapped.
    """
    import textwrap

    ax.text(0.0, y, "\n".join(textwrap.wrap(text, width)), transform=ax.transAxes,
            ha="left", va="top", fontsize=8.2, color=MUTED, linespacing=1.45)


def verdict(ax, text: str, colour: str = GOOD) -> None:
    """The one-phrase claim a panel exists to make, set above its title."""
    ax.text(0.0, 1.16, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=colour, fontweight="bold")
