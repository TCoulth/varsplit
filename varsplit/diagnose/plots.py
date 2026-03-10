"""
Visualization functions for varsplit.diagnose.

Each function takes precomputed metric dicts and returns a matplotlib Figure.
All plots use a consistent two-color style (A=blue, B=orange).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

_COLOR_A = "#4878CF"
_COLOR_B = "#E8762B"
_COLOR_SHARED = "#6BAF92"
_GRAY = "#AAAAAA"


def _base_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_order_distribution(
    order_metrics: dict,
    label_A: str = "train",
    label_B: str = "test",
) -> Figure:
    """
    Overlapping histograms of mutations-per-variant counts in A and B.
    """
    orders_A = order_metrics["orders_A"]
    orders_B = order_metrics["orders_B"]

    max_order = int(max(orders_A.max(), orders_B.max()))
    bins = np.arange(0.5, max_order + 1.5, 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(orders_A, bins=bins, color=_COLOR_A, alpha=0.6, label=label_A,
            density=True, rwidth=0.85)
    ax.hist(orders_B, bins=bins, color=_COLOR_B, alpha=0.6, label=label_B,
            density=True, rwidth=0.85)

    ax.set_xticks(range(1, max_order + 1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)

    mean_A = order_metrics["mean_A"]
    mean_B = order_metrics["mean_B"]
    shift  = order_metrics["mean_shift"]
    ratio  = mean_B / mean_A if mean_A > 0 else float("inf")
    ax.text(
        0.98, 0.95,
        f"mean {label_A}={mean_A:.2f}   mean {label_B}={mean_B:.2f}\n"
        f"shift={shift:+.2f}  ({ratio:.1f}×)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="#555",
    )

    _base_style(ax, "Mutations per Variant", "Number of mutations", "Proportion")
    fig.tight_layout()
    return fig


def _novelty_donuts(
    n_shared: int,
    n_only_A: int,
    n_only_B: int,
    label_A: str,
    label_B: str,
    title: str,
    label_shared: str = "Shared",
    label_only_A: str = None,
    label_only_B: str = None,
) -> Figure:
    """
    Shared helper: two donut charts side by side.
    Left = label_A (shared vs only-A), Right = label_B (shared vs novel).
    """
    label_only_A = label_only_A or f"Only in {label_A}"
    label_only_B = label_only_B or f"New in {label_B}"

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Left: A -- shared vs only-A
    axes[0].pie(
        [n_shared, n_only_A],
        labels=[label_shared, label_only_A],
        colors=[_COLOR_SHARED, _COLOR_A],
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"width": 0.5},
        textprops={"fontsize": 9},
    )
    axes[0].set_title(label_A.capitalize(), fontsize=11, fontweight="bold")

    # Right: B -- shared vs novel
    axes[1].pie(
        [n_shared, n_only_B],
        labels=[label_shared, label_only_B],
        colors=[_COLOR_SHARED, _COLOR_B],
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"width": 0.5},
        textprops={"fontsize": 9},
    )
    axes[1].set_title(label_B.capitalize(), fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_positional_overlap(
    positional_metrics: dict,
    label_A: str = "train",
    label_B: str = "test",
) -> Figure:
    """
    Two donut charts showing position overlap between A and B.
    """
    return _novelty_donuts(
        n_shared  = positional_metrics["n_shared"],
        n_only_A  = len(positional_metrics["positions_only_in_A"]),
        n_only_B  = positional_metrics["n_only_B"],
        label_A   = label_A,
        label_B   = label_B,
        title     = "Positional Overlap",
        label_shared  = "Shared positions",
        label_only_A  = f"Only in {label_A}",
        label_only_B  = f"New positions in {label_B}",
    )


def plot_mutational_novelty(
    mutational_metrics: dict,
    label_A: str = "train",
    label_B: str = "test",
) -> Figure:
    """
    Two donut charts showing mutation (substitution) overlap between A and B.
    """
    return _novelty_donuts(
        n_shared  = mutational_metrics["n_shared"],
        n_only_A  = mutational_metrics["n_subs_A"] - mutational_metrics["n_shared"],
        n_only_B  = mutational_metrics["n_only_B"],
        label_A   = label_A,
        label_B   = label_B,
        title     = "Mutational Novelty",
        label_shared  = "Shared mutations",
        label_only_A  = f"Only in {label_A}",
        label_only_B  = f"New mutations in {label_B}",
    )


def plot_fitness_distributions(
    fitness_metrics: dict,
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    label_A: str = "train",
    label_B: str = "test",
) -> Figure:
    """
    Overlapping histograms of fitness distributions. Shaded regions
    show where B falls outside A's range.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    all_scores = np.concatenate([scores_A, scores_B])
    bins = np.linspace(all_scores.min(), all_scores.max(), 30)

    ax.hist(scores_A, bins=bins, color=_COLOR_A, alpha=0.6, label=label_A, density=True)
    ax.hist(scores_B, bins=bins, color=_COLOR_B, alpha=0.6, label=label_B, density=True)

    a_min, a_max = scores_A.min(), scores_A.max()
    ax.axvline(a_min, color=_COLOR_A, linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(a_max, color=_COLOR_A, linestyle="--", linewidth=1, alpha=0.7)
    ax.axvspan(all_scores.min(), a_min, alpha=0.08, color=_COLOR_B)
    ax.axvspan(a_max, all_scores.max(), alpha=0.08, color=_COLOR_B)

    pct_ext = fitness_metrics["pct_B_extrapolates"]
    ax.text(
        0.98, 0.95,
        f"{pct_ext:.0%} of {label_B} outside {label_A} range",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#555",
    )

    ax.legend(fontsize=9)
    _base_style(ax, "Fitness Score Distributions", "Fitness score", "Density")
    fig.tight_layout()
    return fig
