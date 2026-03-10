"""
Recommendation engine for varsplit.diagnose.

Takes computed metrics and produces:
    - A novelty score per dimension (0-1)
    - A recommended strategy
    - A confidence level
    - A configured VarSplit object ready to use
    - A formatted printable summary
"""

from dataclasses import dataclass, field
from varsplit.core import VarSplit


# Novelty threshold below which a dimension is considered "not a driver"
_LOW_NOVELTY = 0.05
# Threshold above which confidence is "high"
_HIGH_CONFIDENCE = 0.25


@dataclass
class DimensionResult:
    name: str
    novelty: float          # 0-1
    primary_metric: str     # human-readable key metric
    note: str               # one-line explanation


@dataclass
class RecommendationResult:
    recommended_strategy: str
    confidence: str                          # "high" | "moderate" | "low"
    rationale: str                           # one-line top-level summary
    dimensions: dict[str, DimensionResult]  # per-dimension breakdown
    varsplit: VarSplit                        # ready-to-use splitter

    def print_summary(self):
        """Print a formatted summary to terminal."""
        width = 60
        bar_width = 20

        def bar(novelty):
            filled = round(novelty * bar_width)
            return "▓" * filled + "░" * (bar_width - filled)

        print()
        print("=" * width)
        print(f"  Recommendation: {self.recommended_strategy.upper()}")
        print("=" * width)
        print(f"  {self.rationale}")
        print("-" * width)
        print(f"  {'Dimension':<14} {'Novelty':>7}   {'':20}   Note")
        print(f"  {'-'*14} {'-'*7}   {'-'*20}   {'-'*20}")

        # Sort by novelty descending
        sorted_dims = sorted(
            self.dimensions.values(),
            key=lambda d: d.novelty,
            reverse=True,
        )
        for dim in sorted_dims:
            tag = "  PRIMARY" if dim.name == self.recommended_strategy else ""
            print(
                f"  {dim.name:<14} {dim.novelty:>6.0%}   {bar(dim.novelty)}  "
                f"{dim.note}{tag}"
            )

        print("-" * width)
        print(f"  Suggested within-A split:")
        print(f"  VarSplit(strategy=\"{self.recommended_strategy}\")")
        print("=" * width)
        print()

    def to_dict(self) -> dict:
        """Return a plain dict for programmatic use."""
        return {
            "recommended_strategy": self.recommended_strategy,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "dimension_scores": {
                name: {
                    "novelty": round(dim.novelty, 3),
                    "primary_metric": dim.primary_metric,
                    "note": dim.note,
                }
                for name, dim in self.dimensions.items()
            },
        }


def compute_recommendation(
    positional: dict,
    mutational: dict,
    order: dict,
    fitness: dict | None,
    mutation_col: str = "mutations",
    fmt: str = "standard",
    fitness_col: str | None = None,
) -> RecommendationResult:
    """
    Compute novelty scores per dimension and recommend a strategy.

    Args:
        positional:   output of metrics.positional_metrics()
        mutational:   output of metrics.mutational_metrics()
        order:        output of metrics.order_metrics()
        fitness:      output of metrics.fitness_metrics(), or None
        mutation_col: passed through to VarSplit
        fmt:          passed through to VarSplit
        fitness_col:  passed through to VarSplit

    Returns:
        RecommendationResult
    """
    dimensions = {}

    # --- Positional ---
    pct = positional["pct_B_novel"]
    dimensions["positional"] = DimensionResult(
        name="positional",
        novelty=pct,
        primary_metric=f"pct_B_novel={pct:.1%}",
        note=f"{pct:.0%} of B positions not in A",
    )

    # --- Mutational ---
    pct = mutational["pct_B_novel"]
    dimensions["mutational"] = DimensionResult(
        name="mutational",
        novelty=pct,
        primary_metric=f"pct_B_novel={pct:.1%}",
        note=f"{pct:.0%} of B substitutions not in A",
    )

    # --- Order ---
    tvd = order["novelty_score"]
    shift = order["mean_shift"]
    direction = "higher" if shift > 0 else "lower"
    dimensions["order"] = DimensionResult(
        name="order",
        novelty=tvd,
        primary_metric=f"tvd={tvd:.3f}",
        note=f"B order {direction} by {abs(shift):.1f} mutations on avg",
    )

    # --- Fitness (optional) ---
    if fitness is not None:
        ext = fitness["novelty_score"]
        dimensions["fitness"] = DimensionResult(
            name="fitness",
            novelty=ext,
            primary_metric=f"pct_extrapolates={ext:.1%}",
            note=f"{ext:.0%} of B outside A fitness range",
        )

    # --- Pick winner ---
    best_name = max(dimensions, key=lambda k: dimensions[k].novelty)
    best_novelty = dimensions[best_name].novelty
    second_novelty = sorted(
        [d.novelty for d in dimensions.values()], reverse=True
    )[1] if len(dimensions) > 1 else 0.0

    gap = best_novelty - second_novelty
    if gap >= _HIGH_CONFIDENCE and best_novelty > _LOW_NOVELTY:
        confidence = "high"
    elif best_novelty > _LOW_NOVELTY:
        confidence = "moderate"
    else:
        confidence = "low"

    rationale = (
        f"{dimensions[best_name].note}. "
        f"Recommend {best_name} holdout within A for validation."
    )

    vs = VarSplit(
        mutation_col=mutation_col,
        fmt=fmt,
        fitness_col=fitness_col if best_name == "fitness" else None,
    )

    return RecommendationResult(
        recommended_strategy=best_name,
        confidence=confidence,
        rationale=rationale,
        dimensions=dimensions,
        varsplit=vs,
    )
