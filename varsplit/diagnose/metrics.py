"""
Metrics for comparing two sets of protein variants.

All metrics operate on parsed MutationSets. Raw strings are handled
upstream in the Diagnose class before reaching here.

Each metric function returns a dict of raw values. Novelty scores
(0-1, where 1 = completely novel) are computed separately in recommender.py
from these raw values.
"""

import numpy as np
from varsplit.parsing.base import MutationSet


# ---------------------------------------------------------------------------
# Positional metrics
# ---------------------------------------------------------------------------

def positional_metrics(
    variants_A: list[MutationSet],
    variants_B: list[MutationSet],
) -> dict:
    """
    Compare the positions mutated in A vs B.

    Returns:
        positions_A:          set of all positions in A
        positions_B:          set of all positions in B
        positions_only_in_B:  positions in B not seen in A (novel to B)
        positions_only_in_A:  positions in A not queried in B
        positions_shared:     positions in both
        pct_B_novel:          fraction of B positions not in A  [0-1]
        pct_A_covered:        fraction of A positions that appear in B [0-1]
    """
    pos_A = {pos for v in variants_A for (pos, wt, mut) in v}
    pos_B = {pos for v in variants_B for (pos, wt, mut) in v}

    only_B = pos_B - pos_A
    only_A = pos_A - pos_B
    shared = pos_A & pos_B

    pct_B_novel = len(only_B) / len(pos_B) if pos_B else 0.0
    pct_A_covered = len(shared) / len(pos_A) if pos_A else 0.0

    return {
        "positions_A": pos_A,
        "positions_B": pos_B,
        "positions_only_in_B": only_B,
        "positions_only_in_A": only_A,
        "positions_shared": shared,
        "n_positions_A": len(pos_A),
        "n_positions_B": len(pos_B),
        "n_shared": len(shared),
        "n_only_B": len(only_B),
        "pct_B_novel": pct_B_novel,
        "pct_A_covered": pct_A_covered,
    }


# ---------------------------------------------------------------------------
# Mutational metrics
# ---------------------------------------------------------------------------

def mutational_metrics(
    variants_A: list[MutationSet],
    variants_B: list[MutationSet],
) -> dict:
    """
    Compare specific (position, mutant_aa) substitutions in A vs B.

    Returns:
        subs_A:          set of all (pos, mut_aa) in A
        subs_B:          set of all (pos, mut_aa) in B
        subs_only_in_B:  substitutions in B not seen in A
        pct_B_novel:     fraction of B substitutions not in A  [0-1]
        pct_A_covered:   fraction of A substitutions that appear in B [0-1]
    """
    subs_A = {(pos, mut) for v in variants_A for (pos, wt, mut) in v}
    subs_B = {(pos, mut) for v in variants_B for (pos, wt, mut) in v}

    only_B = subs_B - subs_A
    shared = subs_A & subs_B

    pct_B_novel = len(only_B) / len(subs_B) if subs_B else 0.0
    pct_A_covered = len(shared) / len(subs_A) if subs_A else 0.0

    return {
        "subs_A": subs_A,
        "subs_B": subs_B,
        "subs_only_in_B": only_B,
        "subs_shared": shared,
        "n_subs_A": len(subs_A),
        "n_subs_B": len(subs_B),
        "n_shared": len(shared),
        "n_only_B": len(only_B),
        "pct_B_novel": pct_B_novel,
        "pct_A_covered": pct_A_covered,
    }


# ---------------------------------------------------------------------------
# Order metrics
# ---------------------------------------------------------------------------

def order_metrics(
    variants_A: list[MutationSet],
    variants_B: list[MutationSet],
) -> dict:
    """
    Compare mutation order (number of mutations per variant) in A vs B.

    Returns:
        orders_A / orders_B:  numpy arrays of per-variant mutation counts
        mean_A / mean_B:      mean order
        std_A / std_B:        stdev of order
        distribution_A/B:     dict of {order: count}
        mean_shift:           mean_B - mean_A (positive = B has higher order)
        novelty_score:        normalized distributional distance [0-1]
    """
    orders_A = np.array([len(v) for v in variants_A])
    orders_B = np.array([len(v) for v in variants_B])

    max_order = int(max(orders_A.max(), orders_B.max())) + 1
    bins = np.arange(0, max_order + 1)

    dist_A = np.histogram(orders_A, bins=bins)[0].astype(float)
    dist_B = np.histogram(orders_B, bins=bins)[0].astype(float)

    # Normalize to proportions
    prop_A = dist_A / dist_A.sum() if dist_A.sum() > 0 else dist_A
    prop_B = dist_B / dist_B.sum() if dist_B.sum() > 0 else dist_B

    # Total variation distance as novelty score (0=identical, 1=completely different)
    tvd = float(0.5 * np.sum(np.abs(prop_A - prop_B)))

    return {
        "orders_A": orders_A,
        "orders_B": orders_B,
        "mean_A": float(orders_A.mean()),
        "mean_B": float(orders_B.mean()),
        "std_A": float(orders_A.std()),
        "std_B": float(orders_B.std()),
        "max_A": int(orders_A.max()),
        "max_B": int(orders_B.max()),
        "distribution_A": {int(k): int(v) for k, v in zip(bins[:-1], dist_A)},
        "distribution_B": {int(k): int(v) for k, v in zip(bins[:-1], dist_B)},
        "mean_shift": float(orders_B.mean() - orders_A.mean()),
        "novelty_score": tvd,
    }


# ---------------------------------------------------------------------------
# Fitness metrics
# ---------------------------------------------------------------------------

def fitness_metrics(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
) -> dict:
    """
    Compare fitness score distributions in A vs B.

    Returns:
        mean / std / min / max for each set
        pct_B_above_A_max:  fraction of B variants with score > max(A)
        pct_B_below_A_min:  fraction of B variants with score < min(A)
        pct_B_extrapolates: fraction of B outside the range of A entirely
        novelty_score:      driven by extrapolation fraction [0-1]
    """
    scores_A = np.asarray(scores_A, dtype=float)
    scores_B = np.asarray(scores_B, dtype=float)

    above_max = float(np.mean(scores_B > scores_A.max()))
    below_min = float(np.mean(scores_B < scores_A.min()))
    extrapolates = above_max + below_min  # fraction outside A's range

    return {
        "mean_A": float(scores_A.mean()),
        "mean_B": float(scores_B.mean()),
        "std_A": float(scores_A.std()),
        "std_B": float(scores_B.std()),
        "min_A": float(scores_A.min()),
        "max_A": float(scores_A.max()),
        "min_B": float(scores_B.min()),
        "max_B": float(scores_B.max()),
        "mean_shift": float(scores_B.mean() - scores_A.mean()),
        "pct_B_above_A_max": above_max,
        "pct_B_below_A_min": below_min,
        "pct_B_extrapolates": min(extrapolates, 1.0),
        "novelty_score": min(extrapolates, 1.0),
    }


# ---------------------------------------------------------------------------
# Dataset-level summary
# ---------------------------------------------------------------------------

def dataset_summary(
    variants_A: list[MutationSet],
    variants_B: list[MutationSet],
) -> dict:
    """Basic size and composition stats for each set."""
    return {
        "n_variants_A": len(variants_A),
        "n_variants_B": len(variants_B),
        "n_wildtype_A": sum(1 for v in variants_A if len(v) == 0),
        "n_wildtype_B": sum(1 for v in variants_B if len(v) == 0),
        "size_ratio": len(variants_B) / len(variants_A) if variants_A else None,
    }
