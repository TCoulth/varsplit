"""
Fitness-stratified split strategy.

Splits by fitness/score value rather than by mutation content.
Train on one region of the fitness distribution, test on another.

Models the question: "Can my model extrapolate toward high (or low)
fitness variants when trained on the bulk of the distribution?"

Two modes:
    - Quantile: top/bottom X% of variants by fitness go to test
    - Threshold: everything above/below an absolute value goes to test

The split percentage is reported. Due to ties at the threshold,
exact fractions may not be achievable -- actual percentages are printed.
"""

import numpy as np
import pandas as pd
from .base import BaseSplitStrategy
from varsplit.parsing.base import MutationSet


class FitnessSplitStrategy(BaseSplitStrategy):
    """
    Split by fitness score.

    Test set is the upper or lower tail of the fitness distribution,
    defined either by quantile or by an absolute threshold.
    """

    def split(
        self,
        variants: list[MutationSet],
        test_size: float = 0.2,
        random_state: int | None = None,
        fitness_scores: np.ndarray | list | pd.Series | None = None,
        upper_tail: bool = True,
        threshold: float | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split by fitness score.

        Args:
            variants:        List of MutationSets.
            test_size:       Fraction of variants to place in test set.
                             Used when threshold=None (quantile mode).
                             Ignored when threshold is provided.
            random_state:    Not used. Accepted for API consistency.
            fitness_scores:  Array-like of fitness values, one per variant.
                             Must be provided.
            upper_tail:      If True (default), test = top test_size by fitness.
                             If False, test = bottom test_size by fitness.
            threshold:       If provided, split at this exact fitness value.
                             upper_tail=True  -> test is scores >= threshold.
                             upper_tail=False -> test is scores <= threshold.
            verbose:         If True, print actual split percentages and
                             threshold used.

        Returns:
            (train_indices, test_indices)
        """
        if fitness_scores is None:
            raise ValueError(
                "fitness_scores must be provided for FitnessSplitStrategy. "
                "Set fitness_col= in VarSplit, or pass fitness_scores= directly."
            )

        scores = np.asarray(fitness_scores, dtype=float)
        n = len(variants)

        if len(scores) != n:
            raise ValueError(
                f"fitness_scores length ({len(scores)}) must match "
                f"number of variants ({n})."
            )

        if threshold is not None:
            cutoff = threshold
            mode = "threshold"
        else:
            quantile = 1.0 - test_size if upper_tail else test_size
            cutoff = float(np.nanquantile(scores, quantile))
            mode = "quantile"

        if upper_tail:
            test_mask = scores >= cutoff
        else:
            test_mask = scores <= cutoff

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]

        if len(test_idx) == 0:
            raise ValueError(
                f"No variants found in test set with cutoff={cutoff:.4f} "
                f"(mode={mode}, upper_tail={upper_tail}). "
                f"Score range: [{scores.min():.4f}, {scores.max():.4f}]."
            )
        if len(train_idx) == 0:
            raise ValueError(
                f"No variants found in train set with cutoff={cutoff:.4f}. "
                f"Try adjusting test_size or threshold."
            )

        if verbose:
            tail = "top" if upper_tail else "bottom"
            print(
                f"Fitness split ({mode}, {tail} tail, cutoff={cutoff:.4f}): "
                f"train={len(train_idx)} ({len(train_idx)/n:.1%}), "
                f"test={len(test_idx)} ({len(test_idx)/n:.1%})"
            )
            print(
                f"  Train score range: [{scores[train_idx].min():.4f}, "
                f"{scores[train_idx].max():.4f}]"
            )
            print(
                f"  Test score range:  [{scores[test_idx].min():.4f}, "
                f"{scores[test_idx].max():.4f}]"
            )

        self._validate_split(variants, train_idx, test_idx)
        return train_idx, test_idx

    def kfold(
        self,
        variants: list[MutationSet],
        n_splits: int = 5,
        random_state: int | None = None,
        fitness_scores: np.ndarray | list | pd.Series | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        K-fold by fitness quantile bins.

        Divides the fitness range into n_splits equal-quantile bins.
        Each fold holds out one bin as test, trains on the rest.
        Ensures coverage across the full fitness distribution.
        """
        if fitness_scores is None:
            raise ValueError("fitness_scores must be provided for fitness kfold.")

        scores = np.asarray(fitness_scores, dtype=float)
        n = len(variants)
        quantile_edges = np.linspace(0, 1, n_splits + 1)
        thresholds = np.nanquantile(scores, quantile_edges)

        if verbose:
            print(f"Fitness kfold ({n_splits} folds by quantile bins):")

        for i in range(n_splits):
            lo, hi = thresholds[i], thresholds[i + 1]
            if i == n_splits - 1:
                test_mask = (scores >= lo) & (scores <= hi)
            else:
                test_mask = (scores >= lo) & (scores < hi)

            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]

            if len(test_idx) > 0 and len(train_idx) > 0:
                if verbose:
                    print(
                        f"  Fold {i+1} (scores [{lo:.4f}, {hi:.4f}]): "
                        f"train={len(train_idx)} ({len(train_idx)/n:.1%}), "
                        f"test={len(test_idx)} ({len(test_idx)/n:.1%})"
                    )
                yield train_idx, test_idx
