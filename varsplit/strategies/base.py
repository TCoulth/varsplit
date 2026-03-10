"""
Abstract base class for all split strategies.

Every strategy receives variants as a list of MutationSets (already parsed)
and returns train/test indices. Strategies never touch raw strings or DataFrames
-- that's handled upstream in core.py.

To implement a new strategy:
    1. Subclass BaseSplitStrategy
    2. Implement split() -- the core logic
    3. Optionally override kfold() if your strategy needs special fold logic
    4. Register it in strategies/__init__.py

The default kfold() implementation calls split() repeatedly, which works
for most strategies. Override it if you need to ensure balanced representation
across folds (e.g. fitness-stratified folding).
"""

from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np
from varsplit.parsing.base import MutationSet


class BaseSplitStrategy(ABC):
    """
    Abstract base class for variant split strategies.

    All strategies operate on a list of MutationSets -- one per variant --
    and return numpy arrays of indices into that list.
    """

    @abstractmethod
    def split(
        self,
        variants: list[MutationSet],
        test_size: float = 0.2,
        random_state: int | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split variants into train and test index arrays.

        Args:
            variants:     List of MutationSets, one per variant (row).
            test_size:    Fraction of data to place in test set (0 < test_size < 1).
            random_state: Seed for reproducibility where randomness is involved.
            **kwargs:     Strategy-specific parameters.

        Returns:
            (train_indices, test_indices) as numpy arrays of integer indices.

        Contract:
            - Every index in range(len(variants)) must appear in exactly one
              of train_indices or test_indices.
            - No index may appear in both.
        """
        ...

    def kfold(
        self,
        variants: list[MutationSet],
        n_splits: int = 5,
        random_state: int | None = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate k train/test splits for cross-validation.

        Default implementation partitions all indices into n_splits folds
        and yields each fold as the test set with the rest as train.
        Strategies that need special fold logic (e.g. fitness-stratified)
        should override this method.

        Args:
            variants:     List of MutationSets, one per variant.
            n_splits:     Number of folds.
            random_state: Seed for reproducibility.
            **kwargs:     Passed through to split() or strategy-specific logic.

        Yields:
            (train_indices, test_indices) tuples, one per fold.
        """
        n = len(variants)
        indices = np.arange(n)

        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(indices)
        folds = np.array_split(shuffled, n_splits)

        for i in range(n_splits):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
            yield train_idx, test_idx

    def _validate_split(
        self,
        variants: list[MutationSet],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        """
        Assert that train and test indices are valid and non-overlapping.
        Called internally -- raises AssertionError if the split is malformed.
        This is a development-time check, not a user-facing feature.
        """
        n = len(variants)
        all_idx = np.concatenate([train_idx, test_idx])

        assert len(np.unique(all_idx)) == n, (
            f"Split does not cover all {n} variants. "
            f"Got {len(np.unique(all_idx))} unique indices."
        )
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0, (
            f"Train/test overlap detected at indices: {overlap}. "
            f"This is a bug in the {self.__class__.__name__} strategy."
        )
