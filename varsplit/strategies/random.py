"""
Random split strategy -- the baseline.

Splits variants randomly with no awareness of mutation content.
Equivalent to sklearn's train_test_split with shuffle=True.
Useful as a baseline to compare against protein-aware strategies.
"""

import numpy as np
from .base import BaseSplitStrategy
from varsplit.parsing.base import MutationSet


class RandomSplitStrategy(BaseSplitStrategy):
    """
    Random train/test split. No protein-awareness.

    This is the baseline strategy -- it should be the easiest for any model
    to perform well on, since train and test are drawn from the same
    distribution. Use it to sanity-check models before applying
    harder, more realistic splits.
    """

    def split(
        self,
        variants: list[MutationSet],
        test_size: float = 0.2,
        random_state: int | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly assign variants to train and test sets.

        Args:
            variants:     List of MutationSets (content ignored -- split is random).
            test_size:    Fraction of variants to place in test set.
            random_state: Random seed for reproducibility.

        Returns:
            (train_indices, test_indices)
        """
        n = len(variants)
        n_test = max(1, int(np.floor(n * test_size)))
        n_train = n - n_test

        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(n)

        train_idx = shuffled[:n_train]
        test_idx = shuffled[n_train:]

        self._validate_split(variants, train_idx, test_idx)
        return train_idx, test_idx
