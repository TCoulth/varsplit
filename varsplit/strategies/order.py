"""
Mutation order split strategy.

Splits by the number of mutations per variant (mutation order).
Train on lower-order variants, test on higher-order variants.

Models the question: "Can my model trained on singles (and/or doubles)
extrapolate to predict higher-order combinatorial mutants?"

Two modes:
    - Default: all singles -> train, all multimutants -> test
    - Custom:  train_max_order=N, anything above N goes to test

Wildtype variants (order 0) always go to train.

The split percentage is reported but not enforced -- given discrete
order boundaries, exact test_size fractions are rarely achievable.
"""

import numpy as np
from .base import BaseSplitStrategy
from varsplit.parsing.base import MutationSet


class OrderSplitStrategy(BaseSplitStrategy):
    """
    Split by mutation order (number of mutations per variant).

    All variants with order <= train_max_order -> train.
    All variants with order >  train_max_order -> test.
    Wildtype (order 0) always -> train.
    """

    def split(
        self,
        variants: list[MutationSet],
        test_size: float = 0.2,
        random_state: int | None = None,
        train_max_order: int = 1,
        verbose: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split by mutation order.

        Args:
            variants:         List of MutationSets.
            test_size:        Not used -- order splits are deterministic.
                              Accepted for API consistency.
            random_state:     Not used. Accepted for API consistency.
            train_max_order:  Maximum mutation order in train set. Default 1
                              (train on singles only, test on doubles and higher).
                              Set to 2 to train on singles+doubles, etc.
            verbose:          If True, print actual split percentages.

        Returns:
            (train_indices, test_indices)
        """
        orders = np.array([len(v) for v in variants])
        n = len(variants)

        train_idx = np.where(orders <= train_max_order)[0]
        test_idx = np.where(orders > train_max_order)[0]

        if len(test_idx) == 0:
            raise ValueError(
                f"No variants with order > {train_max_order} found. "
                f"Max order in dataset is {orders.max()}. "
                f"Lower train_max_order or provide higher-order variants."
            )

        if verbose:
            print(
                f"Order split (train_max_order={train_max_order}): "
                f"train={len(train_idx)} ({len(train_idx)/n:.1%}), "
                f"test={len(test_idx)} ({len(test_idx)/n:.1%})"
            )
            order_counts = {int(o): int(np.sum(orders == o)) for o in np.unique(orders)}
            print(f"  Order distribution: {order_counts}")

        self._validate_split(variants, train_idx, test_idx)
        return train_idx, test_idx

    def kfold(
        self,
        variants: list[MutationSet],
        n_splits: int = 5,
        random_state: int | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        K-fold by order: each fold holds out one specific mutation order as test,
        trains on all other orders.

        Note: number of folds equals number of unique orders in the dataset,
        not n_splits. n_splits is accepted for API consistency but overridden.
        """
        orders = np.array([len(v) for v in variants])
        unique_orders = np.sort(np.unique(orders))

        if verbose and len(unique_orders) != n_splits:
            print(
                f"Order kfold: using {len(unique_orders)} folds "
                f"(one per unique order: {unique_orders.tolist()}), "
                f"not n_splits={n_splits}."
            )

        for test_order in unique_orders:
            if test_order == 0:
                continue  # wildtype always in train
            test_idx = np.where(orders == test_order)[0]
            train_idx = np.where(orders != test_order)[0]
            if len(test_idx) > 0 and len(train_idx) > 0:
                if verbose:
                    n = len(variants)
                    print(
                        f"  Fold order={test_order}: "
                        f"train={len(train_idx)} ({len(train_idx)/n:.1%}), "
                        f"test={len(test_idx)} ({len(test_idx)/n:.1%})"
                    )
                yield train_idx, test_idx
