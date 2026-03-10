"""
Positional split strategy.

Holds out all variants that contain mutations at a set of positions.
Models the question: "Can my model generalize to new positions in the protein
that were not mutated in the training set?"

If a multi-site variant has any mutation at a held-out position, the entire
variant goes to test. This prevents any positional leakage into training.

Future position selection methods (beyond random) will be added via a
`method` parameter -- e.g. contiguous segments, solvent accessibility,
proximity to active site. These all share the same downstream split logic;
only the position selection differs.
"""

import numpy as np
from .base import BaseSplitStrategy
from varsplit.parsing.base import MutationSet


class PositionalSplitStrategy(BaseSplitStrategy):

    def split(
        self,
        variants: list[MutationSet],
        test_size: float = 0.2,
        random_state: int | None = None,
        held_out_positions: set[int] | None = None,
        method: str = "random",
        verbose: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split by holding out positions.

        Args:
            variants:             List of MutationSets.
            test_size:            Fraction of positions to hold out.
                                  Note: resulting variant fraction will differ
                                  depending on how often positions appear.
            random_state:         Random seed.
            held_out_positions:   Explicitly specify positions to hold out.
                                  If provided, test_size and method are ignored.
            method:               How to select held-out positions.
                                  Currently only "random" is supported.
                                  Future: "contiguous", "sasa", "proximity", etc.
            verbose:              If True, print actual split percentages.

        Returns:
            (train_indices, test_indices)
        """
        all_positions = sorted({
            pos
            for variant in variants
            for (pos, wt, mut) in variant
        })

        if len(all_positions) == 0:
            raise ValueError(
                "No positions found in variants. "
                "Ensure mutation strings were parsed correctly."
            )

        if held_out_positions is not None:
            test_positions = set(held_out_positions)
        elif method == "random":
            n_test_pos = max(1, int(np.floor(len(all_positions) * test_size)))
            rng = np.random.default_rng(random_state)
            test_positions = set(
                rng.choice(all_positions, size=n_test_pos, replace=False).tolist()
            )
        else:
            raise ValueError(
                f"Unknown method='{method}'. Currently supported: 'random'. "
                f"User-specified positions: use held_out_positions=."
            )

        train_idx, test_idx = [], []
        for i, variant in enumerate(variants):
            variant_positions = {pos for (pos, wt, mut) in variant}
            if variant_positions & test_positions:
                test_idx.append(i)
            else:
                train_idx.append(i)

        train_idx = np.array(train_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)
        n = len(variants)

        if verbose:
            print(
                f"Positional split (method={method}, "
                f"{len(test_positions)}/{len(all_positions)} positions held out): "
                f"train={len(train_idx)} ({len(train_idx)/n:.1%}), "
                f"test={len(test_idx)} ({len(test_idx)/n:.1%})"
            )

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
        K-fold by positions -- each fold holds out a disjoint set of positions.
        Every position appears in exactly one test fold.
        """
        all_positions = sorted({
            pos
            for variant in variants
            for (pos, wt, mut) in variant
        })

        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(all_positions).tolist()
        position_folds = np.array_split(shuffled, n_splits)
        n = len(variants)

        if verbose:
            print(f"Positional kfold ({n_splits} folds, "
                  f"{len(all_positions)} total positions):")

        for i, fold_positions in enumerate(position_folds):
            held_out = set(fold_positions.tolist())
            train_idx, test_idx = self.split(
                variants,
                held_out_positions=held_out,
                verbose=False,
            )
            if verbose:
                print(
                    f"  Fold {i+1} ({len(held_out)} positions held out): "
                    f"train={len(train_idx)} ({len(train_idx)/n:.1%}), "
                    f"test={len(test_idx)} ({len(test_idx)/n:.1%})"
                )
            yield train_idx, test_idx
