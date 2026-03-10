"""
Mutational split strategy.

Holds out specific (position, mutant_aa) substitutions.
The same positions may appear in train and test, but the specific
amino acid substitution at that position is unseen in training.

Models the question: "Can my model generalize to new amino acid substitutions
at positions it has already seen mutated?"

For all-singles datasets, this behaves similarly to random splitting --
the meaningful separation only emerges when multi-site variants are present,
where a variant is held out if any of its component substitutions are held out.

Note: wildtype AA is intentionally excluded from the holdout key. What is
held out is the substitution *to* a particular AA at a position, regardless
of the wildtype. For datasets with multiple wildtype backgrounds, this
behavior can be revisited.
"""

import numpy as np
from .base import BaseSplitStrategy
from varsplit.parsing.base import MutationSet


class MutationalSplitStrategy(BaseSplitStrategy):

    def split(
        self,
        variants: list[MutationSet],
        test_size: float = 0.2,
        random_state: int | None = None,
        held_out_substitutions: set[tuple[int, str]] | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split by holding out (position, mutant_aa) substitutions.

        Args:
            variants:               List of MutationSets.
            test_size:              Fraction of unique substitutions to hold out.
            random_state:           Random seed.
            held_out_substitutions: Explicitly specify (pos, mut_aa) pairs to
                                    hold out. If provided, test_size is ignored.
            verbose:                If True, print actual split percentages.

        Returns:
            (train_indices, test_indices)
        """
        all_substitutions = sorted({
            (pos, mut)
            for variant in variants
            for (pos, wt, mut) in variant
        })

        if len(all_substitutions) == 0:
            raise ValueError(
                "No substitutions found in variants. "
                "Ensure mutation strings were parsed correctly."
            )

        n_unique = len(all_substitutions)
        n_positions = len({pos for (pos, mut) in all_substitutions})
        n_singles = sum(1 for v in variants if len(v) == 1)
        if n_singles == len(variants) and verbose:
            print(
                "Warning: dataset contains only single-site variants. "
                "Mutational split will behave similarly to random split. "
                "Consider strategy='random' as a simpler equivalent."
            )

        if held_out_substitutions is not None:
            test_substitutions = set(held_out_substitutions)
        else:
            n_test = max(1, int(np.floor(n_unique * test_size)))
            rng = np.random.default_rng(random_state)
            chosen = rng.choice(n_unique, size=n_test, replace=False)
            test_substitutions = {all_substitutions[i] for i in chosen}

        train_idx, test_idx = [], []
        for i, variant in enumerate(variants):
            variant_subs = {(pos, mut) for (pos, wt, mut) in variant}
            if variant_subs & test_substitutions:
                test_idx.append(i)
            else:
                train_idx.append(i)

        train_idx = np.array(train_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)
        n = len(variants)

        if verbose:
            print(
                f"Mutational split "
                f"({len(test_substitutions)}/{n_unique} substitutions held out "
                f"across {n_positions} positions): "
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
        K-fold by substitutions -- each fold holds out a disjoint set of
        (position, mutant_aa) pairs. Every substitution appears in exactly
        one test fold.
        """
        all_substitutions = sorted({
            (pos, mut)
            for variant in variants
            for (pos, wt, mut) in variant
        })

        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(all_substitutions))
        fold_indices = np.array_split(indices, n_splits)
        n = len(variants)

        if verbose:
            print(f"Mutational kfold ({n_splits} folds, "
                  f"{len(all_substitutions)} total substitutions):")

        for i, fold_idx in enumerate(fold_indices):
            held_out = {all_substitutions[j] for j in fold_idx}
            train_idx, test_idx = self.split(
                variants,
                held_out_substitutions=held_out,
                verbose=False,
            )
            if verbose:
                print(
                    f"  Fold {i+1} ({len(held_out)} substitutions held out): "
                    f"train={len(train_idx)} ({len(train_idx)/n:.1%}), "
                    f"test={len(test_idx)} ({len(test_idx)/n:.1%})"
                )
            yield train_idx, test_idx
