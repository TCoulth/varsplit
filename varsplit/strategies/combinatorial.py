"""
Combinatorial split strategy -- RESERVED FOR FUTURE DEVELOPMENT.

This strategy is intentionally not implemented in the current version.

Rationale:
    The use case for combinatorial splitting overlaps significantly with
    both OrderSplitStrategy (train on singles, test on multimutants) and
    MutationalSplitStrategy (hold out specific substitutions). A distinct
    combinatorial strategy is only justified if there is a clear scientific
    question that neither of those strategies can answer.

    A candidate use case -- "given complete single-mutant coverage, predict
    epistatic interactions in doubles" -- is better served by OrderSplitStrategy
    with train_max_order=1, possibly combined with a dataset filter ensuring
    single-mutant completeness.

    If you have a use case that genuinely requires combinatorial splitting,
    please open an issue describing the scientific question being asked.
    That will guide the design here.

To implement when a clear use case is established:
    1. Subclass BaseSplitStrategy
    2. Implement split() and kfold()
    3. Register in strategies/__init__.py
"""

from .base import BaseSplitStrategy


class CombinatorialSplitStrategy(BaseSplitStrategy):
    """Not yet implemented. See module docstring for rationale."""

    def split(self, variants, test_size=0.2, random_state=None, **kwargs):
        raise NotImplementedError(
            "CombinatorialSplitStrategy is reserved for future development. "
            "For train-on-singles/test-on-multis behavior, use strategy='order'. "
            "For holding out specific substitutions, use strategy='mutational'."
        )

    def kfold(self, variants, n_splits=5, random_state=None, **kwargs):
        raise NotImplementedError(
            "CombinatorialSplitStrategy is reserved for future development."
        )
