"""Tests for PositionalSplitStrategy."""

import numpy as np
import pytest
from varsplit.strategies.positional import PositionalSplitStrategy
from varsplit.parsing.standard import StandardMutationParser

parser = StandardMutationParser()

def make_variants(strings):
    return [parser.parse(s) for s in strings]

strategy = PositionalSplitStrategy()

# Dataset with known structure: positions 1, 2, 3
STRINGS = ['A1V','A1S','G2D','G2E','L3F','L3W','A1V:G2D','A1V:L3F','G2D:L3F','A1S:G2E']
VARIANTS = make_variants(STRINGS)


def test_no_overlap():
    train, test = strategy.split(VARIANTS, random_state=0, verbose=False)
    assert len(np.intersect1d(train, test)) == 0

def test_complete_coverage():
    train, test = strategy.split(VARIANTS, random_state=0, verbose=False)
    assert sorted(np.concatenate([train, test])) == list(range(len(VARIANTS)))

def test_held_out_position_not_in_train():
    """No training variant should contain a mutation at a held-out position."""
    train, test = strategy.split(
        VARIANTS, held_out_positions={1}, verbose=False
    )
    for i in train:
        positions = {pos for (pos, wt, mut) in VARIANTS[i]}
        assert 1 not in positions, (
            f"Variant {STRINGS[i]} with position 1 found in train "
            f"but position 1 is held out."
        )

def test_held_out_position_in_test():
    """Every variant touching a held-out position should be in test."""
    train, test = strategy.split(
        VARIANTS, held_out_positions={1}, verbose=False
    )
    test_set = set(test)
    for i, variant in enumerate(VARIANTS):
        positions = {pos for (pos, wt, mut) in variant}
        if 1 in positions:
            assert i in test_set, (
                f"Variant {STRINGS[i]} touches held-out position 1 "
                f"but is not in test set."
            )

def test_multimutant_with_held_out_goes_to_test():
    """A1V:G2D has positions 1 and 2. If position 2 held out, it should be in test."""
    variants = make_variants(['A1V', 'G2D', 'A1V:G2D'])
    train, test = strategy.split(variants, held_out_positions={2}, verbose=False)
    # index 2 is A1V:G2D -- should be in test
    assert 2 in test
    # index 0 is A1V (position 1 only) -- should be in train
    assert 0 in train

def test_user_specified_positions():
    train, test = strategy.split(
        VARIANTS, held_out_positions={2, 3}, verbose=False
    )
    for i in train:
        positions = {pos for (pos, wt, mut) in VARIANTS[i]}
        assert not positions & {2, 3}

def test_reproducibility():
    train1, test1 = strategy.split(VARIANTS, random_state=7, verbose=False)
    train2, test2 = strategy.split(VARIANTS, random_state=7, verbose=False)
    np.testing.assert_array_equal(sorted(train1), sorted(train2))

def test_no_positions_raises():
    wildtype_variants = [frozenset(), frozenset()]
    with pytest.raises(ValueError, match="No positions found"):
        strategy.split(wildtype_variants, verbose=False)

def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        strategy.split(VARIANTS, method="sasa", verbose=False)

def test_kfold_disjoint_positions():
    """Each fold's test positions should be disjoint from other folds' test positions."""
    all_positions = {pos for v in VARIANTS for (pos, wt, mut) in v}
    seen_test_positions = set()
    for train, test in strategy.kfold(VARIANTS, n_splits=3, verbose=False):
        test_positions = {
            pos for i in test for (pos, wt, mut) in VARIANTS[i]
        }
        # Test positions in this fold should not have appeared in a previous fold's test
        assert not test_positions & seen_test_positions
        seen_test_positions |= test_positions

def test_kfold_coverage():
    """Every variant should appear in exactly one test fold."""
    test_counts = np.zeros(len(VARIANTS), dtype=int)
    for train, test in strategy.kfold(VARIANTS, n_splits=3, verbose=False):
        test_counts[test] += 1
    assert np.all(test_counts == 1)
