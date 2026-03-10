"""Tests for MutationalSplitStrategy."""

import numpy as np
import pytest
from varsplit.strategies.mutational import MutationalSplitStrategy
from varsplit.parsing.standard import StandardMutationParser

parser = StandardMutationParser()

def make_variants(strings):
    return [parser.parse(s) for s in strings]

strategy = MutationalSplitStrategy()

STRINGS = ['A1V','A1S','G2D','G2E','L3F','L3W','A1V:G2D','A1V:L3F','G2D:L3F','A1S:G2E']
VARIANTS = make_variants(STRINGS)


def test_no_overlap():
    train, test = strategy.split(VARIANTS, random_state=0, verbose=False)
    assert len(np.intersect1d(train, test)) == 0

def test_complete_coverage():
    train, test = strategy.split(VARIANTS, random_state=0, verbose=False)
    assert sorted(np.concatenate([train, test])) == list(range(len(VARIANTS)))

def test_held_out_substitution_not_in_train():
    """No training variant should contain a held-out (pos, mut_aa) substitution."""
    train, test = strategy.split(
        VARIANTS, held_out_substitutions={(1, 'V')}, verbose=False
    )
    for i in train:
        subs = {(pos, mut) for (pos, wt, mut) in VARIANTS[i]}
        assert (1, 'V') not in subs, (
            f"Variant {STRINGS[i]} contains held-out sub (1,V) but is in train."
        )

def test_multimutant_with_held_out_sub_goes_to_test():
    """A1V:G2D contains (1,V). If (1,V) held out, A1V:G2D should be in test."""
    variants = make_variants(['A1S', 'G2D', 'A1V', 'A1V:G2D'])
    train, test = strategy.split(
        variants, held_out_substitutions={(1, 'V')}, verbose=False
    )
    # index 2 = A1V, index 3 = A1V:G2D -- both contain (1,V)
    assert 2 in test
    assert 3 in test
    # index 0 = A1S (sub is (1,S)), index 1 = G2D -- neither held out
    assert 0 in train
    assert 1 in train

def test_same_position_different_aa_treated_separately():
    """A1V and A1S are different substitutions -- holding out (1,V) leaves A1S in train."""
    variants = make_variants(['A1V', 'A1S'])
    train, test = strategy.split(
        variants, held_out_substitutions={(1, 'V')}, verbose=False
    )
    assert 0 in test   # A1V -- held out
    assert 1 in train  # A1S -- different substitution, stays in train

def test_reproducibility():
    train1, test1 = strategy.split(VARIANTS, random_state=99, verbose=False)
    train2, test2 = strategy.split(VARIANTS, random_state=99, verbose=False)
    np.testing.assert_array_equal(sorted(train1), sorted(train2))

def test_all_singles_warns(capsys):
    singles = make_variants(['A1V','A1S','G2D','G2E'])
    strategy.split(singles, random_state=0, verbose=True)
    captured = capsys.readouterr()
    assert "Warning" in captured.out

def test_kfold_coverage():
    test_counts = np.zeros(len(VARIANTS), dtype=int)
    for train, test in strategy.kfold(VARIANTS, n_splits=3, verbose=False):
        test_counts[test] += 1
    assert np.all(test_counts == 1)
