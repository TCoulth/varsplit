"""Tests for FitnessSplitStrategy."""

import numpy as np
import pytest
from varsplit.strategies.fitness import FitnessSplitStrategy
from varsplit.parsing.standard import StandardMutationParser

parser = StandardMutationParser()

def make_variants(strings):
    return [parser.parse(s) for s in strings]

strategy = FitnessSplitStrategy()

STRINGS = ['A1V','A1S','G2D','G2E','L3F','L3W','A1V:G2D','A1V:L3F','G2D:L3F','A1S:G2E']
VARIANTS = make_variants(STRINGS)
SCORES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def test_no_overlap():
    train, test = strategy.split(VARIANTS, fitness_scores=SCORES, verbose=False)
    assert len(np.intersect1d(train, test)) == 0

def test_complete_coverage():
    train, test = strategy.split(VARIANTS, fitness_scores=SCORES, verbose=False)
    assert sorted(np.concatenate([train, test])) == list(range(len(VARIANTS)))

def test_upper_tail_test_has_highest_scores():
    """Test set should contain the highest-scoring variants."""
    train, test = strategy.split(
        VARIANTS, fitness_scores=SCORES, test_size=0.2, upper_tail=True, verbose=False
    )
    min_test_score = SCORES[test].min()
    max_train_score = SCORES[train].max()
    assert min_test_score >= max_train_score

def test_lower_tail_test_has_lowest_scores():
    """With upper_tail=False, test should contain lowest-scoring variants."""
    train, test = strategy.split(
        VARIANTS, fitness_scores=SCORES, test_size=0.2,
        upper_tail=False, verbose=False
    )
    max_test_score = SCORES[test].max()
    min_train_score = SCORES[train].min()
    assert max_test_score <= min_train_score

def test_threshold_mode():
    """All test variants should have score >= threshold."""
    train, test = strategy.split(
        VARIANTS, fitness_scores=SCORES, threshold=0.7, verbose=False
    )
    assert np.all(SCORES[test] >= 0.7)
    assert np.all(SCORES[train] < 0.7)

def test_threshold_lower_tail():
    train, test = strategy.split(
        VARIANTS, fitness_scores=SCORES, threshold=0.3,
        upper_tail=False, verbose=False
    )
    assert np.all(SCORES[test] <= 0.3)
    assert np.all(SCORES[train] > 0.3)

def test_no_fitness_scores_raises():
    with pytest.raises(ValueError, match="fitness_scores must be provided"):
        strategy.split(VARIANTS, verbose=False)

def test_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        strategy.split(VARIANTS, fitness_scores=np.array([1.0, 2.0]), verbose=False)

def test_threshold_all_in_test_raises():
    with pytest.raises(ValueError, match="No variants found in train"):
        strategy.split(
            VARIANTS, fitness_scores=SCORES, threshold=-999.0,
            upper_tail=False, verbose=False
        )

def test_kfold_no_overlap():
    for train, test in strategy.kfold(
        VARIANTS, n_splits=3, fitness_scores=SCORES, verbose=False
    ):
        assert len(np.intersect1d(train, test)) == 0

def test_kfold_coverage():
    """Every variant should appear in exactly one test fold."""
    test_counts = np.zeros(len(VARIANTS), dtype=int)
    for train, test in strategy.kfold(
        VARIANTS, n_splits=3, fitness_scores=SCORES, verbose=False
    ):
        test_counts[test] += 1
    assert np.all(test_counts == 1)

def test_kfold_ascending_scores():
    """Later folds should have higher scores than earlier folds (quantile bins)."""
    fold_medians = []
    for train, test in strategy.kfold(
        VARIANTS, n_splits=3, fitness_scores=SCORES, verbose=False
    ):
        fold_medians.append(np.median(SCORES[test]))
    assert fold_medians == sorted(fold_medians)
