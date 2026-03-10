"""Tests for RandomSplitStrategy."""

import numpy as np
import pytest
from varsplit.strategies.random import RandomSplitStrategy
from varsplit.parsing.standard import StandardMutationParser

parser = StandardMutationParser()

def make_variants(strings):
    return [parser.parse(s) for s in strings]

VARIANTS = make_variants(['A1V','A1S','G2D','G2E','L3F','L3W','A1V:G2D','A1V:L3F','G2D:L3F','A1S:G2E'])
strategy = RandomSplitStrategy()


def test_no_overlap():
    train, test = strategy.split(VARIANTS, test_size=0.2, random_state=0)
    assert len(np.intersect1d(train, test)) == 0

def test_complete_coverage():
    train, test = strategy.split(VARIANTS, test_size=0.2, random_state=0)
    assert sorted(np.concatenate([train, test])) == list(range(len(VARIANTS)))

def test_approximate_test_size():
    train, test = strategy.split(VARIANTS, test_size=0.2, random_state=0)
    assert len(test) >= 1
    assert len(test) / len(VARIANTS) <= 0.35  # some tolerance

def test_reproducibility():
    train1, test1 = strategy.split(VARIANTS, random_state=42)
    train2, test2 = strategy.split(VARIANTS, random_state=42)
    np.testing.assert_array_equal(sorted(train1), sorted(train2))
    np.testing.assert_array_equal(sorted(test1), sorted(test2))

def test_different_seeds_differ():
    _, test1 = strategy.split(VARIANTS, random_state=1)
    _, test2 = strategy.split(VARIANTS, random_state=2)
    assert not np.array_equal(sorted(test1), sorted(test2))

def test_single_variant():
    v = make_variants(['A1V'])
    train, test = strategy.split(v, test_size=0.2)
    assert len(train) + len(test) == 1
