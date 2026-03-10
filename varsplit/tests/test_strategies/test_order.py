"""Tests for OrderSplitStrategy."""

import numpy as np
import pytest
from varsplit.strategies.order import OrderSplitStrategy
from varsplit.parsing.standard import StandardMutationParser

parser = StandardMutationParser()

def make_variants(strings):
    return [parser.parse(s) for s in strings]

strategy = OrderSplitStrategy()

# 6 singles, 4 doubles, 2 triples
STRINGS = ['A1V','A1S','G2D','G2E','L3F','L3W','A1V:G2D','A1V:L3F','G2D:L3F','A1S:G2E','A1V:G2D:L3F','A1S:G2E:L3W']
VARIANTS = make_variants(STRINGS)


def test_no_overlap():
    train, test = strategy.split(VARIANTS, train_max_order=1, verbose=False)
    assert len(np.intersect1d(train, test)) == 0

def test_complete_coverage():
    train, test = strategy.split(VARIANTS, train_max_order=1, verbose=False)
    assert sorted(np.concatenate([train, test])) == list(range(len(VARIANTS)))

def test_default_singles_in_train_multis_in_test():
    """Default train_max_order=1: all singles in train, all multis in test."""
    train, test = strategy.split(VARIANTS, train_max_order=1, verbose=False)
    orders = [len(v) for v in VARIANTS]
    for i in train:
        assert orders[i] <= 1, f"Order-{orders[i]} variant found in train."
    for i in test:
        assert orders[i] > 1, f"Order-{orders[i]} variant found in test."

def test_train_max_order_2():
    """train_max_order=2: singles+doubles in train, triples in test."""
    train, test = strategy.split(VARIANTS, train_max_order=2, verbose=False)
    orders = [len(v) for v in VARIANTS]
    for i in train:
        assert orders[i] <= 2
    for i in test:
        assert orders[i] > 2

def test_wildtype_in_train():
    """Wildtype (order 0) should always be in train."""
    variants = make_variants(['A1V', 'G2D']) + [frozenset()]  # add wildtype
    train, test = strategy.split(variants, train_max_order=1, verbose=False)
    wt_idx = 2  # the wildtype
    assert wt_idx in train

def test_no_higher_order_raises():
    singles_only = make_variants(['A1V', 'A1S', 'G2D'])
    with pytest.raises(ValueError, match="No variants with order"):
        strategy.split(singles_only, train_max_order=1, verbose=False)

def test_reproducibility():
    """Order split is deterministic -- random_state has no effect."""
    train1, test1 = strategy.split(VARIANTS, train_max_order=1, verbose=False, random_state=1)
    train2, test2 = strategy.split(VARIANTS, train_max_order=1, verbose=False, random_state=99)
    np.testing.assert_array_equal(sorted(train1), sorted(train2))

def test_kfold_each_order_is_test_once():
    """Each unique order should appear as the test set exactly once."""
    orders = np.array([len(v) for v in VARIANTS])
    unique_orders = set(np.unique(orders)) - {0}  # exclude wildtype order
    test_orders_seen = set()
    for train, test in strategy.kfold(VARIANTS, verbose=False):
        test_orders = {len(VARIANTS[i]) for i in test}
        assert len(test_orders) == 1  # each fold tests exactly one order
        test_orders_seen |= test_orders
    assert test_orders_seen == unique_orders

def test_kfold_no_overlap():
    for train, test in strategy.kfold(VARIANTS, verbose=False):
        assert len(np.intersect1d(train, test)) == 0
