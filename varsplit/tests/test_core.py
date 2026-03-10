"""Tests for the public API: train_test_split, KFold, VarSplit."""

import numpy as np
import pandas as pd
import pytest
from varsplit import train_test_split, KFold, VarSplit

# Shared test DataFrame
DF = pd.DataFrame({
    'mutations': ['A1V','A1S','G2D','G2E','L3F','L3W',
                  'A1V:G2D','A1V:L3F','G2D:L3F','A1S:G2E',
                  'A1V:G2D:L3F','A1S:G2E:L3W'],
    'score':     [0.1, 0.2, 0.3, 0.4, 0.5, 0.55,
                  0.6, 0.7, 0.8, 0.85, 0.9, 1.0],
})


# --- train_test_split ---

def test_functional_returns_indices():
    train, test = train_test_split(DF, strategy='random', random_state=0)
    assert isinstance(train, np.ndarray)
    assert isinstance(test, np.ndarray)

def test_functional_no_overlap():
    train, test = train_test_split(DF, strategy='random', random_state=0)
    assert len(np.intersect1d(train, test)) == 0

def test_functional_complete_coverage():
    train, test = train_test_split(DF, strategy='random', random_state=0)
    assert sorted(np.concatenate([train, test])) == list(range(len(DF)))

def test_functional_all_strategies():
    """All active strategies should run without error."""
    for strategy in ['random', 'positional', 'mutational', 'order']:
        train, test = train_test_split(DF, strategy=strategy, random_state=0, verbose=False)
        assert len(train) + len(test) == len(DF)

def test_functional_fitness_strategy():
    train, test = train_test_split(
        DF, strategy='fitness', fitness_col='score', verbose=False
    )
    assert len(train) + len(test) == len(DF)

def test_functional_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        train_test_split(DF, strategy='nonexistent')

def test_functional_wrong_mutation_col_raises():
    with pytest.raises(ValueError, match="not found in DataFrame"):
        train_test_split(DF, mutation_col='wrong_col')

def test_functional_indices_usable_with_iloc():
    train, test = train_test_split(DF, strategy='random', random_state=0)
    train_df = DF.iloc[train]
    test_df = DF.iloc[test]
    assert len(train_df) + len(test_df) == len(DF)

def test_functional_fmt_infer():
    train, test = train_test_split(DF, strategy='random', fmt='infer', random_state=0)
    assert len(train) + len(test) == len(DF)

def test_functional_custom_mutation_col():
    df2 = DF.rename(columns={'mutations': 'mutant'})
    train, test = train_test_split(df2, mutation_col='mutant', strategy='random', random_state=0)
    assert len(train) + len(test) == len(df2)


# --- KFold ---

def test_kfold_yields_tuples():
    kf = KFold(n_splits=3, strategy='random')
    folds = list(kf.split(DF))
    assert len(folds) == 3
    for train, test in folds:
        assert isinstance(train, np.ndarray)
        assert isinstance(test, np.ndarray)

def test_kfold_no_overlap_any_fold():
    kf = KFold(n_splits=3, strategy='random')
    for train, test in kf.split(DF):
        assert len(np.intersect1d(train, test)) == 0

def test_kfold_random_coverage():
    """Every variant should appear in test exactly once across all folds."""
    kf = KFold(n_splits=3, strategy='random')
    test_counts = np.zeros(len(DF), dtype=int)
    for train, test in kf.split(DF):
        test_counts[test] += 1
    assert np.all(test_counts == 1)

def test_kfold_fitness_requires_fitness_col():
    kf = KFold(n_splits=3, strategy='fitness', fitness_col='score')
    folds = list(kf.split(DF))
    assert len(folds) == 3

def test_kfold_order_overrides_n_splits():
    """Order kfold uses number of unique orders, not n_splits."""
    kf = KFold(n_splits=10, strategy='order')
    folds = list(kf.split(DF, verbose=False))
    # Dataset has orders 1, 2, 3 -> 3 folds regardless of n_splits=10
    assert len(folds) == 3


# --- VarSplit ---

def test_varsplit_configured_instance():
    vs = VarSplit(mutation_col='mutations', fitness_col='score')
    train, test = vs.train_test_split(DF, strategy='fitness', verbose=False)
    assert len(train) + len(test) == len(DF)

def test_varsplit_kfold():
    vs = VarSplit(mutation_col='mutations')
    folds = list(vs.kfold(DF, n_splits=3, strategy='random'))
    assert len(folds) == 3

def test_varsplit_wrong_fitness_col_raises():
    vs = VarSplit(mutation_col='mutations', fitness_col='wrong_score')
    with pytest.raises(ValueError, match="wrong_score"):
        vs.train_test_split(DF, strategy='fitness')

def test_varsplit_fitness_strategy_without_fitness_col_raises():
    vs = VarSplit(mutation_col='mutations')  # no fitness_col set
    with pytest.raises(ValueError, match="fitness_scores must be provided"):
        vs.train_test_split(DF, strategy='fitness')
