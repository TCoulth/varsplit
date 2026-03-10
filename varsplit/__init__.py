"""
varsplit -- protein variant-aware train/test splitting.

Quick start:
    from varsplit import train_test_split, KFold, VarSplit

    # Drop-in sklearn style
    train_idx, test_idx = train_test_split(df, strategy="positional")

    # Cross-validation
    for train_idx, test_idx in KFold(n_splits=5, strategy="combinatorial").split(df):
        ...

    # Configured instance
    vs = VarSplit(mutation_col="mutant", fmt="standard", fitness_col="score")
    train_idx, test_idx = vs.train_test_split(df, strategy="fitness")

Available strategies: random, positional, mutational, combinatorial, order, fitness
"""

from varsplit.core import VarSplit, train_test_split, KFold
from varsplit.strategies import REGISTRY as STRATEGIES

__version__ = "0.1.0"
__all__ = ["VarSplit", "train_test_split", "KFold", "STRATEGIES"]
