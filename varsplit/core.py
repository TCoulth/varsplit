"""
Core public API for varsplit.

Provides:
    VarSplit         -- configurable splitter class (stateful)
    train_test_split -- functional shortcut (mirrors sklearn)
    KFold            -- cross-validation class (mirrors sklearn)
"""

import numpy as np
import pandas as pd
from typing import Iterator

from varsplit.parsing import parse_mutations, get_parser
from varsplit.parsing.base import MutationSet
from varsplit.strategies import get_strategy, REGISTRY


class VarSplit:
    """
    Configurable protein variant splitter.

    Set up once with your dataset's column names and mutation format,
    then call split() or kfold() as needed.

    Args:
        mutation_col:  Column name containing mutation strings. Default "mutations".
        fmt:           Mutation string format: "standard", "hgvs", "mavedb", "infer".
        fitness_col:   Column name containing fitness/score values.
                       Required for strategy="fitness".
        reference:     Reference (wildtype) sequence string, or "consensus" to
                       infer it from the data. Used for sequence-based parsing
                       (not yet required for string-based parsing).

    Example:
        vs = VarSplit(mutation_col="mutant", fmt="standard", fitness_col="score")
        train_idx, test_idx = vs.train_test_split(df, test_size=0.2, strategy="positional")
        for train_idx, test_idx in vs.kfold(df, n_splits=5, strategy="combinatorial"):
            ...
    """

    def __init__(
        self,
        mutation_col: str = "mutations",
        fmt: str = "standard",
        fitness_col: str | None = None,
        reference: str | None = None,
    ):
        self.mutation_col = mutation_col
        self.fmt = fmt
        self.fitness_col = fitness_col
        self.reference = reference

    def __repr__(self) -> str:
        parts = [f"mutation_col='{self.mutation_col}'", f"fmt='{self.fmt}'"]
        if self.fitness_col is not None:
            parts.append(f"fitness_col='{self.fitness_col}'")
        if self.reference is not None:
            ref_display = (
                f"'{self.reference[:12]}...'"
                if isinstance(self.reference, str) and len(self.reference) > 12
                else f"'{self.reference}'"
            )
            parts.append(f"reference={ref_display}")
        return f"VarSplit({', '.join(parts)})"

    def _parse(self, df: pd.DataFrame) -> list[MutationSet]:
        """Parse mutation strings from DataFrame into MutationSets."""
        if self.mutation_col not in df.columns:
            raise ValueError(
                f"Column '{self.mutation_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        return parse_mutations(df[self.mutation_col], fmt=self.fmt)

    def _get_fitness(self, df: pd.DataFrame) -> np.ndarray | None:
        """Extract fitness scores from DataFrame if fitness_col is set."""
        if self.fitness_col is None:
            return None
        if self.fitness_col not in df.columns:
            raise ValueError(
                f"fitness_col='{self.fitness_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        return df[self.fitness_col].to_numpy(dtype=float)

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        strategy: str = "random",
        random_state: int | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split a DataFrame into train and test index arrays.

        Args:
            df:           DataFrame with mutation strings (and optionally fitness scores).
            test_size:    Fraction of data for the test set.
            strategy:     Split strategy name. One of: {strategies}
            random_state: Random seed for reproducibility.
            **kwargs:     Additional strategy-specific parameters
                          (e.g. held_out_positions, train_max_order).

        Returns:
            (train_indices, test_indices) as numpy arrays of integer indices
            into df. Use df.iloc[train_indices] to get the train DataFrame.
        """
        variants = self._parse(df)
        fitness = self._get_fitness(df)
        splitter = get_strategy(strategy)

        if fitness is not None:
            kwargs.setdefault("fitness_scores", fitness)

        return splitter.split(
            variants,
            test_size=test_size,
            random_state=random_state,
            **kwargs,
        )

    def kfold(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        strategy: str = "random",
        random_state: int | None = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate k-fold cross-validation splits.

        Args:
            df:           DataFrame with mutation strings.
            n_splits:     Number of folds.
            strategy:     Split strategy name.
            random_state: Random seed.
            **kwargs:     Strategy-specific parameters.

        Yields:
            (train_indices, test_indices) tuples, one per fold.
        """
        variants = self._parse(df)
        fitness = self._get_fitness(df)
        splitter = get_strategy(strategy)

        if fitness is not None:
            kwargs.setdefault("fitness_scores", fitness)

        yield from splitter.kfold(
            variants,
            n_splits=n_splits,
            random_state=random_state,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Functional API -- mirrors sklearn for drop-in familiarity
# ---------------------------------------------------------------------------

def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    strategy: str = "random",
    random_state: int | None = None,
    mutation_col: str = "mutations",
    fmt: str = "standard",
    fitness_col: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a DataFrame of protein variants into train and test index arrays.

    Drop-in replacement for sklearn's train_test_split, with protein-aware
    splitting strategies.

    Args:
        df:           DataFrame containing variant data.
        test_size:    Fraction of data for the test set. Default 0.2.
        strategy:     How to split. Options: "random", "positional",
                      "mutational", "combinatorial", "order", "fitness".
        random_state: Random seed for reproducibility.
        mutation_col: Column name containing mutation strings. Default "mutations".
        fmt:          Mutation string format: "standard", "hgvs", "mavedb", "infer".
        fitness_col:  Column with fitness values (required for strategy="fitness").
        **kwargs:     Strategy-specific parameters.

    Returns:
        (train_indices, test_indices) as numpy arrays.

    Example:
        train_idx, test_idx = train_test_split(df, test_size=0.2, strategy="positional")
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]
    """
    vs = VarSplit(mutation_col=mutation_col, fmt=fmt, fitness_col=fitness_col)
    return vs.train_test_split(df, test_size=test_size, strategy=strategy,
                               random_state=random_state, **kwargs)


class KFold:
    """
    Protein-aware k-fold cross-validation splitter.

    Mirrors sklearn's KFold interface.

    Args:
        n_splits:     Number of folds. Default 5.
        strategy:     Split strategy name. Default "random".
        random_state: Random seed.
        mutation_col: Column name for mutation strings.
        fmt:          Mutation string format.
        fitness_col:  Column name for fitness scores.

    Example:
        kf = KFold(n_splits=5, strategy="positional")
        for train_idx, test_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            test_df  = df.iloc[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        strategy: str = "random",
        random_state: int | None = None,
        mutation_col: str = "mutations",
        fmt: str = "standard",
        fitness_col: str | None = None,
    ):
        self.n_splits = n_splits
        self.strategy = strategy
        self.random_state = random_state
        self._vs = VarSplit(mutation_col=mutation_col, fmt=fmt, fitness_col=fitness_col)

    def split(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate k train/test splits.

        Args:
            df:       DataFrame containing variant data.
            **kwargs: Strategy-specific parameters.

        Yields:
            (train_indices, test_indices) tuples.
        """
        yield from self._vs.kfold(
            df,
            n_splits=self.n_splits,
            strategy=self.strategy,
            random_state=self.random_state,
            **kwargs,
        )

