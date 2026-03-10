"""
varsplit.diagnose -- retrospective comparison of two variant sets.

Usage:
    from varsplit.diagnose import Diagnose

    diag = Diagnose(df_A, df_B, mutation_col="mutations", fitness_col="score")

    vs   = diag.recommend()                    # prints summary, returns VarSplit
    diag.report("report.html")                 # full HTML report
    diag.report("report.html", preset="deviations")  # only differing dimensions
"""

import numpy as np
import pandas as pd

from varsplit.parsing import parse_mutations
from varsplit.parsing.base import MutationSet

from .metrics import (
    dataset_summary,
    positional_metrics,
    mutational_metrics,
    order_metrics,
    fitness_metrics,
)
from .recommender import compute_recommendation, RecommendationResult
from .plots import (
    plot_order_distribution,
    plot_positional_overlap,
    plot_mutational_novelty,
    plot_fitness_distributions,
)
from .report import build_html_report, save_report


class Diagnose:
    """
    Compare two sets of protein variants and recommend a split strategy.

    Args:
        df_A:          DataFrame for set A (train candidates).
        df_B:          DataFrame for set B (test / holdout set).
        mutation_col:  Column name containing mutation strings.
        fitness_col:   Column name containing fitness scores. Optional.
        fmt:           Mutation string format: "standard", "hgvs", "mavedb", "infer".
        label_A:       Display label for set A. Default "train".
        label_B:       Display label for set B. Default "test".
    """

    def __init__(
        self,
        df_A: pd.DataFrame,
        df_B: pd.DataFrame,
        mutation_col: str = "mutations",
        fitness_col: str | None = None,
        fmt: str = "standard",
        label_A: str = "train",
        label_B: str = "test",
    ):
        self.label_A = label_A
        self.label_B = label_B
        self.mutation_col = mutation_col
        self.fitness_col = fitness_col
        self.fmt = fmt

        # Parse mutation strings
        self._variants_A: list[MutationSet] = parse_mutations(df_A[mutation_col], fmt=fmt)
        self._variants_B: list[MutationSet] = parse_mutations(df_B[mutation_col], fmt=fmt)

        # Fitness scores (optional)
        self._scores_A: np.ndarray | None = None
        self._scores_B: np.ndarray | None = None
        if fitness_col is not None:
            if fitness_col not in df_A.columns or fitness_col not in df_B.columns:
                raise ValueError(
                    f"fitness_col='{fitness_col}' not found in both DataFrames."
                )
            self._scores_A = df_A[fitness_col].to_numpy(dtype=float)
            self._scores_B = df_B[fitness_col].to_numpy(dtype=float)

        # Compute all metrics eagerly -- they're cheap and used everywhere
        self._summary    = dataset_summary(self._variants_A, self._variants_B)
        self._positional = positional_metrics(self._variants_A, self._variants_B)
        self._mutational = mutational_metrics(self._variants_A, self._variants_B)
        self._order      = order_metrics(self._variants_A, self._variants_B)
        self._fitness    = (
            fitness_metrics(self._scores_A, self._scores_B)
            if self._scores_A is not None else None
        )

        # Compute recommendation once
        self._recommendation: RecommendationResult = compute_recommendation(
            positional=self._positional,
            mutational=self._mutational,
            order=self._order,
            fitness=self._fitness,
            mutation_col=mutation_col,
            fmt=fmt,
            fitness_col=fitness_col,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def recommend(self) -> "VarSplit":
        """
        Print a formatted recommendation summary and return a configured
        VarSplit object ready to use for within-A splitting.

        Returns:
            VarSplit configured with the recommended strategy.
        """
        self._recommendation.print_summary()
        return self._recommendation.varsplit

    def summary(self) -> None:
        """Print a plain-text summary of dataset sizes and key metrics."""
        s  = self._summary
        p  = self._positional
        m  = self._mutational
        o  = self._order
        print()
        print(f"  {self.label_A}: {s['n_variants_A']} variants  |  "
              f"{self.label_B}: {s['n_variants_B']} variants")
        print(f"  Positional novelty:   {p['pct_B_novel']:.0%} of {self.label_B} positions not in {self.label_A}")
        print(f"  Mutational novelty:   {m['pct_B_novel']:.0%} of {self.label_B} substitutions not in {self.label_A}")
        print(f"  Order shift:          {self.label_A} mean={o['mean_A']:.2f}, "
              f"{self.label_B} mean={o['mean_B']:.2f} (shift={o['mean_shift']:+.2f})")
        if self._fitness:
            f = self._fitness
            print(f"  Fitness extrapolation: {f['pct_B_extrapolates']:.0%} of {self.label_B} outside {self.label_A} range")
        print()

    def report(
        self,
        output: str,
        preset: str = "full",
    ) -> None:
        """
        Generate and save an HTML (or PDF) report.

        Args:
            output: File path. Use .html or .pdf extension.
            preset: "full" — all sections.
                    "deviations" — only dimensions with meaningful novelty.
        """
        if preset not in ("full", "deviations"):
            raise ValueError(f"preset must be 'full' or 'deviations', got '{preset}'.")

        figures = self._build_figures()
        html = build_html_report(
            summary=self._summary,
            positional=self._positional,
            mutational=self._mutational,
            order=self._order,
            fitness=self._fitness,
            recommendation=self._recommendation,
            figures=figures,
            label_A=self.label_A,
            label_B=self.label_B,
            preset=preset,
        )
        save_report(html, output)
        print(f"Report saved to: {output}")

    # ------------------------------------------------------------------
    # Individual plots (can be called directly)
    # ------------------------------------------------------------------

    def plot_order_distribution(self):
        return plot_order_distribution(self._order, self.label_A, self.label_B)

    def plot_positional_overlap(self):
        return plot_positional_overlap(self._positional, self.label_A, self.label_B)

    def plot_mutational_novelty(self):
        return plot_mutational_novelty(self._mutational, self.label_A, self.label_B)

    def plot_fitness_distributions(self):
        if self._fitness is None:
            raise ValueError("No fitness scores provided. Pass fitness_col= to Diagnose.")
        return plot_fitness_distributions(
            self._fitness, self._scores_A, self._scores_B, self.label_A, self.label_B
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_figures(self) -> dict:
        figs = {
            "order":      plot_order_distribution(self._order, self.label_A, self.label_B),
            "positional": plot_positional_overlap(self._positional, self.label_A, self.label_B),
            "mutational": plot_mutational_novelty(self._mutational, self.label_A, self.label_B),
        }
        if self._fitness is not None:
            figs["fitness"] = plot_fitness_distributions(
                self._fitness, self._scores_A, self._scores_B, self.label_A, self.label_B
            )
        return figs
