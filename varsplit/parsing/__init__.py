"""
Mutation string parsers for varsplit.

Public API:
    get_parser(fmt)  ->  BaseMutationParser
    parse_mutations(series, fmt)  ->  list[MutationSet]
"""

import pandas as pd
from .base import BaseMutationParser, MutationSet, MutationParseError
from .standard import StandardMutationParser
from .hgvs import HGVSMutationParser
from .mavedb import MaveDBMutationParser
from .infer import InferMutationParser

_PARSER_REGISTRY: dict[str, type[BaseMutationParser]] = {
    "standard": StandardMutationParser,
    "hgvs": HGVSMutationParser,
    "mavedb": MaveDBMutationParser,
    "infer": InferMutationParser,
}


def get_parser(fmt: str) -> BaseMutationParser:
    """Return a parser instance for the given format string."""
    if fmt not in _PARSER_REGISTRY:
        raise ValueError(
            f"Unknown mutation format '{fmt}'. "
            f"Available: {list(_PARSER_REGISTRY)}"
        )
    return _PARSER_REGISTRY[fmt]()


def parse_mutations(
    mutation_series: pd.Series,
    fmt: str = "standard",
) -> list[MutationSet]:
    """
    Parse a pandas Series of mutation strings into a list of MutationSets.

    Args:
        mutation_series: Series of mutation strings.
        fmt: Format name — "standard", "hgvs", "mavedb", or "infer".

    Returns:
        List of frozensets, one per row. Wildtype rows -> frozenset().
    """
    parser = get_parser(fmt)

    if fmt == "infer":
        parser.fit(mutation_series.tolist())

    return [parser.parse(s) for s in mutation_series]


__all__ = [
    "BaseMutationParser",
    "MutationSet",
    "MutationParseError",
    "StandardMutationParser",
    "HGVSMutationParser",
    "MaveDBMutationParser",
    "InferMutationParser",
    "get_parser",
    "parse_mutations",
]
