"""
Infer parser — automatically detects mutation string format.

Tries each registered parser in priority order and selects the one
that can successfully parse the highest fraction of input strings.
"""

from .base import BaseMutationParser, Mutation, MutationSet, MutationParseError
from .standard import StandardMutationParser
from .hgvs import HGVSMutationParser
from .mavedb import MaveDBMutationParser

# Priority order for format detection (most common DMS format first)
_PARSERS: list[BaseMutationParser] = [
    StandardMutationParser(),
    MaveDBMutationParser(),
    HGVSMutationParser(),
]


def detect_format(mutation_strings: list[str]) -> BaseMutationParser:
    """
    Detect the mutation format from a list of mutation strings.

    Tries each parser and returns the one that can parse the most strings.
    Raises MutationParseError if no parser achieves >80% success rate.

    Args:
        mutation_strings: List of raw mutation strings to probe.

    Returns:
        The best-matching BaseMutationParser instance.
    """
    # Filter out nulls/wildcards for detection
    candidates = [
        s for s in mutation_strings
        if s and str(s).strip() not in ("", "nan", "WT", "wt")
    ]
    if not candidates:
        # All wildtype — any parser will do
        return StandardMutationParser()

    best_parser = None
    best_score = -1.0

    for parser in _PARSERS:
        successes = sum(1 for s in candidates if parser.can_parse(s))
        score = successes / len(candidates)
        if score > best_score:
            best_score = score
            best_parser = parser

    if best_score < 0.8:
        raise MutationParseError(
            f"Could not confidently detect mutation format. "
            f"Best match was {best_parser.__class__.__name__} "
            f"with {best_score:.0%} success rate. "
            f"Please specify fmt= explicitly."
        )

    return best_parser


class InferMutationParser(BaseMutationParser):
    """
    Parser that auto-detects format from the data.

    On first call to parse(), fits itself to the input and delegates
    to the detected parser for all subsequent calls.
    """

    def __init__(self):
        self._detected_parser: BaseMutationParser | None = None

    def fit(self, mutation_strings: list[str]) -> "InferMutationParser":
        """Detect format from a list of strings. Call before parse()."""
        self._detected_parser = detect_format(mutation_strings)
        return self

    @property
    def detected_format(self) -> str | None:
        if self._detected_parser is None:
            return None
        return self._detected_parser.__class__.__name__

    def parse_single(self, token: str) -> Mutation:
        if self._detected_parser is None:
            raise MutationParseError(
                "InferMutationParser must be fit() before parsing. "
                "Call fit(mutation_strings) first, or use fmt='infer' via VarSplit."
            )
        return self._detected_parser.parse_single(token)

    def can_parse(self, mutation_str: str) -> bool:
        if self._detected_parser is None:
            return False
        return self._detected_parser.can_parse(mutation_str)
