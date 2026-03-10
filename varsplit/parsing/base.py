"""
Base class for mutation string parsers.

All parsers convert mutation strings into a canonical internal representation:
    frozenset[tuple[int, str, str]]  ->  {(position, wildtype_aa, mutant_aa), ...}

Examples:
    "A23V"        ->  frozenset({(23, 'A', 'V')})
    "A23V:G105S"  ->  frozenset({(23, 'A', 'V'), (105, 'G', 'S')})
    ""  or  None  ->  frozenset()   # wildtype
"""

from abc import ABC, abstractmethod

# Canonical mutation type: (position, wildtype_aa, mutant_aa)
Mutation = tuple[int, str, str]
MutationSet = frozenset[Mutation]


class BaseMutationParser(ABC):
    """
    Abstract base class for mutation string parsers.

    Subclasses implement `parse_single` to handle one mutation token
    (e.g. "A23V"). The base class handles multi-mutant strings by
    splitting on common delimiters and aggregating.
    """

    # Delimiters that separate individual mutations in a multi-mutant string
    DELIMITERS: list[str] = [":", ",", "/", "+", ";", " ","-"]

    def parse(self, mutation_str: str | None) -> MutationSet:
        """
        Parse a mutation string (single or multi-mutant) into a frozenset
        of (position, wt_aa, mut_aa) tuples.

        Args:
            mutation_str: Mutation string in the format this parser handles.
                          None or empty string returns frozenset() (wildtype).

        Returns:
            frozenset of (position, wildtype_aa, mutant_aa) tuples.

        Raises:
            MutationParseError: If the string cannot be parsed.
        """
        if not mutation_str or str(mutation_str).strip() in ("", "nan", "WT", "wt"):
            return frozenset()

        mutation_str = str(mutation_str).strip()
        tokens = self._split_multimutant(mutation_str)

        mutations = set()
        for token in tokens:
            token = token.strip()
            if token:
                mutations.add(self.parse_single(token))

        return frozenset(mutations)

    @abstractmethod
    def parse_single(self, token: str) -> Mutation:
        """
        Parse a single mutation token into a (position, wt_aa, mut_aa) tuple.

        Args:
            token: A single mutation string (e.g. "A23V", "p.Ala23Val").

        Returns:
            (position, wildtype_aa, mutant_aa) tuple.

        Raises:
            MutationParseError: If the token cannot be parsed.
        """
        ...

    @abstractmethod
    def can_parse(self, mutation_str: str) -> bool:
        """
        Return True if this parser can handle the given mutation string.
        Used by InferParser to select the right parser automatically.
        """
        ...

    def _split_multimutant(self, mutation_str: str) -> list[str]:
        """Split a multi-mutant string into individual mutation tokens."""
        for delim in self.DELIMITERS:
            if delim in mutation_str:
                return mutation_str.split(delim)
        return [mutation_str]


class MutationParseError(ValueError):
    """Raised when a mutation string cannot be parsed."""
    pass
