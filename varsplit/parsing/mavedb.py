"""
Parser for MaveDB mutation notation.

MaveDB uses HGVS-like notation but with some conventions:
    - Synonymous variants: "p.="  or  "p.Ala23="
    - Nonsense: "p.Ala23*"  (stop codon as mutant)
    - Multiple: "p.[Ala23Val;Gly105Ser]"  (semicolon-separated in brackets)

References:
    https://www.mavedb.org/docs/
"""

import re
from .base import BaseMutationParser, Mutation, MutationParseError
from .hgvs import HGVSMutationParser, _AA3_TO_1, _AA3_PATTERN

# MaveDB wraps multi-mutants in square brackets: p.[Ala23Val;Gly105Ser]
_MAVEDB_MULTI_RE = re.compile(r"^p\.\[(.+)\]$")

# Single variant (extends HGVS to allow * as mutant directly)
_MAVEDB_SINGLE_RE = re.compile(
    rf"^(?:p\.)?({_AA3_PATTERN})(\d+)({_AA3_PATTERN}|=|\*)$"
)

_hgvs_parser = HGVSMutationParser()


class MaveDBMutationParser(BaseMutationParser):
    """
    Parser for MaveDB notation: p.[Ala23Val;Gly105Ser].
    Extends HGVS parser to handle MaveDB multi-mutant bracket notation.
    """

    def parse(self, mutation_str: str | None):
        """Override to handle bracket notation before delegating."""
        if not mutation_str or str(mutation_str).strip() in ("", "nan", "WT", "wt", "p.="):
            return frozenset()

        mutation_str = str(mutation_str).strip()

        # Unwrap bracket notation: p.[A;B] -> ["A", "B"]
        bracket_match = _MAVEDB_MULTI_RE.match(mutation_str)
        if bracket_match:
            inner = bracket_match.group(1)
            tokens = inner.split(";")
        else:
            tokens = self._split_multimutant(mutation_str)

        mutations = set()
        for token in tokens:
            token = token.strip()
            if token and token != "=":
                mutations.add(self.parse_single(token))

        return frozenset(mutations)

    def parse_single(self, token: str) -> Mutation:
        token = token.strip()
        # Delegate to HGVS for standard tokens
        if _hgvs_parser.can_parse(token):
            return _hgvs_parser.parse_single(token)
        raise MutationParseError(
            f"Cannot parse '{token}' as MaveDB format."
        )

    def can_parse(self, mutation_str: str) -> bool:
        if not mutation_str or str(mutation_str).strip() in ("", "nan", "WT", "wt", "p.="):
            return True
        mutation_str = str(mutation_str).strip()
        bracket_match = _MAVEDB_MULTI_RE.match(mutation_str)
        if bracket_match:
            tokens = bracket_match.group(1).split(";")
            return all(_hgvs_parser.can_parse(t.strip()) for t in tokens if t.strip())
        return _hgvs_parser.can_parse(mutation_str)
