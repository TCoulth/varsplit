"""
Parser for standard DMS mutation notation.

Format: {WT_AA}{position}{MUT_AA}
Examples:
    Single:     "A23V"
    Multi:      "A23V:G105S"  (colon-separated, but other delimiters also handled)
    Synonymous: "A23A"        (wildtype == mutant, valid)

Position is 1-indexed by convention in most DMS datasets.
"""

import re
from .base import BaseMutationParser, Mutation, MutationParseError

# Matches: one letter (wt), digits (position), one letter (mut)
_STANDARD_RE = re.compile(r"^([A-Za-z*])(\d+)([A-Za-z*])$")

# Valid amino acid single-letter codes including stop codon (*)
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy*")


class StandardMutationParser(BaseMutationParser):
    """
    Parser for standard DMS format: {WT}{POS}{MUT}  e.g. A23V, G105S.
    """

    def parse_single(self, token: str) -> Mutation:
        token = token.strip()
        match = _STANDARD_RE.match(token)
        if not match:
            raise MutationParseError(
                f"Cannot parse '{token}' as standard mutation format (expected e.g. 'A23V')."
            )
        wt, pos, mut = match.groups()
        wt = wt.upper()
        mut = mut.upper()

        if wt not in _VALID_AA:
            raise MutationParseError(f"Invalid wildtype amino acid '{wt}' in '{token}'.")
        if mut not in _VALID_AA:
            raise MutationParseError(f"Invalid mutant amino acid '{mut}' in '{token}'.")

        return (int(pos), wt, mut)

    def can_parse(self, mutation_str: str) -> bool:
        """Return True if all tokens in the string match standard format."""
        if not mutation_str or str(mutation_str).strip() in ("", "nan", "WT", "wt"):
            return True  # wildtype is valid
        tokens = self._split_multimutant(str(mutation_str).strip())
        return all(_STANDARD_RE.match(t.strip()) for t in tokens if t.strip())
