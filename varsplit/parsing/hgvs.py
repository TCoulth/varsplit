"""
Parser for HGVS protein notation.

Format: p.{WT_AA_3letter}{position}{MUT_AA_3letter}
Examples:
    "p.Ala23Val"   ->  (23, 'A', 'V')
    "p.Gly105Ser"  ->  (105, 'G', 'S')
    "p.Ala23="     ->  (23, 'A', 'A')   synonymous (= means no change)
    "p.Ter23Val"   ->  (23, '*', 'V')   stop codon as wildtype

Also handles shorthand "Ala23Val" without the "p." prefix.
"""

import re
from .base import BaseMutationParser, Mutation, MutationParseError

# Three-letter to one-letter amino acid code mapping
_AA3_TO_1: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U", "Pyl": "O", "Xaa": "X",
}

_AA3_PATTERN = "|".join(_AA3_TO_1.keys())
_HGVS_RE = re.compile(
    rf"^(?:p\.)?({_AA3_PATTERN})(\d+)({_AA3_PATTERN}|=)$"
)


class HGVSMutationParser(BaseMutationParser):
    """
    Parser for HGVS protein notation: p.Ala23Val.
    """

    def parse_single(self, token: str) -> Mutation:
        token = token.strip()
        match = _HGVS_RE.match(token)
        if not match:
            raise MutationParseError(
                f"Cannot parse '{token}' as HGVS format (expected e.g. 'p.Ala23Val')."
            )
        wt_3, pos, mut_3 = match.groups()
        wt = _AA3_TO_1[wt_3]
        # "=" means synonymous — mutant is same as wildtype
        mut = wt if mut_3 == "=" else _AA3_TO_1[mut_3]

        return (int(pos), wt, mut)

    def can_parse(self, mutation_str: str) -> bool:
        if not mutation_str or str(mutation_str).strip() in ("", "nan", "WT", "wt"):
            return True
        tokens = self._split_multimutant(str(mutation_str).strip())
        return all(_HGVS_RE.match(t.strip()) for t in tokens if t.strip())
