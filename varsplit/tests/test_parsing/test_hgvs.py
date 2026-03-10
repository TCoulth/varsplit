import pytest
from varsplit.parsing.hgvs import HGVSMutationParser
from varsplit.parsing.base import MutationParseError

parser = HGVSMutationParser()

def test_standard_hgvs():
    assert parser.parse("p.Ala23Val") == frozenset({(23, 'A', 'V')})

def test_without_prefix():
    assert parser.parse("Ala23Val") == frozenset({(23, 'A', 'V')})

def test_synonymous():
    assert parser.parse("p.Ala23=") == frozenset({(23, 'A', 'A')})

def test_stop_codon_wt():
    assert parser.parse("p.Ter23Val") == frozenset({(23, '*', 'V')})

def test_wildtype():
    assert parser.parse("") == frozenset()
    assert parser.parse(None) == frozenset()

def test_invalid_raises():
    with pytest.raises(MutationParseError):
        parser.parse("A23V")  # standard format, not HGVS

def test_can_parse():
    assert parser.can_parse("p.Ala23Val") is True
    assert parser.can_parse("A23V") is False
