import pytest
from varsplit.parsing.standard import StandardMutationParser
from varsplit.parsing.base import MutationParseError

parser = StandardMutationParser()

def test_single_mutation():
    assert parser.parse("A23V") == frozenset({(23, 'A', 'V')})

def test_multi_mutation_colon():
    assert parser.parse("A23V:G105S") == frozenset({(23, 'A', 'V'), (105, 'G', 'S')})

def test_multi_mutation_comma():
    assert parser.parse("A23V,G105S") == frozenset({(23, 'A', 'V'), (105, 'G', 'S')})

def test_synonymous():
    assert parser.parse("A23A") == frozenset({(23, 'A', 'A')})

def test_wildtype_empty():
    assert parser.parse("") == frozenset()
    assert parser.parse(None) == frozenset()
    assert parser.parse("WT") == frozenset()

def test_stop_codon():
    assert parser.parse("A23*") == frozenset({(23, 'A', '*')})

def test_invalid_raises():
    with pytest.raises(MutationParseError):
        parser.parse("p.Ala23Val")

def test_can_parse():
    assert parser.can_parse("A23V") is True
    assert parser.can_parse("p.Ala23Val") is False
    assert parser.can_parse("A23V:G105S") is True
