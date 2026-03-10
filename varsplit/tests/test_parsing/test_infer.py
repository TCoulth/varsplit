from varsplit.parsing.infer import InferMutationParser, detect_format
from varsplit.parsing.standard import StandardMutationParser
from varsplit.parsing.hgvs import HGVSMutationParser

def test_detect_standard():
    strings = ["A23V", "G105S", "A23V:G105S"]
    parser = detect_format(strings)
    assert isinstance(parser, StandardMutationParser)

def test_detect_hgvs():
    strings = ["p.Ala23Val", "p.Gly105Ser"]
    parser = detect_format(strings)
    assert isinstance(parser, HGVSMutationParser)

def test_infer_parser_fit_and_parse():
    parser = InferMutationParser()
    parser.fit(["A23V", "G105S"])
    assert parser.parse("A23V") == frozenset({(23, 'A', 'V')})

def test_infer_parser_all_wildtype():
    parser = InferMutationParser()
    parser.fit(["", "WT", None])
    assert parser.parse("") == frozenset()
