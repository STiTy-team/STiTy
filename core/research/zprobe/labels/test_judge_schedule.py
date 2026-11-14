import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_schedule import build_prompt, verdict  # noqa: E402


def test_verdict_codes_only():
    assert verdict({"defects": ["grammar", "style", "omission"], "note": "x"}) == (["grammar", "omission"], "x")
    assert verdict({"defects": "grammar"}) is None and verdict({}) is None


def test_prompt_mentions_reference_only_when_present():
    row = {"lang": "ko", "tgt": "en", "src": "아피아는", "ref": "Apia is"}
    assert "Reference translation" in build_prompt(row, "Apia is.")
    assert "Reference translation" not in build_prompt({**row, "ref": ""}, "Apia is.")


def test_prompt_allows_implied_subject_rendered_passive():
    from judge_schedule import build_prompt
    p = build_prompt({"sid": "a", "lang": "ko", "tgt": "en", "src": "유전자를 설계해요"}, "Genes are designed.")
    assert "implied subject" in p


def test_prompt_excludes_punctuation_and_source_faithful_readings():
    p = build_prompt({"lang": "ko", "tgt": "en", "src": "그것은 성과입니다.", "ref": "It is an acquisition."}, "It is an achievement.")
    assert "punctuation" in p.lower()
    assert "faithful to the source" in p


def test_prompt_excludes_fillers_implied_determiners_and_name_internal_words():
    p = build_prompt({"lang": "ko", "tgt": "en", "src": "이제 좀 기쁩니다.", "ref": None}, "I'm happy for my neighbors.")
    for s in ("discourse filler", "possessive", "particle", "institution"):
        assert s in p, s
    assert "changes the meaning" in p


def test_combine_votes_passes_unless_every_judge_finds_a_defect():
    from judge_schedule import combine
    assert combine([(["grammar"], "a"), ([], "")]) == ([], "", True)
    assert combine([(["grammar"], "a"), (["omission"], "b")]) == (["grammar", "omission"], "a", False)
    assert combine([(["grammar", "order"], ""), (["order"], "x")]) == (["order"], "x", False)
    assert combine([None, (["grammar"], "a")]) == (["grammar"], "a", False)  # a malformed vote does not count
    assert combine([None, None]) is None
