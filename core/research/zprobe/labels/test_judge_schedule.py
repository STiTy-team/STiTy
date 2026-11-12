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
