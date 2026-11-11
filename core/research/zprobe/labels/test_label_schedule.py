import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_schedule import build_prompt, label_one, should_stop  # noqa: E402

ROW = {"sid": "s1", "lang": "en", "tgt": "ko", "src": "The manager approved the plan", "ref": "관리자가 계획을 승인했다", "source": "x", "recs": []}
HY = {"sid": "s1", "prefixes": ["관리자", "관리자", "관리자 승인했다", "관리자 승인했다", "관리자가 계획을 승인했다"]}
GOOD = {"schedule": [{"after": 2, "text": "관리자가"}, {"after": 5, "text": "계획을 승인했다"}]}
BAD_RANGE = {"schedule": [{"after": 99, "text": "관리자가"}]}


def fake(replies):
    calls = []

    def ask(row, note):
        calls.append(note)
        return replies[len(calls) - 1]
    ask.calls = calls
    return ask


def test_retry_then_ok():
    ask = fake([BAD_RANGE, GOOD])
    rec = label_one(ROW, HY, ask)
    assert rec["attempts"] == 2 and rec["schedule"] == [[2, "관리자가"], [5, "계획을 승인했다"]]
    assert ask.calls[0] == "" and ask.calls[1].startswith("Previous attempt was rejected: range")


def test_both_fail():
    reasons = []
    assert label_one(ROW, HY, fake([BAD_RANGE, BAD_RANGE]), fail=reasons.append) is None
    assert reasons[0].startswith("range")


def test_format_error_is_gate_failure():
    assert label_one(ROW, HY, fake([{}, {"x": 1}])) is None


def test_no_hy_no_call():
    ask = fake([GOOD])
    assert label_one(ROW, None, ask) is None
    assert label_one(ROW, {"sid": "s1", "prefixes": ["a"]}, ask) is None
    assert ask.calls == []


def test_should_stop():
    assert should_stop(20.01, 20.0) and not should_stop(20.0, 20.0)


def test_prompt():
    p = build_prompt(ROW, "NOTE")
    assert "1:The 2:manager" in p and "Reference translation (Korean)" in p and "more than 8 source words" in p
    assert "NOTE" in p and "Reference" not in build_prompt({**ROW, "ref": None})
    assert "more than 6" in build_prompt({**ROW, "lang": "ko", "tgt": "en", "src": "a b"})
