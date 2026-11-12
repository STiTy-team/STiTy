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


def test_spent_so_far(tmp_path):
    from label_schedule import spent_so_far
    p = tmp_path / "u.jsonl"
    assert spent_so_far(str(p)) == 0.0
    p.write_text('{"est_usd": 1.25}\n{"est_usd": 0.5}\n{"est_us', encoding="utf-8")
    assert abs(spent_so_far(str(p)) - 1.75) < 1e-9


def test_with_retry():
    import httpx
    import openai
    import pytest
    from label_schedule import retryable, with_retry
    req = httpx.Request("POST", "http://x")
    e429 = openai.RateLimitError("rl", response=httpx.Response(429, request=req), body=None)
    e400 = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    e503 = openai.InternalServerError("down", response=httpx.Response(503, request=req), body=None)
    assert retryable(e429) and retryable(e503) and retryable(openai.APIConnectionError(request=req)) and not retryable(e400)
    seq, slept = [e429, e503, "ok"], []

    def fn():
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    assert with_retry(fn, sleep=slept.append) == "ok" and slept == [2, 4]
    slept.clear()
    with pytest.raises(openai.RateLimitError):
        with_retry(lambda: (_ for _ in ()).throw(e429), sleep=slept.append)
    assert slept == [2, 4, 8]
    with pytest.raises(openai.BadRequestError):
        with_retry(lambda: (_ for _ in ()).throw(e400), sleep=slept.append)


def test_too_many_errors():
    from label_schedule import too_many_errors
    assert not too_many_errors(1, 100) and too_many_errors(2, 100) and not too_many_errors(0, 0)


def test_src_words_recorded_and_prompt_asks_for_src():
    good = {"schedule": [{"after": 2, "text": "관리자가", "src": [1, 2]}, {"after": 5, "text": "계획을 승인했다", "src": [3, 4, 5]}]}
    rec = label_one(ROW, HY, fake([good]))
    assert rec["src_words"] == [[1, 2], [3, 4, 5]] and rec["hy_disagree"] == 0
    assert '"src"' in build_prompt(ROW)


def test_rules_and_cap_flags_reach_the_prompt(tmp_path):
    from label_schedule import DEFERRAL_CAP, RULES, n_cap
    rules = tmp_path / "r.txt"
    rules.write_text("RULE-MARKER {N}", encoding="utf-8")
    old = dict(RULES), dict(DEFERRAL_CAP)
    try:
        RULES["path"] = str(rules)
        DEFERRAL_CAP["en"] = DEFERRAL_CAP["ko"] = 0
        assert "RULE-MARKER 1000000" in build_prompt(ROW, "") and n_cap("en") == 10**6
    finally:
        RULES.update(old[0])
        DEFERRAL_CAP.update(old[1])


def test_note_goes_into_first_request():
    ask = fake([GOOD])
    label_one(ROW, HY, ask, note="A reviewer found: missing subject. Fix that.")
    assert ask.calls[0].endswith("A reviewer found: missing subject. Fix that.")


def test_reviewer_notes_from_judge_file(tmp_path):
    from label_schedule import reviewer_notes
    p = tmp_path / "j.jsonl"
    p.write_text('{"sid": "a", "pass": false, "note": "missing subject"}\n{"sid": "b", "pass": true, "note": ""}\n', encoding="utf-8")
    assert reviewer_notes(str(p)) == {"a": "A reviewer found: missing subject. Fix that."}
