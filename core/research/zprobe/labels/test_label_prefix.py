import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_prefix import build_regret_prompt, build_step_prompt, parse_add, parse_regret  # noqa: E402

ROW = {"sid": "s1", "lang": "en", "tgt": "ko", "src": "the manager approved the budget yesterday", "ref": "매니저가 어제 예산을 승인했습니다"}


def test_step_prompt_shows_only_prefix_and_committed():
    p = build_step_prompt(ROW, 2, "", False)
    assert "1:the 2:manager" in p and "approved" not in p and "승인했습니다" not in p
    assert "More source may follow" in p
    p2 = build_step_prompt(ROW, 6, "매니저가", True)
    assert "매니저가" in p2 and "ENDS here" in p2 and "6:yesterday" in p2


def test_parse_add():
    assert parse_add({"add": "매니저가 어제", "src": [2, 6]}) == ("매니저가 어제", {2, 6})
    assert parse_add({"add": "  ", "src": []}) == ("", set())
    with pytest.raises(ValueError):
        parse_add({"text": "x"})


def test_regret_prompt_and_parse():
    p = build_regret_prompt(ROW, [(2, "매니저가"), (6, "어제 예산을 승인했습니다.")])
    assert "[1] 매니저가" in p and "[2] 어제" in p and "승인했습니다" in p and ROW["ref"] in p
    assert parse_regret({"regret": [1, 9], "note": "n"}, 2) == ([1], "n")
    with pytest.raises(ValueError):
        parse_regret({"x": 1}, 2)


# --- label_sentence: the per-sentence loop ----------------------------------------------------------------------
from label_prefix import label_sentence  # noqa: E402

ROW_KO = {"sid": "k1", "lang": "ko", "tgt": "en", "src": "어제 회의에서 홍 지사는 새 예산을 1990년에 승인했다"}  # points [2,4,7,8]
FINAL = {"add": "Governor Hong approved the new budget in 1990.", "src": [3, 4, 5, 6, 7, 8]}
EMPTY = {"add": "", "src": []}


def fake_steps(answers):
    calls = []

    def ask(row, k, committed, is_end, note):
        calls.append((k, committed, is_end, note))
        return answers.pop(0)
    ask.calls = calls
    return ask


def no_regret(row, pieces):
    return {"regret": [], "note": ""}


def test_happy_path_builds_schedule_with_waits():
    ask = fake_steps([{"add": "At yesterday's meeting,", "src": [1, 2]}, EMPTY, EMPTY, FINAL])
    rec = label_sentence(ROW_KO, ask, no_regret)
    assert rec["schedule"] == [[2, "At yesterday's meeting,"], [4, ""], [7, ""], [8, FINAL["add"]]]
    assert rec["src_words"] == [[1, 2], [], [], [3, 4, 5, 6, 7, 8]]
    assert rec["calls"] == 4 and rec["regret_rounds"] == 0 and rec["attempts"] == 0
    assert ask.calls[3][2] is True and ask.calls[0][2] is False
    assert ask.calls[1][1] == "At yesterday's meeting,"  # committed is passed on


def test_single_point_sentence():
    row = {"sid": "e1", "lang": "en", "tgt": "ko", "src": "hello there"}
    rec = label_sentence(row, fake_steps([{"add": "안녕하세요.", "src": [1, 2]}]), no_regret)
    assert rec["schedule"] == [[2, "안녕하세요."]] and rec["calls"] == 1


def test_final_point_empty_reasked():
    row = {"sid": "e1", "lang": "en", "tgt": "ko", "src": "hello there"}
    ask = fake_steps([EMPTY, {"add": "안녕하세요.", "src": [1, 2]}])
    rec = label_sentence(row, ask, no_regret)
    assert rec["schedule"] == [[2, "안녕하세요."]] and "has ended" in ask.calls[1][3]


def test_final_point_empty_twice_fails():
    row = {"sid": "e1", "lang": "en", "tgt": "ko", "src": "hello there"}
    reasons = []
    assert label_sentence(row, fake_steps([EMPTY, EMPTY]), no_regret, fail=reasons.append) is None
    assert reasons == ["final point empty"]


def test_lookahead_src_becomes_wait():
    ask = fake_steps([{"add": "Governor Hong", "src": [3, 4]},  # k=2 but src 3,4 -> lookahead -> wait, no re-ask
                      EMPTY, EMPTY,
                      {"add": "At yesterday's meeting, Governor Hong approved the new budget in 1990.", "src": list(range(1, 9))}])
    rec = label_sentence(ROW_KO, ask, no_regret)
    assert rec["schedule"][0] == [2, ""] and rec["calls"] == 4 and rec["attempts"] == 0


def test_open_violation_reasks_once_then_waits():
    ask = fake_steps([EMPTY,                                                 # k=2
                      {"add": "The governor.", "src": [3, 4]},                # k=4: mid-source piece closes -> re-ask
                      {"add": "The governor.", "src": [3, 4]},                # closes again -> wait
                      EMPTY,                                                 # k=7
                      {"add": "At yesterday's meeting, Governor Hong approved the new budget in 1990.", "src": list(range(1, 9))}])
    rec = label_sentence(ROW_KO, ask, no_regret)
    assert rec["schedule"][1] == [4, ""] and rec["attempts"] == 1 and rec["calls"] == 5
    assert ask.calls[2][3].startswith("Previous attempt was rejected: open")


def test_regret_turns_piece_into_wait_and_reextracts():
    steps = fake_steps([{"add": "At yesterday's meeting,", "src": [1, 2]}, {"add": "The governor said", "src": [3, 4]},
                        EMPTY, {"add": "approved the new budget in 1990.", "src": [5, 6, 7, 8]},
                        # regret on piece 2 -> k=4 becomes a wait, k=7 and k=8 are asked again
                        EMPTY, FINAL])
    regrets = iter([{"regret": [2], "note": "said is not in the source"}, {"regret": [], "note": ""}])
    rec = label_sentence(ROW_KO, steps, lambda row, pieces: next(regrets))
    assert rec["schedule"] == [[2, "At yesterday's meeting,"], [4, ""], [7, ""], [8, FINAL["add"]]]
    assert rec["regret_rounds"] == 1 and rec["calls"] == 6
    assert steps.calls[4][1] == "At yesterday's meeting,"  # committed rolled back to before the regretted piece


def test_regret_on_last_piece_reasks_last():
    steps = fake_steps([{"add": "At yesterday's meeting,", "src": [1, 2]}, EMPTY, EMPTY,
                        {"add": "Governor Hong rejected the budget.", "src": [3, 4, 5, 6, 7, 8]},
                        FINAL])
    regrets = iter([{"regret": [2], "note": "rejected contradicts the source"}, {"regret": [], "note": ""}])
    rec = label_sentence(ROW_KO, steps, lambda row, pieces: next(regrets))
    assert rec["schedule"][-1] == [8, FINAL["add"]] and "contradicts" in steps.calls[4][3]
    assert rec["regret_rounds"] == 1


def test_regret_unresolved_fails():
    steps = fake_steps([{"add": "At yesterday's meeting,", "src": [1, 2]}, EMPTY, EMPTY,
                        {"add": "Governor Hong rejected it.", "src": [3, 4, 5, 6, 7, 8]}] +
                       [{"add": "Governor Hong rejected it.", "src": [3, 4, 5, 6, 7, 8]}] * 4)
    reasons = []
    assert label_sentence(ROW_KO, steps, lambda r, p: {"regret": [2], "note": "bad"}, max_regret=2,
                          fail=reasons.append) is None
    assert reasons[-1] == "regret unresolved"


# --- main: resume / budget --------------------------------------------------------------------------------------
def test_resume_skips_done_and_counts_spent(tmp_path):
    from label_prefix import plan_todo
    out = tmp_path / "o.jsonl"
    out.write_text('{"sid": "a"}\n')
    (tmp_path / "o.jsonl.fail.jsonl").write_text('{"sid": "b", "reason": "x"}\n')
    (tmp_path / "o.jsonl.usage.jsonl").write_text('{"est_usd": 0.5}\n{"est_usd": 0.25}\n')
    rows = [{"sid": s} for s in "abc"]
    todo, spent = plan_todo(rows, str(out))
    assert [r["sid"] for r in todo] == ["c"] and spent == 0.75
