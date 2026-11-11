import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule_gate import check_schedule, clause_ends, eojeol_end_times, parse_schedule, words  # noqa: E402


def test_words_ko_eojeol():
    assert words("회의가 끝나면 보고서를 보낼게요.", "ko") == ["회의가", "끝나면", "보고서를", "보낼게요."]


def test_parse_schedule_rejects_missing_after():
    with pytest.raises(ValueError):
        parse_schedule({"schedule": [{"text": "x"}]})


def test_check_range_and_order():
    toks = words("The manager approved the budget", "en")
    assert check_schedule([(0, "매니저가")], toks, "en", "ko", known_at=lambda k: set(), full_known=set(), N=8, clause_end=[5]).startswith("range")
    assert check_schedule([(3, "a"), (2, "b")], toks, "en", "ko", known_at=lambda k: set(), full_known=set(), N=8, clause_end=[5]).startswith("range")


def test_check_all_empty():
    toks = words("Hello there", "en")
    assert check_schedule([(2, "")], toks, "en", "ko", known_at=lambda k: set(), full_known=set(), N=8, clause_end=[2]) == "all additions are empty"


def test_check_lookahead():
    toks = words("I will send the report tomorrow", "en")
    known = {1: set(), 2: set(), 3: {"보내"}, 4: {"보내"}, 5: {"보내", "보고"}, 6: {"보내", "보고", "내일"}}
    why = check_schedule([(3, "내일 보낼게요")], toks, "en", "ko", known_at=lambda k: known[k], full_known={"보내", "보고", "내일"}, N=8, clause_end=[6])
    assert why.startswith("lookahead") and "내일" in why


def test_check_deferral_cap():
    toks = words("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12", "en")
    known = lambda k: {"매니"} if k >= 1 else set()
    assert check_schedule([(12, "매니저가")], toks, "en", "ko", known_at=known, full_known={"매니"}, N=8, clause_end=[12]).startswith("deferral")
    assert check_schedule([(9, "매니저가")], toks, "en", "ko", known_at=known, full_known={"매니"}, N=8, clause_end=[12]) is None


def test_check_clause_end():
    toks = words("He left, and she stayed.", "en")  # clause ends at 2 ("left,") and 5
    known = lambda k: {"매니"} if k >= 2 else set()
    assert check_schedule([(5, "매니저가 떠났고")], toks, "en", "ko", known_at=known, full_known={"매니"}, N=8, clause_end=clause_ends(toks, "en")).startswith("clause")


def test_eojeol_end_times_merged_tokens():
    toks = ["자각몽을", "꾸었다"]
    wt = [{"w": "자각", "end": 0.4}, {"w": "몽을", "end": 0.8}, {"w": "꾸었다", "end": 1.3}]
    assert eojeol_end_times(toks, wt) == [0.8, 1.3]


def test_check_repeated_word_checked_on_first_emission_only():
    # "We collected the data, and then we finally analyzed the data" - 데이터 is heard at 4,
    # emitted at 4, and repeated at 11; the repeat must not count as a late emission.
    toks = words("We collected the data, and then we finally analyzed the data", "en")
    known = lambda k: ({"데이"} if k >= 4 else set()) | ({"분석"} if k >= 10 else set())
    items = [(4, "우리는 데이터를 수집했고,"), (11, "마침내 데이터를 분석했어요.")]
    assert check_schedule(items, toks, "en", "ko", known_at=known, full_known={"데이", "분석"}, N=5, clause_end=clause_ends(toks, "en")) is None


def test_check_empty_items_are_waits():
    toks = words("I will send the report tomorrow", "en")
    known = {1: set(), 2: set(), 3: {"보내"}, 4: {"보내"}, 5: {"보내", "보고"}, 6: {"보내", "보고", "내일"}}
    items = [(1, ""), (3, "  "), (6, "내일 보고서를 보낼게요.")]
    assert check_schedule(items, toks, "en", "ko", known_at=lambda k: known[k], full_known={"보내", "보고", "내일"}, N=8, clause_end=[6]) is None
