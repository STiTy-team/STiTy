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


def test_clause_ends_ko_particles_are_not_clause_ends():
    toks = words("학교에서 친구를 만났고 최고 광고를 측면 보았다", "ko")
    # 학교에서(1) particle, 만났고(3) connective, 최고(4)/측면(6) one-syllable noun remainders
    assert clause_ends(toks, "ko") == [3, 7]


def test_clause_ends_ko_short_verbal_stems():
    toks = words("밥을 먹었고 숙제를 하고 잤다", "ko")
    assert clause_ends(toks, "ko") == [2, 4, 5]


def test_parse_covers_and_self_aligned_rules():
    from schedule_gate import parse_covers
    obj = {"schedule": [{"after": 2, "text": "서한은", "src": [1, 2]}, {"after": 5, "text": "옹호했다", "src": [3]}, {"after": 6, "text": "", "src": []}]}
    assert parse_covers(obj) == [{1, 2}, {3}, set()]
    assert parse_covers({"schedule": [{"after": 1, "text": "x"}]}) == [None]
    try:
        parse_covers({"schedule": [{"after": 1, "text": "x", "src": "1"}]})
        assert False
    except ValueError as e:
        assert str(e).startswith("format")
    toks = words("This letter advocated constitutional and parliamentary governance.", "en")
    hy = lambda k: set()  # Hy knows nothing: without src every non-empty item would be a lookahead FAIL
    kw = dict(known_at=hy, full_known={"서한", "옹호"}, N=8, clause_end=clause_ends(toks, "en"))
    items = [(2, "서한은"), (4, "옹호했다")]  # clause ends at 4 (before "and") and 7
    assert check_schedule(items, toks, "en", "ko", covers=[{1, 2}, {3}], **kw) is None
    assert check_schedule(items, toks, "en", "ko", covers=[{1, 3}, {3}], **kw).startswith("lookahead: src words [3]")
    assert check_schedule([(2, "서한은"), (7, "옹호했다")], toks, "en", "ko", covers=[{1, 2}, {3}], **kw).startswith("clause")
    assert check_schedule([(2, "서한은")], toks, "en", "ko", covers=[set()], **kw).startswith("range: item after=2 has text")
    assert check_schedule(items, toks, "en", "ko", covers=[{1, 2}], **kw).startswith("range: 1 src lists")
    # a src-less item still goes through the Hy rule
    assert check_schedule(items, toks, "en", "ko", covers=[{1, 2}, None], **kw).startswith("lookahead: 옹호")


def test_shape_bare_noun_rejected_unless_particle_follows():
    from schedule_gate import shape_check
    assert shape_check([(2, "이 문장"), (4, "특별히"), (7, "당신을 위해 만들어졌습니다.")], "ko").startswith("shape: piece 1")
    assert shape_check([(4, "응답률"), (6, "도"), (7, "더 높습니다.")], "ko") is None
    assert shape_check([(1, "그것은"), (3, "영국의"), (4, "가장 오래된"), (7, "골격이다.")], "ko") is None


def test_shape_english_dangling_and_lone_conjunction():
    from schedule_gate import shape_check
    assert "dangling" in shape_check([(2, "Based on this,"), (9, "is one of the sources, so"), (12, "I introduced it.")], "en")
    assert shape_check([(1, "But"), (2, "even this"), (7, "I can't.")], "en") is None
    assert "splits a word" in shape_check([(2, "the son-in-law"), (3, "'s"), (5, "sentence was light.")], "en")


def test_glues_needs_bare_noun_before_and_particle_after():
    from schedule_gate import glues, join_texts
    assert join_texts(["응답률", "도", "더 높습니다."]) == "응답률도 더 높습니다."
    assert join_texts(["팔과 턱", "에 상처를 입었고,"]) == "팔과 턱에 상처를 입었고,"
    assert join_texts(["빠르게", "이 숫자를"]) == "빠르게 이 숫자를"
    assert join_texts(["그래서,", "이 GU 종양 영역에서"]) == "그래서, 이 GU 종양 영역에서"
    assert glues("그래서,", "도") is False


def test_fix_leading_punct_and_merge_same_after():
    from schedule_gate import fix_leading_punct, merge_same_after
    assert fix_leading_punct([(9, "there is a bomb"), (10, ", namely,"), (15, " the spread")]) == [(9, "there is a bomb,"), (10, " namely,"), (15, " the spread")]
    items, covers = merge_same_after([(2, "또한,"), (13, "여러분의 모든 전기에"), (13, "깊은 인상을 받았습니다.")], [{1, 2}, {3}, {4}])
    assert items == [(2, "또한,"), (13, "여러분의 모든 전기에 깊은 인상을 받았습니다.")] and covers == [{1, 2}, {3, 4}]
