import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix_points import candidate_points, ko_phrase_end  # noqa: E402


def test_en_points_are_clause_ends():
    toks = "the manager approved the budget , and the team left early".split()
    assert candidate_points(toks, "en") == [6, 11]  # ","(token 6) before the conjunction, sentence end


def test_ko_phrase_end_tags():
    assert ko_phrase_end("지사는") and ko_phrase_end("회의에서") and ko_phrase_end("서울부터") and ko_phrase_end("1990년에,")
    assert not ko_phrase_end("학생이") and not ko_phrase_end("도구를") and not ko_phrase_end("승인했다")


def test_ko_points_punct_and_numbers():
    toks = "어제 회의에서 홍 지사는 새 예산을 1990년에 승인했다".split()
    # 회의에서(2), 지사는(4): gap 2 -> both; 1990년에(7) next to the end(8), which never displaces a point
    assert candidate_points(toks, "ko") == [2, 4, 7, 8]


def test_ko_min_gap_drops_adjacent():
    toks = "그는 서울에서 일했다".split()  # 그는(1, JX) and 서울에서(2) are 1 apart -> 2 survives, then the end 3
    assert candidate_points(toks, "ko", min_gap=2) == [2, 3]


def test_last_token_always_included():
    assert candidate_points(["hello"], "en") == [1]
    assert candidate_points(["안녕하세요"], "ko") == [1]


def test_en_phrase_end_before_preposition_or_relativizer():
    toks = "in one methodology , a static document is loaded into printer memory".split()
    # "," (4) ; before "into" (9) ; end (12)
    assert candidate_points(toks, "en") == [4, 9, 12]
    toks = "it is britain's oldest complete human skeleton".split()
    assert candidate_points(toks, "en") == [7]  # no phrase boundary: "of"-less NP stays one unit
    toks = "the skeleton that was found of the man".split()
    assert candidate_points(toks, "en") == [2, 8]  # before "that"; never before "of"


def test_consecutive_candidates_do_not_collapse_into_one():
    # 회의에서(2) 서울시는(3) 내년부터(4): 3 is within the gap of 4 and dropped, 2 and 4 are 2 apart and both stay
    toks = "어제 회의에서 서울시는 내년부터 새 예산을 심의한다".split()
    assert candidate_points(toks, "ko") == [2, 4, 7]
