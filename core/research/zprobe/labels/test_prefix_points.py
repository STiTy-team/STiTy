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
