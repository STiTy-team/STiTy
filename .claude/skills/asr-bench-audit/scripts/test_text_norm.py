# -*- coding: utf-8 -*-
"""표기 정규화 회귀 테스트.

여기 있는 사례는 전부 **실제로 틀렸던 것**이다. 주석으로만 남겨 두면 다음 사람이
같은 방식으로 다시 깨뜨린다.

    python3 test_text_norm.py        # pytest 없이도 돈다
"""
from __future__ import print_function
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_norm import (canon_numbers_en, canon_numbers_ko,  # noqa: E402
                       en_read, score, score_pair, strip_group_sep)


# ── 불변식: 정규화는 점수를 나쁘게 만들 수 없다 ──────────────────────────────
# 참조와 똑같은 전사는 원값이 0 이다. 정규화 후에도 0 이어야 한다.
# 이 불변식이 깨지면 정규화가 맞는 글자를 틀리게 바꾼 것이다.
IDENTITY_CASES = [
    ("en", "wer", "The 2005 census counted twenty five villages."),
    ("en", "wer", "By 2005, twenty five percent had left."),
    ("en", "wer", "Between 2001 and 2009 the number rose to twenty one."),
    ("en", "wer", "He walked five hundred metres in 1905."),
    ("en", "wer", "The 1900 report listed nineteen cases."),
    ("ko", "cer", "2011년에 스물다섯 명이 모였다."),
    ("ko", "cer", "데이포트로 가는 길에 타이브레이크가 있었다."),
    ("ko", "cer", "2015년 기준 사육 두수는 육십만 마리였다."),
]


def test_identity_never_worsens():
    for lang, unit, text in IDENTITY_CASES:
        norm, raw = score_pair(text, text, unit, lang)
        assert raw == 0.0, "원값이 0 이 아니다: %r -> %r" % (text, raw)
        assert norm == 0.0, "정규화가 정답을 깨뜨렸다: %r -> %r" % (text, norm)


# ── 영어: 연도 읽기가 다른 수와 겹치면 안 된다 ───────────────────────────────
def test_year_reading_does_not_collide():
    # 2005 를 "twenty five" 로 읽으면 25 와 구별되지 않는다.
    for n in range(2001, 2010):
        forms = en_read(n)
        for f in forms:
            assert f != "twenty " + en_read(n % 100)[0], \
                "%d 가 %r 을 만든다 — %d 와 겹친다" % (n, f, n % 100)
    assert "twenty oh five" in en_read(2005), en_read(2005)
    assert "nineteen oh five" in en_read(1905), en_read(1905)
    assert "nineteen hundred" in en_read(1900), en_read(1900)


def test_en_true_positive():
    # 말로 쓴 수는 참조의 숫자 표기로 되돌아와야 한다.
    assert canon_numbers_en("in 2011", "in twenty eleven") == "in 2011"
    assert canon_numbers_en("in 2011", "in two thousand eleven") == "in 2011"
    assert canon_numbers_en("in 1905", "in nineteen oh five") == "in 1905"
    assert canon_numbers_en("about 25", "about twenty five") == "about 25"


def test_en_guard_blocks_partial_match():
    # 더 긴 수의 일부를 떼어 바꾸면 안 된다.
    assert canon_numbers_en("only 5", "twenty five people") == "twenty five people"
    assert canon_numbers_en("only 5", "five hundred people") == "five hundred people"


# ── 한국어: 일상 음절과 겹치는 수사를 건드리면 안 된다 ───────────────────────
def test_ko_false_positive_regression():
    # 실제로 났던 오검출: 데이포트로 -> 데2포트로, 타이브레이크 -> 타2브레2크
    assert canon_numbers_ko("2번 출구", "데이포트로 갔다") == "데이포트로 갔다"
    assert canon_numbers_ko("2세트", "타이브레이크 끝에") == "타이브레이크 끝에"
    # 사육(飼育)이 46 이 되면 안 된다
    assert "46" not in canon_numbers_ko("4명과 6명", "사육 환경이 나쁘다")


def test_ko_true_positive():
    assert canon_numbers_ko("2011년", "이천십일년") == "2011년"
    assert canon_numbers_ko("15미터", "십오미터") == "15미터"


def test_group_separator():
    # 자릿점 때문에 참조의 수가 쪼개져, "ten thousand" 가 "10 thousand" 로 반쪽만
    # 치환되던 사례(FLEURS en, Nemotron). 전사는 맞았는데 점수가 깎였다.
    ref = ("Goats seem to have been first domesticated roughly 10,000 years ago "
           "in the Zagros Mountains of Iran.")
    hyp = ("Goods seem to have been first domesticated roughly ten thousand years ago "
           "in the Zagros Mountains of Iran")
    norm, raw = score_pair(ref, hyp, "wer", "en")
    # 남는 오류는 Goats -> Goods 하나뿐이어야 한다 (18 낱말 중 1)
    assert abs(norm - 1.0 / 17) < 0.01, norm
    assert norm < raw, (norm, raw)
    assert strip_group_sep("1,234,567 and 1,2,3") == "1234567 and 1,2,3"


def test_score_pair_reports_raw():
    # 정규화 후와 원값을 둘 다 돌려준다. 보고서가 둘을 나란히 싣는다.
    norm, raw = score_pair("2011년", "이천십일년", "cer", "ko")
    assert norm == 0.0, norm
    assert raw > 0.0, raw


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print("FAIL %s: %s" % (name, e))
        else:
            print("ok   %s" % name)
    print("\n%d/%d 통과" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
