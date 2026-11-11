import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_word_prefixes import known_sets, prefix_strings  # noqa: E402


def test_prefix_strings_one_per_token():
    p = prefix_strings("회의가  끝나면 보고서를 보낼게요.", "ko")
    assert p == ["회의가", "회의가 끝나면", "회의가 끝나면 보고서를", "회의가 끝나면 보고서를 보낼게요."]
    e = prefix_strings("It was done today.", "en")
    assert len(e) == 4 and e[-1] == "It was done today."


def test_known_sets_cumulative_ko():
    ks = known_sets(["보고서를", "보고서를 보낼게요", "보고서를 보낼게요 회의가"], "ko")
    assert len(ks) == 3
    assert ks[0] and ks[0] <= ks[1] <= ks[2]


def test_known_sets_never_shrinks_when_later_prefix_drops_words():
    ks = known_sets(["The report was sent yesterday", "Something else"], "en")
    assert ks[0] and ks[0] <= ks[1]
    assert known_sets([], "en") == []
