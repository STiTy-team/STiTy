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
