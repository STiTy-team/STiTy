import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose_judged import compose  # noqa: E402


def test_each_sid_lands_in_exactly_one_place():
    s1 = [{"sid": "ok", "schedule": [[1, "a"]]}, {"sid": "fixed", "schedule": [[1, "b"]]},
          {"sid": "still", "schedule": [[1, "c"]]}, {"sid": "blind", "schedule": [[1, "d"]]}]
    j1 = [{"sid": "ok", "pass": True}, {"sid": "fixed", "pass": False}, {"sid": "still", "pass": False},
          {"sid": "blind", "pass": True, "unjudged": True}]
    s2 = [{"sid": "fixed", "schedule": [[1, "B"]]}, {"sid": "still", "schedule": [[1, "C"]]}]
    j2 = [{"sid": "fixed", "pass": True}, {"sid": "still", "pass": False}]
    kept, rescue = compose(s1, j1, s2, j2, fail_sids=["gatefail", "ok"])
    by = {r["sid"]: r for r in kept}
    assert by["ok"]["judge_attempts"] == 1 and by["blind"]["judge_attempts"] == 1
    assert by["fixed"]["schedule"] == [[1, "B"]] and by["fixed"]["judge_attempts"] == 2
    assert rescue == [{"sid": "still", "why": "judge"}, {"sid": "gatefail", "why": "fail"}]
    all_sids = [r["sid"] for r in kept] + [r["sid"] for r in rescue]
    assert sorted(all_sids) == ["blind", "fixed", "gatefail", "ok", "still"]


def test_error_sids_rescued_and_missing_judge_row_flagged():
    s1 = [{"sid": "ok", "schedule": [[1, "a"]]}, {"sid": "nojudge", "schedule": [[1, "b"]]}]
    j1 = [{"sid": "ok", "pass": True}]
    kept, rescue = compose(s1, j1, [], [], fail_sids=["gatefail"], all_sids=["ok", "nojudge", "gatefail", "apierr"])
    by = {r["sid"]: r for r in kept}
    assert by["nojudge"].get("unjudged") is True and "unjudged" not in by["ok"]
    assert rescue == [{"sid": "gatefail", "why": "fail"}, {"sid": "apierr", "why": "error"}]
