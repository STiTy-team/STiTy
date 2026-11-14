import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_go  # noqa: E402


def _args(**kw):
    d = dict(min_kept=2, min_kept_share=0.5, dump_enko=0.05, dump_other=0.10, empty_ratio=2.0, first_after=6)
    d.update(kw)
    return type("A", (), d)()


def _row(lang, tgt, src, sch):
    return {"sid": f"{lang}{len(src)}", "lang": lang, "tgt": tgt, "src": src, "schedule": sch}


def test_pass():
    rows = [_row("en", "ko", "a b c d e f", [[2, "x"], [4, "y"], [6, "z"]]),
            _row("ko", "en", "가 나 다 라 마 바", [[2, "x"], [4, "y"], [6, "z"]])]
    _, reasons = auto_go.check(rows, 2, _args())
    assert reasons == []


def test_dump_and_kept_and_ratio():
    rows = [_row("en", "ko", "a b c d e f", [[6, "x y z"]]),
            _row("ko", "en", "가 나 다 라 마 바", [[1, "x"], [2, "y"], [3, "z"], [4, "w"], [5, "v"], [6, "u"]])]
    _, reasons = auto_go.check(rows, 10, _args())
    joined = " ".join(reasons)
    assert "en-ko dump_share 1.000 > 0.05" in joined
    assert "50% of labeled 10" in joined
    assert "empty_share ratio" in joined


def test_main_merges_src_from_sentences(tmp_path, capsys):
    import json
    (tmp_path / "sent.jsonl").write_text(
        json.dumps({"sid": "a", "lang": "en", "tgt": "ko", "src": "a b c d e f", "ref": "x", "recs": []}) + "\n"
        + json.dumps({"sid": "b", "lang": "ko", "tgt": "en", "src": "가 나 다 라 마 바", "ref": "y", "recs": []}) + "\n")
    gated = [{"sid": "a", "lang": "en", "tgt": "ko", "schedule": [[2, "x"], [4, "y"], [6, "z"]], "attempts": 1},
             {"sid": "b", "lang": "ko", "tgt": "en", "schedule": [[2, "x"], [4, "y"], [6, "z"]], "attempts": 1}]
    (tmp_path / "g.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gated))
    (tmp_path / "l.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gated))
    rc = auto_go.main(["--gated", str(tmp_path / "g.jsonl"), "--labeled", str(tmp_path / "l.jsonl"),
                       "--sentences", str(tmp_path / "sent.jsonl"), "--min-kept", "2"])
    assert rc == 0 and "AUTO_GO_YES" in capsys.readouterr().out


def test_too_many_wait_labels_is_no():
    good = [_row("en", "ko", "a b c d e f", [[2, "x"], [4, "y"], [6, "z"]]),
            _row("ko", "en", "가 나 다 라 마 바", [[2, "x"], [4, "y"], [6, "z"]])]
    waits = [{**_row("en", "ko", "a b c d e f g", [[7, "all"]]), "sid": "w1", "rescue": "wait"}]
    _, reasons = auto_go.check(good + waits, 3, _args(rescued_max=0.25))
    assert any("rescued wait labels 1/3" in r for r in reasons)
    _, reasons = auto_go.check(good + waits, 3, _args(rescued_max=0.5))
    assert reasons == []


def test_new_thresholds_stop():
    import argparse
    rows = [{"sid": f"a{i}", "lang": "en", "tgt": "ko", "src": "the manager approved the budget , and the team left early",
             "schedule": [[6, "매니저가 예산을 승인했습니다."], [11, "팀은 일찍 떠났습니다."]],
             "src_words": [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]]} for i in range(10)]
    a = argparse.Namespace(min_kept=1, min_kept_share=0.0, dump_enko=1.0, dump_other=1.0, empty_ratio=99, first_after=99,
                           rescued_max=1.0, mid_closed=0.03, first_bare=0.05, empty_min=0.0, regret_max=1.0, lag_max=99)
    _, reasons = auto_go.check(rows, 10, a)
    assert any(r.startswith("en-ko mid_closed_share") for r in reasons)
    a.mid_closed = 1.0
    a.empty_min = 0.05  # no empty items at all -> below the floor
    _, reasons = auto_go.check(rows, 10, a)
    assert any(r.startswith("en-ko point_empty_share") for r in reasons)
