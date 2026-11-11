import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schedule_stats as ss  # noqa: E402

SRC = "a b c d e f"


def _row(sid, sched):
    return {"sid": sid, "lang": "en", "tgt": "ko", "src": SRC, "ref": None, "schedule": sched}


def toy():
    return [
        _row("dump", [[5, "x y"], [6, "z"]]),
        _row("spread", [[2, "x"], [4, "y"], [6, "z"]]),
        _row("one", [[1, "x y z"]]),
    ]


def tok(src, lang):
    return src.split()


def test_stats_toy():
    s = ss.stats_for(toy(), tok)
    assert s["n"] == 3
    assert abs(s["dump_share"] - 1 / 3) < 1e-9
    assert s["first_after_median"] == 2


def test_empty_share_spread():
    s = ss.stats_for([toy()[1]], tok)
    assert abs(s["empty_share"] - 0.5) < 1e-9


def test_deferral_uses_known_fn():
    known = {"spread": [{"x"}, {"x"}, {"x", "y"}, {"x", "y"}, {"x", "y"}, {"x", "y", "z"}]}
    row = toy()[1]
    s = ss.stats_for([row], tok, known_fn=lambda sid: known[sid], content_fn=lambda t, tgt: {t})
    # x opens at 2, known at k=1 -> 1; y at 4, known k=3 -> 1; z at 6, k=6 -> 0
    assert s["deferral_max"] == 1 and s["deferral_median"] == 1


def test_skip_completeness_main(tmp_path):
    sd = tmp_path
    sent = [dict(sid=r["sid"], lang="en", tgt="ko", src=SRC, ref=None, source="t", recs=[]) for r in toy()]
    (sd / "s.jsonl").write_text("\n".join(json.dumps(r) for r in sent))
    (sd / "h.jsonl").write_text("\n".join(json.dumps({"sid": r["sid"], "prefixes": ["a"] * 6}) for r in toy()))
    sch = [dict(sid=r["sid"], lang="en", tgt="ko", schedule=r["schedule"], attempts=1) for r in toy()]
    (sd / "sch.jsonl").write_text("\n".join(json.dumps(r) for r in sch))
    out = sd / "g.jsonl"
    ss.main(["--schedule", str(sd / "sch.jsonl"), "--sentences", str(sd / "s.jsonl"), "--hy", str(sd / "h.jsonl"),
             "--out-gated", str(out), "--skip-completeness"])
    assert len(out.read_text().strip().splitlines()) == 3


def test_empty_items_are_not_emissions():
    # an early empty item at word 1 must not count as an emission position, first item, or anti-dump
    s = ss.stats_for([_row("w", [[1, ""], [5, "x y"], [6, "z"]])], tok)
    assert s["first_after_median"] == 5
    assert abs(s["empty_share"] - (1 - 2 / 6)) < 1e-9
    assert s["dump_share"] == 1.0


def test_fully_dropped_direction_gets_zero_row(tmp_path, monkeypatch, capsys):
    sent = [dict(sid="e", lang="en", tgt="ko", src=SRC, ref=None, source="t", recs=[]),
            dict(sid="k", lang="ko", tgt="en", src=SRC, ref=None, source="t", recs=[])]
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in sent))
    (tmp_path / "h.jsonl").write_text("\n".join(json.dumps({"sid": r["sid"], "prefixes": ["a"] * 6}) for r in sent))
    sch = [dict(sid=r["sid"], lang=r["lang"], tgt=r["tgt"], schedule=[[6, "x"]], attempts=1) for r in sent]
    (tmp_path / "sch.jsonl").write_text("\n".join(json.dumps(r) for r in sch))
    monkeypatch.setattr(ss, "gate", lambda rows, *a: [r for r in rows if r["lang"] == "en"])
    ss.main(["--schedule", str(tmp_path / "sch.jsonl"), "--sentences", str(tmp_path / "s.jsonl"),
             "--hy", str(tmp_path / "h.jsonl"), "--out-gated", str(tmp_path / "g.jsonl")])
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("ko->en")][0]
    assert line.split()[1] == "0" and line.split()[-1] == "1"
