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
    monkeypatch.setattr(ss, "gate", lambda rows, *a, **k: ([r for r in rows if r["lang"] == "en"],
                                                            [r for r in rows if r["lang"] != "en"], {}))
    ss.main(["--schedule", str(tmp_path / "sch.jsonl"), "--sentences", str(tmp_path / "s.jsonl"),
             "--hy", str(tmp_path / "h.jsonl"), "--out-gated", str(tmp_path / "g.jsonl")])
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.startswith("ko->en") and "kept" not in l][0]
    assert line.split()[1] == "0" and line.split()[1 + ss._COLS.index("gated_out")] == "1"
    assert "wait 1" in [l for l in out.splitlines() if l.startswith("ko->en") and "kept" in l][0]  # dropped -> wait label (Hy)


def _sent(sid, lang, tgt, src, ref):
    return {"sid": sid, "lang": lang, "tgt": tgt, "src": src, "ref": ref, "source": "t", "recs": []}


def test_wait_labels_reference_whole_and_clause_split():
    sents = {"a": _sent("a", "en", "ko", "The manager approved the budget yesterday.", "매니저가 어제 예산을 승인했어요."),
             "b": _sent("b", "en", "ko", "If it rains, we stay home.", "비가 오면, 우리는 집에 있어요.")}
    labels, dropped = ss.make_wait_labels(["a", "b"], sents, {}, {}, None)
    assert dropped == []
    assert labels[0]["schedule"] == [[6, "매니저가 어제 예산을 승인했어요."]] and labels[0]["rescue"] == "wait"
    assert labels[0]["why"] == "comet"
    assert labels[1]["schedule"] == [[6, "비가 오면, 우리는 집에 있어요."]]  # one sentence, two clauses: whole


def test_wait_labels_two_sentences_split_at_clause_ends():
    sents = {"c": _sent("c", "ko", "en", "회의가 끝났어요. 자료를 보낼게요.", "The meeting ended. I'll send the file."),
             "d": _sent("d", "ko", "en", "회의가 끝나면 자료를 보낼게요.", "When the meeting ends. I'll send the file.")}
    labels, _ = ss.make_wait_labels(["c", "d"], sents, {}, {}, None, {"c": "judge"})
    assert labels[0]["schedule"] == [[2, "The meeting ended."], [4, "I'll send the file."]] and labels[0]["why"] == "judge"
    assert labels[1]["schedule"] == [[4, "When the meeting ends. I'll send the file."]]  # a connective ending is not a split point


def test_wait_labels_covost_uses_hy_and_kiwi_floor():
    sents = {"d": _sent("d", "en", "ko", "a b c", None), "e": _sent("e", "en", "ko", "a b c", None)}
    hy = {"d": ["x", "xy", "Hy 번역 d"], "e": ["x", "xy", "Hy 번역 e"]}
    labels, dropped = ss.make_wait_labels(["d", "e"], sents, hy, {"d": 0.9, "e": 0.5}, 0.7)
    assert [l["sid"] for l in labels] == ["d"] and labels[0]["schedule"] == [[3, "Hy 번역 d"]]
    assert dropped == [{"sid": "e", "why": "hy_kiwi_floor", "kiwi_hy": 0.5}]


def test_short_share_excludes_connectives_and_refuses_wait_rows():
    rows = [{"sid": "x", "lang": "en", "tgt": "ko", "src": "a b c d", "schedule": [[1, "그리고"], [2, "응답률도"], [4, "더 높습니다."]]}]
    s = ss.stats_for(rows, tok)
    assert s["short_share"] == 0.0 and s["piece_len_median"] == 4 and s["single_share"] == 0.0
    import pytest
    with pytest.raises(ValueError):
        ss.stats_for([{**rows[0], "rescue": "wait"}], tok)


def test_rescue_report_counts():
    sents = {"a": _sent("a", "en", "ko", "a", "x"), "b": _sent("b", "en", "ko", "a", "x"), "c": _sent("c", "ko", "en", "가", "y")}
    kept = [{"sid": "a", "attempts": 2, "judge_attempts": 2}, {"sid": "c", "attempts": 1, "judge_attempts": 1}]
    waits = [{"sid": "b", "rescue": "wait", "why": "judge"}]
    judge = [{"sid": "a", "pass": False}, {"sid": "b", "pass": False}, {"sid": "c", "pass": True}]
    r = ss.rescue_report(kept, waits, sents, judge)
    assert r["en->ko"]["wait"] == 1 and r["en->ko"]["rescued_wait_share"] == 0.5 and r["en->ko"]["by_why"] == {"judge": 1}
    assert r["en->ko"]["retried_share"] == 1.0 and r["en->ko"]["relabeled_share"] == 1.0 and r["ko->en"]["relabeled_share"] == 0.0
    assert r["en->ko"]["judge_fail_share"] == 1.0 and r["ko->en"]["judge_fail_share"] == 0.0


def test_wait_labels_do_not_split_at_connectives_or_abbreviations():
    sents = {"f": _sent("f", "ko", "en", "트럭은 더 달려 박씨를 치고 멈췄다.", "The truck ran on, hit Mr. Park and stopped."),
             "g": _sent("g", "en", "ko", "He left. She stayed.", "그는 떠났다. 그녀는 남았다.")}
    labels, _ = ss.make_wait_labels(["f", "g"], sents, {}, {}, None)
    assert labels[0]["schedule"] == [[6, "The truck ran on, hit Mr. Park and stopped."]]  # connective ends are not splits
    assert labels[1]["schedule"] == [[2, "그는 떠났다."], [4, "그녀는 남았다."]]


# --- prefix16 columns -------------------------------------------------------------------------------------------
def test_new_columns():
    from prefix_points import candidate_points
    from schedule_gate import words
    rows = [{"sid": "a", "lang": "en", "tgt": "ko", "src": "the manager approved the budget , and the team left early",
             "schedule": [[6, "매니저가 예산을 승인했고,"], [11, "팀은 일찍 떠났습니다."]],
             "src_words": [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]], "regret_rounds": 1},
            {"sid": "b", "lang": "en", "tgt": "ko", "src": "hello there friend",
             "schedule": [[3, "안녕 친구."]], "src_words": [[1, 2, 3]], "regret_rounds": 0}]
    m = ss.stats_for(rows, words, points_fn=candidate_points)
    assert m["mid_closed_share"] == 0.0 and m["first_bare_subj"] == 0.0 and m["regret_share"] == 0.5
    assert m["point_empty_share"] == 0.0
    assert abs(m["first_after_frac"] - (6 / 11 + 1.0) / 2) < 1e-6  # median of two values = their mean
    rows[0]["schedule"][0][1] = "매니저가 예산을 승인했습니다."
    assert ss.stats_for(rows, words, points_fn=candidate_points)["mid_closed_share"] == 1.0


def test_point_empty_share_and_bare_subject():
    from schedule_gate import words
    rows = [{"sid": "a", "lang": "ko", "tgt": "en", "src": "어제 회의에서 홍 지사는 새 예산을 1990년에 승인했다",
             "schedule": [[2, "At yesterday's meeting,"], [4, ""], [7, ""], [8, "Governor Hong approved the new budget in 1990."]],
             "src_words": [[1, 2], [], [], [3, 4, 5, 6, 7, 8]]},
            {"sid": "b", "lang": "ko", "tgt": "en", "src": "지각은 가까운 쪽이 약 70km 두께입니다",
             "schedule": [[1, "The crust"], [6, "is about 70 km thick."]], "src_words": [[1], [2, 3, 4, 5, 6]]}]
    m = ss.stats_for(rows, words)
    assert abs(m["point_empty_share"] - 2 / 6) < 1e-6 and m["first_bare_subj"] == 0.5


def test_est_lag_from_word_times():
    from schedule_gate import words
    rows = [{"sid": "a", "lang": "en", "tgt": "ko", "src": "the manager approved",
             "schedule": [[3, "매니저가 승인했습니다."]], "src_words": [[1, 2, 3]]}]
    wt = {"a": [{"w": "the", "end": 0.5}, {"w": "manager", "end": 1.0}, {"w": "approved", "end": 2.0}]}
    m = ss.stats_for(rows, words, word_times=wt)
    assert abs(m["est_lag_s"] - 1.5) < 1e-6  # earliest src word (the, 0.5 s) -> after=3 (2.0 s)
    assert ss.stats_for(rows, words)["est_lag_s"] is None


def test_print_table_tolerates_missing_columns(capsys):
    ss.print_table("old", {"en->ko": {"n": 3}})  # stats_old rows lack the prefix16 columns
    out = capsys.readouterr().out
    assert "en->ko" in out and "-" in out
