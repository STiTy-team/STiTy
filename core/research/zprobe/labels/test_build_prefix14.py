import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_prefix14 as b  # noqa: E402

ITEMS = [(2, "매니저가"), (9, "새 사무실 예산을"), (10, "어제 승인했어요")]
ENDS = [1.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7]
TAG = "language Korean<asr_text>"


def test_on_time_rows():
    rows = b.rows_on_time(ITEMS, ENDS, 5.0, "en-ko", "/x/a.wav")
    early = [r for r in rows if float(r["audio"].split("#end=")[1]) < 2.5]
    assert early and all(r["text"] == "" and r["cond"] == TAG for r in early)
    first = next(r for r in rows if r["text"])
    assert first["cond"] == TAG and first["text"] == "매니저가"
    assert first["audio"] == "/x/a.wav#end=2.500"
    last = rows[-1]
    assert "매니저가" in last["cond"] and "새 사무실 예산을" in last["cond"]
    assert all(r["pair"] == "en-ko" and r["prompt"] == "" for r in rows)
    # the 2nd item is delta-joined with a leading space onto a non-empty cond
    second = next(r for r in rows if r["text"].strip() == "새 사무실 예산을")
    assert second["text"].startswith(" ") and second["cond"] == TAG + "매니저가"


def test_start_end_cut():
    assert b.cut("/x/a.wav", 2.0) == "/x/a.wav#end=2.000"
    assert b.cut("/x/a.wav#start=3.000#end=9.000", 2.0) == "/x/a.wav#start=3.000#end=5.000"
    assert b.cut("/x/a.wav#start=3.000#end=9.000", 20.0) == "/x/a.wav#start=3.000#end=9.000"


def test_catchup_text_joins_lagged_items():
    rows = b.rows_catchup(ITEMS, ENDS, 5.0, "en-ko", "/x/a.wav", random.Random(0), n=3, lag=(2, 2))
    assert len(rows) == 1  # only t=5.0 has >= 2 items in the on-time cond at the previous cut (4.9)
    assert rows[0]["cond"] == TAG + "매니저가"
    assert rows[0]["text"] == " 새 사무실 예산을 어제 승인했어요"


def test_catchup_cond_lags_on_time_cond():
    on = {r["audio"]: r for r in b.rows_on_time(ITEMS, ENDS, 5.0, "en-ko", "/x/a.wav")}
    rows = b.rows_catchup(ITEMS, ENDS, 5.0, "en-ko", "/x/a.wav", random.Random(0), n=20, lag=(1, 1))
    assert rows
    for r in rows:
        if r["audio"] not in on:  # t = dur: on_time leaves that cut to rows_end
            continue
        o = on[r["audio"]]
        assert r["cond"] != o["cond"] and o["cond"].startswith(r["cond"])
        assert r["text"].strip().endswith(o["text"].strip())
    # t=5.0: on-time prev state (4.9) has all 3 items; catch-up holds the last one back
    r5 = next(r for r in rows if r["audio"].endswith("#end=5.000"))
    assert r5["cond"] == TAG + "매니저가 새 사무실 예산을" and r5["text"] == " 어제 승인했어요"


def test_word_heard_only_after_pad():
    rows = {r["audio"]: r for r in b.rows_on_time(ITEMS, ENDS, 5.0, "en-ko", "/x/a.wav")}
    assert rows["/x/a.wav#end=4.500"]["text"] == ""  # word 9 ends 4.4, heard from 4.6
    r = rows["/x/a.wav#end=4.600"]
    assert r["text"] == " 새 사무실 예산을" and r["cond"] == TAG + "매니저가"


def test_holdout_keeps_sid_together():
    sids = [f"s{i}" for i in range(500)]
    h = b.split_holdout(sids, 0.02, 14)
    assert len(h) == 10 and h <= set(sids)
    assert h == b.split_holdout(sids * 3, 0.02, 14)  # duplicate recs of a sid do not change the sid set


def test_two_recs_different_cut_times():
    toks = "a b c d".split()
    wa = [{"w": t, "end": 0.5 * (i + 1)} for i, t in enumerate(toks)]
    wb = [{"w": t, "end": 0.9 * (i + 1)} for i, t in enumerate(toks)]
    items = [(2, "X"), (4, "Y")]
    ra = b.rows_on_time(items, b.end_times_for(toks, wa), 2.0, "en-ko", "/x/a.wav")
    rb = b.rows_on_time(items, b.end_times_for(toks, wb), 3.6, "en-ko", "/x/b.wav")
    ta = [r["audio"].split("#end=")[1] for r in ra if r["text"]]
    tb = [r["audio"].split("#end=")[1] for r in rb if r["text"]]
    assert ta == ["1.200"] and tb == ["2.000"]


def test_end_and_silence_rows():
    rows = b.rows_end(ITEMS, "en-ko", "/x/a.wav", 5.0, ENDS)
    assert rows[0]["audio"] == "/x/a.wav#end=5.000"
    assert rows[0]["cond"] == TAG + "매니저가 새 사무실 예산을" and rows[0]["text"] == " 어제 승인했어요"
    assert rows[1]["audio"] == "/x/a.wav" and rows[1]["cond"] == TAG
    assert rows[1]["text"] == "매니저가 새 사무실 예산을 어제 승인했어요"
    s = b.rows_silence(ITEMS, "en-ko", "/x/a.wav", "/x/sil.wav", random.Random(1), 5.0, catchup=False)
    assert "|/x/sil.wav#end=" in s["audio"] and s["text"] == ""
    c = b.rows_silence(ITEMS, "en-ko", "/x/a.wav", "/x/sil.wav", random.Random(1), 5.0, catchup=True)
    assert c["text"] != ""


def test_seg_pairs_skip_same_sid():
    A = [{"sid": "s1", "audio": "/x/a.wav", "full": "F"}, {"sid": "s2", "audio": "/x/c.wav", "full": "G"}]
    B = [{"sid": "s1", "audio": "/x/b.wav", "first_end": 1.0}, {"sid": "s3", "audio": "/x/d.wav", "first_end": 1.0}]
    rows = b.rows_seg_pairs(A, B, "en-ko")
    assert len(rows) == 1 and rows[0]["audio"].startswith("/x/c.wav|/x/d.wav")


def test_on_time_stops_before_dur():
    rows = b.rows_on_time(ITEMS, ENDS, 5.0, "en-ko", "/x/a.wav")
    assert rows and all(float(r["audio"].split("#end=")[1]) < 5.0 for r in rows)
    end_a = b.rows_end(ITEMS, "en-ko", "/x/a.wav", 5.0, ENDS)[0]
    assert end_a["audio"] == "/x/a.wav#end=5.000"


def test_rows_end_whole_once():
    assert len(b.rows_end(ITEMS, "en-ko", "/x/a.wav", 5.0, ENDS)) == 2
    only = b.rows_end(ITEMS, "en-ko", "/x/b.wav", 5.0, ENDS, whole=False)
    assert len(only) == 1 and only[0]["audio"] == "/x/b.wav#end=5.000"


def test_cap_reserves_pre_emit_wait_rows():
    # 12.5 s recording, 15 words, one item per word: every grid cut but a few is an emission
    ends = [0.8 * (i + 1) for i in range(15)]
    items = [(i + 1, f"w{i + 1}") for i in range(15)]
    full = b.rows_on_time(items, ends, 12.5, "en-ko", "/x/a.wav")
    cap = 5
    rows = b.rows_on_time(items, ends, 12.5, "en-ko", "/x/a.wav", cap, random.Random(3))
    assert len(rows) == cap
    empty = [r for r in rows if not r["text"]]
    assert len(empty) >= cap / 3
    # every kept wait row is the cut right before an emission in the uncapped list
    idx = {r["audio"]: i for i, r in enumerate(full)}
    for r in empty:
        i = idx[r["audio"]]
        assert i + 1 < len(full) and full[i + 1]["text"]


def _e2e_inputs(tmp_path):
    import json

    import numpy as np
    import soundfile as sf
    sr = 16000
    sil = tmp_path / "sil.wav"
    sf.write(sil, np.zeros(sr * 4, dtype="float32"), sr)
    long_wav = tmp_path / "long.wav"  # clipped recordings point into this one
    sf.write(long_wav, np.zeros(sr * 60, dtype="float32"), sr)
    sents, wts, scheds = [], [], []
    for i in range(30):
        lang, tgt = ("en", "ko") if i % 2 == 0 else ("ko", "en")
        toks = [f"w{i}x{j}" for j in range(6)]
        recs = []
        for k in range(2):
            if i % 3 == 0 and k == 1:
                audio = f"{long_wav}#start={i:.3f}#end={i + 4.0:.3f}"
            else:
                p = tmp_path / f"s{i}_{k}.wav"
                sf.write(p, np.zeros(int(sr * 4.0), dtype="float32"), sr)
                audio = str(p)
            recs.append({"audio": audio})
            step = 0.5 + 0.05 * k
            wts.append({"audio": audio, "words": [{"w": t, "end": round(0.3 + step * (j + 1), 3)} for j, t in enumerate(toks)]})
        sents.append({"sid": f"s{i}", "lang": lang, "tgt": tgt, "src": " ".join(toks), "ref": None, "source": "t", "recs": recs})
        scheds.append({"sid": f"s{i}", "schedule": [[1, ""], [2, f"A{i}"], [4, f"B{i}"], [6, f"C{i}."]]})
    for name, rows in (("sentences.jsonl", sents), ("wt.jsonl", wts), ("sched.jsonl", scheds)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return sil


def _run_main(tmp_path, monkeypatch, *extra, make=True, range_check=False):
    sil = _e2e_inputs(tmp_path) if make else tmp_path / "sil.wav"
    argv = ["build_prefix14.py", "--schedule", str(tmp_path / "sched.jsonl"), "--sentences", str(tmp_path / "sentences.jsonl"),
            "--word-times", str(tmp_path / "wt.jsonl"), "--sil-wav", str(sil), "--out", str(tmp_path / "train.jsonl"),
            "--val-out", str(tmp_path / "val.jsonl"), "--holdout", "0.1", "--sil-cap", "5", "--seg-cap", "5",
            "--reset-rows", "10", *([] if range_check else ["--no-range-check"]), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    b.main()


def test_main_end_to_end(tmp_path, monkeypatch):
    import json
    import re
    _run_main(tmp_path, monkeypatch)
    train = [json.loads(l) for l in open(tmp_path / "train.jsonl", encoding="utf-8")]
    val = [json.loads(l) for l in open(tmp_path / "val.jsonl", encoding="utf-8")]
    assert train and val and not (tmp_path / "train.jsonl.tmp").exists()
    sents = [json.loads(l) for l in open(tmp_path / "sentences.jsonl", encoding="utf-8")]
    held = b.split_holdout([s["sid"] for s in sents], 0.1, 14)
    assert held

    plain = {r["audio"] for s in sents for r in s["recs"] if "#" not in r["audio"]}

    def rec_key(audio):  # plain file -> path (reset tails cut it with #start); clip -> path#start=S
        path, _, params = audio.partition("#")
        st = re.search(r"start=([0-9.]+)", params)
        return f"{path}#start={float(st.group(1)):.3f}" if st and path not in plain else path
    owner = {rec_key(r["audio"]): s["sid"] for s in sents for r in s["recs"]}
    sil = str(tmp_path / "sil.wav")
    piece = re.compile(r"^[^#|]+(#end=[0-9.]+|#start=[0-9.]+#end=[0-9.]+)?$")
    seen = {}
    for name, rows in (("train", train), ("val", val)):
        for r in rows:
            for p in r["audio"].split("|"):
                assert piece.match(p), p
                if p.partition("#")[0] == sil:
                    continue
                sid = owner[rec_key(p)]
                if name == "train":
                    assert sid not in held, (sid, r["audio"])
            assert r["cond"].startswith(b.tag_of(r["pair"]))
            key = (r["audio"], r["cond"])
            assert seen.setdefault(key, r["text"]) == r["text"], key
    assert {owner[rec_key(r["audio"].split("|")[-1])] for r in val} <= held
    # the whole-sentence row (uncut recording, cond = tag only) appears once per sentence
    whole = {}
    for r in train + val:
        if r["cond"] == b.tag_of(r["pair"]) and r["audio"] in {x["audio"] for s in sents for x in s["recs"]}:
            sid = owner[rec_key(r["audio"])]
            whole[sid] = whole.get(sid, 0) + 1
    assert whole and all(n == 1 for n in whole.values())


def test_main_drops_missing_audio(tmp_path, monkeypatch, capsys):
    import pytest
    _e2e_inputs(tmp_path)
    (tmp_path / "s4_0.wav").unlink()  # 1 of 60 recordings gone: 1.7% > 0.5% -> fail, nothing written
    with pytest.raises(SystemExit) as e:
        _run_main(tmp_path, monkeypatch, make=False)
    assert e.value.code == 1 and "BUILD_PREFIX14_FAIL missing audio" in capsys.readouterr().out
    assert not (tmp_path / "train.jsonl").exists() and not (tmp_path / "val.jsonl").exists()


def test_main_range_check(tmp_path, monkeypatch, capsys):
    import pytest
    with pytest.raises(SystemExit) as e:
        _run_main(tmp_path, monkeypatch, range_check=True)  # a few hundred rows, far below 60000
    assert e.value.code == 1 and "outside [60000, 200000]" in capsys.readouterr().out
    assert not (tmp_path / "train.jsonl").exists()


def test_main_val_cap(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch, "--val-cap", "7")
    assert sum(1 for _ in open(tmp_path / "val.jsonl", encoding="utf-8")) == 7


def test_drop_missing_threshold():
    rows = [{"audio": f"/ok/{i}.wav#end=1.0"} for i in range(999)] + [{"audio": "/ok/a.wav|/gone/b.wav#end=1"}]
    keep, miss = b.drop_missing(rows, exists=lambda p: not p.startswith("/gone"))
    assert len(keep) == 999 and miss == ["/gone/b.wav"]
    assert b.audio_paths("/a.wav#start=1#end=2|/b.wav#end=3") == ["/a.wav", "/b.wav"]


def test_delta_glues_particle_piece():
    assert b._delta([(4, "응답률")], [(6, "도")]) == "도"
    assert b._delta([(4, "응답률도")], [(7, "더 높습니다.")]) == " 더 높습니다."
