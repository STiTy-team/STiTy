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
