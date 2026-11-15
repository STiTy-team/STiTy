import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mix_p16c as M  # noqa: E402


def test_formal_rows_are_found_in_cond_or_text_of_korean_targets_only():
    assert M.is_formal({"pair": "en-ko", "cond": "language Korean<asr_text>", "text": "구매하고 싶습니다."})
    assert M.is_formal({"pair": "en-ko", "cond": "language Korean<asr_text>네, 알겠습니다. ", "text": "그럼 가요."})
    assert not M.is_formal({"pair": "en-ko", "cond": "language Korean<asr_text>", "text": "정말요? 얼마나 됐나요?"})
    assert not M.is_formal({"pair": "ko-en", "cond": "language English<asr_text>", "text": "Okay."})


def test_cap_empty_keeps_every_text_row_and_limits_empties():
    rows = [{"text": ""} for _ in range(60)] + [{"text": "x"} for _ in range(40)]
    out = M.cap_empty(rows, 0.25, random.Random(0))
    assert sum(1 for r in out if r["text"]) == 40 and sum(1 for r in out if not r["text"]) <= 40 * 0.25 / 0.75 + 1


def test_remap_silence_path():
    r = {"audio": "/x/a.wav#end=2.0|/home/skkai/STiTy-wt/mtpl/models/zprobe/data/silence_tail_4s.wav#end=1.2"}
    out = M.remap(r, "/local/sil.wav")
    assert out["audio"] == "/x/a.wav#end=2.0|/local/sil.wav#end=1.2"


def test_pick_carried_caps_per_pair_and_source():
    rows = [{"pair": "en-ko", "audio": f"/d/covost2_train_en/{i}.wav"} for i in range(10)] + \
           [{"pair": "en-ko", "audio": "/d/fleurs/a.wav"}] + [{"pair": "zh-en", "audio": "/d/fleurs/z.wav"}] * 5 + \
           [{"pair": "en-ja", "audio": "/d/fleurs/j.wav"}]
    out = M.pick_carried(rows, {"en-ko/covost": 4, "x-en": 2}, random.Random(0))
    pairs = [r["pair"] for r in out]
    assert pairs.count("en-ko") == 5 and pairs.count("zh-en") == 2 and "en-ja" not in pairs


def test_end_dump_rows_are_full_audio_emissions_only():
    assert M.is_end_dump({"audio": "/a.wav", "text": "통째 번역이에요."})
    assert not M.is_end_dump({"audio": "/a.wav#end=2.0", "text": "앞부분을"})
    assert not M.is_end_dump({"audio": "/a.wav#end=3.0|/sil.wav#end=1.2", "text": "나머지예요."})  # silence tail catch-up stays
    assert not M.is_end_dump({"audio": "/a.wav", "text": ""})
