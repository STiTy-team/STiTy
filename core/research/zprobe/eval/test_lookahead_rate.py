import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookahead_rate import collect, heard_words, judge, segment_at  # noqa: E402

AL = [{"word": "a", "start": 0.0, "end": 0.5}, {"word": "b", "start": 0.5, "end": 1.0},
      {"word": "c", "start": 1.0, "end": 1.5}]
SEG = {"offset": 0.0, "duration": 2.0, "src_lang": "en"}


def test_heard():
    assert heard_words(AL, 1.0, SEG) == ["a", "b"]
    assert heard_words(AL, 0.4, SEG) == []


def test_heard_other_segment_excluded():
    assert heard_words(AL, 9.0, {"offset": 1.2, "duration": 2.0}) == []


def test_judge():
    assert judge({"내일"}, {"보고"}, {"보고", "내일"}) is True
    assert judge({"보고"}, {"보고"}, {"보고", "내일"}) is False


def test_segment_at():
    assert segment_at([SEG], 1.0) == 0
    assert segment_at([SEG], 2.5) is None


def test_unjudged():
    segs = [SEG, {"offset": 2.0, "duration": 2.0, "src_lang": "ko"}]
    rec = lambda t: {"type": "translated", "target_lang": "ko", "translation": "x", "decision_audio_sec": t}
    (_, rows), = collect([{"reference_segmentation": segs, "records": [rec(1.0), rec(2.5), rec(5.0)]}], "ko")
    assert [j for _, _, j in rows] == [True, False, False]


def test_grace_window():
    segs = [SEG, {"offset": 3.0, "duration": 1.0, "src_lang": "ko"}]
    assert segment_at(segs, 2.5, 0.0) is None
    assert segment_at(segs, 2.5, 2.0) == 0
    assert segment_at(segs, 3.0, 2.0) == 1  # capped by next segment offset
    assert segment_at(segs, 4.5, 2.0) == 1
    assert segment_at(segs, 7.0, 2.0) is None


def test_run_dir_end_to_end(tmp_path):
    import json
    from lookahead_rate import run_dir
    segs = [{"offset": 0.0, "duration": 2.0, "src_lang": "en", "transcript": "a b c"},
            {"offset": 4.0, "duration": 2.0, "src_lang": "ko", "transcript": "x"}]
    rec = lambda t: {"type": "translated", "target_lang": "ko", "translation": "a", "decision_audio_sec": t}
    item = {"id": "s1", "reference_segmentation": segs, "reference_alignment": AL, "records": [rec(1.0), rec(3.0), rec(5.0)]}
    (tmp_path / "items.jsonl").write_text(json.dumps(item) + "\n")
    res = run_dir(str(tmp_path), "ko", no_hy=True, grace=2.0)
    assert (res["pieces"], res["judged"], res["unjudged"]) == (3, 2, 1)
    saved = json.loads((tmp_path / "lookahead.json").read_text())
    assert (saved["pieces"], saved["judged"], saved["unjudged"], saved["grace"]) == (3, 2, 1, 2.0)


def test_heard_uses_pad():
    from lookahead_rate import PAD
    assert PAD == 0.2
    assert heard_words(AL, 1.0, SEG, PAD) == ["a"]  # b ends at 1.0, heard from 1.2
    assert heard_words(AL, 1.2, SEG, PAD) == ["a", "b"]


def test_known_is_cumulative_over_prefixes(tmp_path, monkeypatch):
    # prefix "a" translates to "manager", prefix "a b" to "boss approved": manag stays known at 2 words
    import json
    import types
    from lookahead_rate import run_dir
    tr = {"a": "manager", "a b": "boss approved", "a b c": "the manager approved the budget"}
    fake = types.SimpleNamespace(load_model=lambda: (None, None, None, None),
                                 translate=lambda reqs, *a: [tr[t] for t, _, _ in reqs])
    monkeypatch.setitem(sys.modules, "hy_word_prefixes", fake)
    segs = [{"offset": 0.0, "duration": 2.0, "src_lang": "ko", "transcript": "a b c"}]
    rec = {"type": "translated", "target_lang": "en", "translation": "manager approved", "decision_audio_sec": 1.25}
    item = {"id": "s1", "reference_segmentation": segs, "reference_alignment": AL, "records": [rec]}
    (tmp_path / "items.jsonl").write_text(json.dumps(item) + "\n")
    res = run_dir(str(tmp_path), "en", grace=0.0)
    assert res["judged"] == 1 and res["violations"] == 0
