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
