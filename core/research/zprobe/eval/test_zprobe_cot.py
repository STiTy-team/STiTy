"""CPU tests for core/components/transcription/zprobe_cot.py (no model, no GPU).

Lives here, not next to the component: transcription/__init__ imports every module in its package."""
import numpy as np

from core.components.registry import Partial, Transcribed, Translated
from core.components.transcription import transcribers
from core.components.transcription.zprobe_cot import (
    HEAD,
    ZprobeCotTranscription,
    parse_steps,
    source_lang,
    step_line,
)


def test_registered():
    assert transcribers.get("zprobe-cot") is ZprobeCotTranscription


def test_parse_normal():
    assert parse_steps("by John Smith<asr_text>존 스미스가\n", final=False) == [("by John Smith", "존 스미스가")]


def test_parse_empty():
    assert parse_steps("", final=False) == []
    assert parse_steps("", final=True) == []
    assert parse_steps("\n", final=False) == []


def test_parse_no_newline_mid_stream_is_no_output():
    assert parse_steps("by John<asr_text>존", final=False) == []


def test_parse_mid_stream_takes_first_line_only():
    assert parse_steps("a b<asr_text>가\nc d<asr_text>나\n", final=False) == [("a b", "가")]


def test_parse_final_with_tail():
    assert parse_steps("a<asr_text>가\nb c<asr_text>나", final=True) == [("a", "가"), ("b c", "나")]


def test_parse_missing_separator_is_source():
    assert parse_steps("in the 1970s\n", final=False) == [("in the 1970s", "")]


def test_parse_several_lines_final():
    text = "It was<asr_text>그것은\n developed <asr_text> 개발되었다.\nyes<asr_text>\n"
    assert parse_steps(text, final=True) == [("It was", "그것은"), ("developed", "개발되었다."), ("yes", "")]


def test_parse_stray_special():
    assert parse_steps("a<asr_text>가\n<|im_end|>", final=True) == [("a", "가")]
    assert parse_steps("a<asr_text>가<|im_end|>", final=True) == [("a", "가")]
    assert parse_steps("<|endoftext|>", final=False) == []


def test_parse_merged_dot_newline():
    # BPE merges ".\n" into one token; parsing on the decoded string still splits there
    assert parse_steps("개발하였습니다.\n by", final=False) == [("개발하였습니다.", "")]


def test_round_trip():
    steps = [("It was developed", "그것은"), ("by John Smith", "존 스미스가")]
    text = "".join(step_line(s, t) for s, t in steps)
    assert parse_steps(text, final=True) == steps


def test_source_lang():
    assert source_lang("그것은 개발되었다") == "ko"
    assert source_lang("It was developed") == "en"
    assert source_lang("", "그것은") == "en"
    assert source_lang("", "It") == "ko"


def _component():
    c = ZprobeCotTranscription.__new__(ZprobeCotTranscription)
    c.settings = {}
    c._np = np
    c._vad = None
    c._fed_samples = 16000
    c._seq = 0
    c._reset_utterance()
    return c


def test_emit_history_and_records():
    c = _component()
    out = c._emit("It was developed<asr_text>그것은\n", final=False)
    assert c._hist == "It was developed<asr_text>그것은\n"
    assert isinstance(out[0], Partial) and out[0].text == "It was developed" and out[0].language == "en"
    assert isinstance(out[1], Translated) and out[1].translation == "그것은" and out[1].target_lang == "ko"
    assert out[1].original == "It was developed" and out[1].commit_reason == "cot"
    assert c._emit("", final=False) == [] and c._hist.count("\n") == 1  # wait step: history unchanged
    out = c._emit(" by John <asr_text>\n", final=False)
    assert [type(r) for r in out] == [Partial] and out[0].text == "It was developed by John"
    assert c._hist == "It was developed<asr_text>그것은\nby John<asr_text>\n"
    closed = c._close("vad")
    assert isinstance(closed[0], Transcribed) and closed[0].original == "It was developed by John"
    assert closed[0].language == "en" and closed[0].commit_reason == "vad"


def test_emit_korean_source():
    c = _component()
    out = c._emit("오늘 날씨가<asr_text>The weather today\n", final=False)
    tr = [r for r in out if isinstance(r, Translated)][0]
    assert tr.language == "ko" and tr.target_lang == "en"


def test_head_constant():
    assert HEAD == "language Interpret<asr_text>"
