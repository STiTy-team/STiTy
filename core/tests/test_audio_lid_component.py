import asyncio

import numpy as np
import pytest

from core.components.langid.audio import component as comp
from core.components.langid.audio.component import AudioLanguageDetector
from core.components.registry import Transcribed
from core.utils import audio as audio_mod

SR = audio_mod.SAMPLING_RATE


class FakeScorer:
    def __init__(self, scores):
        self.value = scores
        self.calls = 0

    def scores(self, audio, candidates):
        self.calls += 1
        return {c: self.value.get(c, float("-inf")) for c in candidates}


class FakeSpan:
    def __init__(self, span):
        self.span = span

    def find(self, audio, start, text, candidates):
        return self.span


def detector(scores, span=None, **settings):
    d = AudioLanguageDetector({"aligner": True, **settings}, cfg=None)
    d.scorer = FakeScorer(scores)
    d.spanner = FakeSpan(span)
    d.start(languages=["ko", "en"])
    return d


def commit(text, at):
    return Transcribed(
        original=text,
        language="ko",
        commit_reason="seg",
        decision_audio_sec=at,
        committed_elapsed_sec=at,
    )


@pytest.fixture(autouse=True)
def all_speech(monkeypatch):
    monkeypatch.setattr(comp, "speech_only", lambda a: a)


def pcm(sec):
    return audio_mod.to_pcm_bytes(np.full(int(sec * SR), 0.1, dtype=np.float32))


def test_segment_decides_confident_language_and_logs_aligner_span():
    d = detector({"ko": 0.0, "en": -5.0}, span=(1.0, 2.5))
    d.hear(pcm(4.0))
    assert asyncio.run(d.detect(commit("hi", 4.0))) == "ko"
    assert d.trace[-1]["src"] == "aligner" and d.trace[-1]["span"] == (1.0, 2.5)


def test_segment_keeps_previous_below_conf():
    d = detector({"ko": -0.6, "en": -0.8}, span=(1.0, 2.5))
    d.previous = "en"
    d.hear(pcm(4.0))
    assert asyncio.run(d.detect(commit("hi", 4.0))) == "en"


def test_first_unsure_commit_returns_none():
    d = detector({"ko": -0.69, "en": -0.70}, span=(1.0, 2.5))
    d.hear(pcm(4.0))
    assert asyncio.run(d.detect(commit("hi", 4.0))) is None


def test_lag_span_when_aligner_fails():
    d = detector({"ko": 0.0, "en": -5.0}, span=None, lag_sec=0.5)
    d.hear(pcm(4.0))
    asyncio.run(d.detect(commit("hi", 4.0)))
    assert d.trace[-1]["src"] == "lag" and d.trace[-1]["span"] == (0.0, 3.5)


def test_silent_span_keeps_previous():
    d = detector({"ko": 0.0, "en": -5.0}, span=(1.0, 1.1))
    d.previous = "en"
    d.hear(pcm(4.0))
    assert asyncio.run(d.detect(commit("hi", 4.0))) == "en"


def test_timeline_smooths_over_a_weak_step():
    d = detector({}, span=(1.2, 1.4), mode="timeline", windows=(1.0,))
    strong_ko, weak_en = {"ko": 0.0, "en": -6.0}, {"ko": -1.0, "en": 0.0}
    d.steps = [
        (1.0, strong_ko),
        (1.5, strong_ko),
        (2.0, weak_en),
        (2.5, strong_ko),
        (3.0, strong_ko),
    ]
    d.audio = np.zeros(int(3.0 * SR), dtype=np.float32)
    assert asyncio.run(d.detect(commit("hi", 3.0))) == "ko"


def test_timeline_scores_steps_while_hearing():
    d = detector({"ko": 0.0, "en": -3.0}, span=(0.5, 1.5), mode="timeline", windows=(1.0,))
    d.hear(pcm(2.0))
    assert [t for t, _ in d.steps] == [0.5, 1.0, 1.5, 2.0]
    assert asyncio.run(d.detect(commit("hi", 2.0))) == "ko"


def test_timeline_span_before_buffer_uses_nearest_step():
    d = detector({"ko": 0.0, "en": -3.0}, span=(0.0, 0.1), mode="timeline", windows=(1.0,))
    d.hear(pcm(2.0))
    assert asyncio.run(d.detect(commit("hi", 2.0))) == "ko"


def test_temperature_sharpens_scores():
    d = detector({"ko": -0.6, "en": -0.8}, span=(1.0, 2.5), temperature=10.0, conf=0.7)
    d.hear(pcm(4.0))
    assert asyncio.run(d.detect(commit("hi", 4.0))) == "ko"
