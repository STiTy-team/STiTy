import asyncio

import numpy as np
import pytest

from core.components.langid.probe import ProbeLanguageDetector
from core.components.registry import Transcribed
from core.errors import ConfigError
from core.utils import audio as audio_mod

SR = audio_mod.SAMPLING_RATE


class FakeScorer:
    def __init__(self, verdict):
        self.verdict = verdict
        self.spans = []

    def judge(self, audio):
        self.spans.append(len(audio))
        return self.verdict


def detector(verdict, **settings):
    d = ProbeLanguageDetector({"head": "head.npz", **settings}, cfg=None)
    d.scorer = FakeScorer(verdict)
    d.start()
    return d


def committed(text, at_sec, language="ko"):
    return Transcribed(
        original=text,
        language=language,
        commit_reason="seg",
        decision_audio_sec=at_sec,
        committed_elapsed_sec=at_sec,
    )


def pcm(sec):
    return audio_mod.to_pcm_bytes(np.zeros(int(sec * SR), dtype=np.float32))


def test_confident_probe_wins_over_the_alphabet():
    d = detector(("en", 0.999))
    d.hear(pcm(3.0))
    assert asyncio.run(d.detect(committed("안녕하세요", 3.0))) == "en"


def test_unsure_probe_falls_back_to_the_alphabet():
    d = detector(("en", 0.9))
    d.hear(pcm(3.0))
    assert asyncio.run(d.detect(committed("안녕하세요", 3.0))) == "ko"


def test_unsure_probe_without_fallback_keeps_the_asr_tag():
    d = detector(("en", 0.9), fallback="none")
    d.hear(pcm(3.0))
    assert asyncio.run(d.detect(committed("안녕하세요", 3.0))) is None


def test_each_commit_judges_only_its_own_span():
    d = detector(("ko", 1.0))
    d.hear(pcm(5.0))
    asyncio.run(d.detect(committed("하나", 2.0)))
    asyncio.run(d.detect(committed("둘", 5.0)))
    assert d.scorer.spans == [2 * SR, 3 * SR]


def test_head_is_required():
    with pytest.raises(ConfigError):
        ProbeLanguageDetector.validate({}, kind="langid")


def test_lag_moves_the_span_back():
    d = detector(("ko", 1.0), lag_sec=0.5)
    d.hear(pcm(5.0))
    asyncio.run(d.detect(committed("하나", 2.0)))
    asyncio.run(d.detect(committed("둘", 5.0)))
    assert d.scorer.spans == [int(1.5 * SR), 3 * SR]
