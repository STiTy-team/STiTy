import numpy as np

from core.components.langid.audio import vad
from core.components.transcription.qwen3_seg import pause_near

SR = 16000


def fake_spans(monkeypatch, spans_sec):
    monkeypatch.setattr(
        vad, "speech_spans", lambda a: [(int(s * SR), int(e * SR)) for s, e in spans_sec]
    )


def test_cut_falls_in_the_pause_nearest_the_switch(monkeypatch):
    # Korean until 1.1 s, silence, English from 3.0 s; the langid puts the switch at 3.3 s.
    fake_spans(monkeypatch, [(0.0, 1.1), (3.0, 4.5)])
    assert abs(pause_near(np.zeros(int(4.5 * SR)), 3.3) - 2.05) < 1e-6


def test_no_pause_near_the_switch_keeps_the_switch_time(monkeypatch):
    fake_spans(monkeypatch, [(0.0, 4.5)])
    assert pause_near(np.zeros(int(4.5 * SR)), 2.0) == 2.0


def test_pause_too_far_back_is_ignored(monkeypatch):
    fake_spans(monkeypatch, [(0.0, 0.5), (1.0, 6.0)])
    assert pause_near(np.zeros(int(6.0 * SR)), 5.0) == 5.0
