"""Speech-only audio for language scoring (silero VAD)."""

import threading

import numpy as np

from core.utils import audio as audio_mod

SR = audio_mod.SAMPLING_RATE
_model = None
_lock = threading.Lock()


def speech_only(audio: np.ndarray) -> np.ndarray:
    """The speech parts of `audio`, concatenated (empty when there is none)."""
    global _model
    if len(audio) < SR // 10:
        return audio[:0]
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    with _lock:
        if _model is None:
            _model = load_silero_vad()
        spans = get_speech_timestamps(torch.from_numpy(audio), _model, sampling_rate=SR)
    if not spans:
        return audio[:0]
    return np.concatenate([audio[s["start"] : s["end"]] for s in spans])


def speech_spans(audio: np.ndarray) -> list[tuple[int, int]]:
    """Speech (start, end) sample ranges in `audio`."""
    global _model
    if len(audio) < SR // 10:
        return []
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    with _lock:
        if _model is None:
            _model = load_silero_vad()
        found = get_speech_timestamps(torch.from_numpy(audio), _model, sampling_rate=SR)
    return [(s["start"], s["end"]) for s in found]
