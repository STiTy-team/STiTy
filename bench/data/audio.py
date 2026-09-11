"""Audio loading. The only place that knows a dataset's on-disk encoding.

Window slicing (offset/duration) reads just the requested frames, so a corpus of
concatenated clips never needs a second copy on disk. The semantics match
evaluation/ast/test_ast.py's load_segment_audio and evaluation/harness/audio.py.
"""
from pathlib import Path

import numpy as np

from ..errors import BenchDataError

SAMPLING_RATE = 16000


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return audio


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def probe_duration(path: Path, fmt: str, sample_rate: int = SAMPLING_RATE) -> float:
    if fmt == "pcm_s16le":
        return path.stat().st_size / 2 / sample_rate
    import soundfile as sf
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def load_window(path: Path, fmt: str, *, offset: float | None = None,
                duration: float | None = None,
                sample_rate: int = SAMPLING_RATE) -> np.ndarray:
    """Returns float32 mono at `sample_rate`, sliced to [offset, offset+duration)."""
    if not path.is_file():
        raise BenchDataError(f"audio missing: {path}")

    if fmt == "pcm_s16le":
        if offset is None and duration is None:
            raw = np.fromfile(str(path), dtype=np.int16)
        else:
            start_frame = int(round((offset or 0.0) * sample_rate))
            count = int(round(duration * sample_rate)) if duration is not None else -1
            raw = np.fromfile(str(path), dtype=np.int16,
                              count=count, offset=start_frame * 2)
        return raw.astype(np.float32) / 32767.0

    import soundfile as sf
    if offset is None and duration is None:
        audio, sr = sf.read(str(path), dtype="float32")
    else:
        info = sf.info(str(path))
        sr = info.samplerate
        start = int(round((offset or 0.0) * sr))
        frames = int(round(duration * sr)) if duration is not None else -1
        audio, sr = sf.read(str(path), dtype="float32", start=start, frames=frames)
    audio = _to_mono(audio)
    return _resample(audio, sr, sample_rate)


def to_pcm_bytes(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] -> s16le bytes, which is what the handler consumes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def silence_bytes(ms: int, sample_rate: int = SAMPLING_RATE) -> bytes:
    return np.zeros(int(sample_rate * ms / 1000), dtype=np.int16).tobytes()
