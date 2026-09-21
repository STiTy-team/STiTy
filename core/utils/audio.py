"""WAV in, 16 kHz mono s16le out -- what the streaming handler eats.

WAV only: the header carries the sample rate and channel count, so nothing about a
file's encoding has to be declared alongside it and then kept true.

Window slicing (offset/duration) reads just the requested frames, so a corpus of
long recordings cut into segments never needs a second copy on disk.
"""
from pathlib import Path

import numpy as np

from core.errors import AudioError

SAMPLING_RATE = 16000


def probe_duration(path: Path) -> float:
    """Seconds, from the header. Does not decode."""
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def load_window(path: Path, *, offset: float | None = None,
                duration: float | None = None) -> np.ndarray:
    """float32 mono at SAMPLING_RATE, sliced to [offset, offset+duration).

    The mono-mixing and resampling below only ever fire when a corpus was converted
    to something other than 16 kHz mono. Nothing in the dataset contract promises
    that yet, so they cannot be dropped.
    """
    if not Path(path).is_file():
        raise AudioError(f"audio missing: {path}")

    import soundfile as sf

    if offset is None and duration is None:
        audio, sr = sf.read(str(path), dtype="float32")
    else:
        sr = sf.info(str(path)).samplerate
        start = int(round((offset or 0.0) * sr))
        frames = int(round(duration * sr)) if duration is not None else -1
        audio, sr = sf.read(str(path), dtype="float32", start=start, frames=frames)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLING_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    return audio


def to_pcm_bytes(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] -> s16le bytes."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def from_pcm_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def silence_bytes(ms: int) -> bytes:
    return np.zeros(int(SAMPLING_RATE * ms / 1000), dtype=np.int16).tobytes()
