from collections import deque

import numpy as np

from core.utils import audio as audio_mod

from . import enhancers
from .base import Enhancer, Routed

FRAME = 512
HOP = 256
ROUTES = {"raw", "enhanced", "mix"}
NOISE_WINDOW_FRAMES = 96
MIN_STATISTICS_BIAS = 2.0


def _route(value) -> str:
    if value not in ROUTES:
        raise ValueError(f"expected one of {sorted(ROUTES)}, got {value!r}")
    return value


def _weight(value) -> float:
    weight = float(value)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"mix_weight must be in [0, 1], got {weight}")
    return weight


@enhancers.register("spectral")
class SpectralGate(Enhancer):
    SETTINGS = {
        "vad_input": ("vad_input", _route),
        "asr_input": ("asr_input", _route),
        "mix_weight": ("mix_weight", _weight),
        "over_subtraction": ("over_subtraction", float),
        "gain_floor": ("gain_floor", float),
        "noise_smoothing": ("noise_smoothing", float),
    }

    def start(self, **_) -> None:
        self.window = np.sqrt(np.hanning(FRAME + 1)[:-1]).astype(np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.overlap = np.zeros(FRAME - HOP, dtype=np.float32)
        self.ready = np.zeros(FRAME, dtype=np.float32)
        self.delayed = np.zeros(FRAME, dtype=np.float32)
        self.smoothed: np.ndarray | None = None
        self.history: deque = deque(maxlen=NOISE_WINDOW_FRAMES)

    def enhance(self, pcm: bytes) -> Routed:
        raw = audio_mod.from_pcm_bytes(pcm)
        clean = self._process(raw)
        raw_aligned = self._delay(raw)
        vad_route = self.settings.get("vad_input", "enhanced")
        asr_route = self.settings.get("asr_input", "raw")
        return Routed(vad=self._pick(vad_route, raw_aligned, clean),
                      asr=self._pick(asr_route, raw_aligned, clean))

    def _pick(self, route: str, raw: np.ndarray, clean: np.ndarray) -> bytes:
        if route == "raw":
            return audio_mod.to_pcm_bytes(raw)
        if route == "enhanced":
            return audio_mod.to_pcm_bytes(clean)
        weight = self.settings.get("mix_weight", 0.2)
        return audio_mod.to_pcm_bytes(weight * raw + (1.0 - weight) * clean)

    def _delay(self, raw: np.ndarray) -> np.ndarray:
        joined = np.concatenate([self.delayed, raw])
        self.delayed = joined[len(raw):]
        return joined[:len(raw)]

    def _process(self, raw: np.ndarray) -> np.ndarray:
        signal = np.concatenate([self.pending, raw])
        produced, start = [], 0
        while start + FRAME <= len(signal):
            frame = self._clean_frame(signal[start:start + FRAME])
            frame[:FRAME - HOP] += self.overlap
            produced.append(frame[:HOP])
            self.overlap = frame[HOP:].copy()
            start += HOP
        self.pending = signal[start:]
        self.ready = np.concatenate([self.ready, *produced])
        out, self.ready = self.ready[:len(raw)], self.ready[len(raw):]
        return out

    def _clean_frame(self, frame: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(frame * self.window)
        power = np.abs(spectrum) ** 2
        noise = self._track_noise(power)
        over = self.settings.get("over_subtraction", 2.0)
        floor = self.settings.get("gain_floor", 0.1)
        gain = np.maximum(1.0 - over * noise / np.maximum(power, 1e-12), floor ** 2)
        cleaned = np.fft.irfft(spectrum * np.sqrt(gain), n=FRAME).astype(np.float32)
        return cleaned * self.window

    def _track_noise(self, power: np.ndarray) -> np.ndarray:
        smoothing = self.settings.get("noise_smoothing", 0.8)
        self.smoothed = power if self.smoothed is None else \
            smoothing * self.smoothed + (1.0 - smoothing) * power
        self.history.append(self.smoothed)
        return MIN_STATISTICS_BIAS * np.min(np.stack(self.history), axis=0)
