import numpy as np

from core.errors import ConfigError
from core.utils import audio as audio_mod
from core.utils import logging

from . import enhancers
from .base import Enhancer

log = logging.getLogger(__name__)

STFT_SIZE = 512
STFT_SHIFT = 128
MIN_BLOCK_SAMPLES = STFT_SIZE * 2
MIN_RMS = 1e-6


@enhancers.register("wpe")
class WpeEnhancer(Enhancer):

    SETTINGS = {
        "dereverb": ("dereverb", bool),
        "taps": ("taps", int),
        "delay": ("delay", int),
        "iterations": ("iterations", int),
        "loudness": ("loudness", bool),
        "loudness_target_dbfs": ("loudness_target_dbfs", float),
        "loudness_smoothing": ("loudness_smoothing", float),
    }

    def start(self, **_) -> None:
        self._gain_db = 0.0

    async def enhance(self, audio: bytes) -> bytes:
        samples = audio_mod.from_pcm_bytes(audio)
        if samples.size == 0:
            return audio
        if self.settings.get("dereverb", True):
            samples = self._dereverb(samples)
        if self.settings.get("loudness", False):
            samples = self._normalize_loudness(samples)
        return audio_mod.to_pcm_bytes(samples)

    def _dereverb(self, samples: np.ndarray) -> np.ndarray:
        if samples.shape[0] < MIN_BLOCK_SAMPLES:
            return samples
        try:
            from nara_wpe.utils import istft, stft
            from nara_wpe.wpe import wpe
        except ImportError as e:
            raise ConfigError(
                "enhancement 'wpe' needs the nara_wpe package "
                "(pip install nara_wpe) -- or set dereverb: false to run without it."
            ) from e
        taps = int(self.settings.get("taps", 10))
        delay = int(self.settings.get("delay", 3))
        iterations = int(self.settings.get("iterations", 3))
        y = samples[np.newaxis, :].astype(np.float64)
        spectrum = stft(y, size=STFT_SIZE, shift=STFT_SHIFT).transpose(2, 0, 1)
        dereverbed = wpe(spectrum, taps=taps, delay=delay, iterations=iterations)
        time_domain = istft(dereverbed.transpose(1, 2, 0), size=STFT_SIZE, shift=STFT_SHIFT)
        out = time_domain[0]
        if out.shape[0] < samples.shape[0]:
            out = np.pad(out, (0, samples.shape[0] - out.shape[0]))
        else:
            out = out[:samples.shape[0]]
        return out.astype(np.float32)

    def _normalize_loudness(self, samples: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms < MIN_RMS:
            return samples
        current_dbfs = 20.0 * np.log10(rms)
        target_dbfs = float(self.settings.get("loudness_target_dbfs", -26.0))
        smoothing = min(max(float(self.settings.get("loudness_smoothing", 0.1)), 0.0), 1.0)
        target_gain_db = target_dbfs - current_dbfs
        self._gain_db += (target_gain_db - self._gain_db) * smoothing
        gain = 10.0 ** (self._gain_db / 20.0)
        return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)
