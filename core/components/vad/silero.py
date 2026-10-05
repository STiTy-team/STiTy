import numpy as np

from core.errors import ConfigError
from core.utils import audio as audio_mod
from core.utils import cache
from core.utils import logging

from . import detectors
from .base import Detector, Speech

log = logging.getLogger(__name__)

SILERO_WINDOW_SAMPLES_AT_16K = 512


class _ProbTap:
    """Passes calls to the silero model and keeps the last speech probability.

    The iterator stays "in speech" until the probability falls below threshold - 0.15
    and only reports the end after min_silence_ms, so neither says whether this window
    itself sounded like speech.
    """

    def __init__(self, model):
        self.model = model
        self.prob = 0.0

    def __call__(self, x, sampling_rate):
        out = self.model(x, sampling_rate)
        self.prob = out.item()
        return out

    def __getattr__(self, name):
        return getattr(self.model, name)


@detectors.register('silero')
class SileroVad(Detector):
    SETTINGS = {
        "min_silence_ms": ("min_silence_ms", int),
        "threshold": ("threshold", float),
        "speech_pad_ms": ("speech_pad_ms", int),
    }
    model_bytes = None

    async def load(self) -> None:
        self.model_bytes = await cache.load("silero-vad", self._build_weights)
        log.info("[LOAD] silero VAD min_silence=%dms threshold=%.2f",
                 self.settings.get("min_silence_ms", 800),
                 self.settings.get("threshold", 0.5))

    async def _build_weights(self) -> bytes:
        import io

        import torch

        try:
            from silero_vad import load_silero_vad
        except ImportError as e:
            raise ConfigError(
                "vad 'silero' needs the silero-vad package (pip install silero-vad). "
                "Use `vad: none` to run without detection -- but say so in the "
                "config rather than letting a missing package decide the policy."
            ) from e

        buffer = io.BytesIO()
        torch.jit.save(load_silero_vad(), buffer)
        return buffer.getvalue()

    def start(self, **_) -> None:
        import io

        import torch
        from silero_vad import VADIterator

        super().start(**_)

        self.iterator = VADIterator(
            model=_ProbTap(torch.jit.load(io.BytesIO(self.model_bytes))),
            threshold=float(self.settings.get("threshold", 0.5)),
            sampling_rate=audio_mod.SAMPLING_RATE,
            min_silence_duration_ms=int(self.settings.get("min_silence_ms", 800)),
            speech_pad_ms=int(self.settings.get("speech_pad_ms", 160)),
        )
        self.buffer = np.empty(0, dtype=np.float32)
        self.disabled = False
        self.consumed = 0
        self.speech_started = 0.0
        self.voiced_until = 0.0

    def detect(self, audio: bytes) -> Speech | None:
        if self.disabled:
            return None
        import torch

        self.buffer = np.concatenate([self.buffer, audio_mod.from_pcm_bytes(audio)])
        ended = None
        try:
            offset = 0
            while offset + SILERO_WINDOW_SAMPLES_AT_16K <= self.buffer.size:
                window = self.buffer[offset:offset + SILERO_WINDOW_SAMPLES_AT_16K]
                mark = self.iterator(torch.from_numpy(window), return_seconds=False)
                if mark is not None:
                    at = ((self.consumed + offset + SILERO_WINDOW_SAMPLES_AT_16K)
                          / audio_mod.SAMPLING_RATE)
                    # `at` is when the iterator decided; the mark holds where speech
                    # was (padded). The end is decided only after min_silence_ms of
                    # quiet, so `at` would stretch the span over that silent tail and
                    # the transcriber's no-speech check would see speech there.
                    if "start" in mark:
                        self.speech_started = max(0.0, at)
                        self.spans.append([mark["start"] / audio_mod.SAMPLING_RATE, None])
                    if "end" in mark:
                        silence = float(self.settings.get("min_silence_ms", 800)) / 1000
                        ended = Speech(started_at=self.speech_started,
                                       ended_at=max(0.0, at), silence_waited_out_sec=silence)
                        if self.spans and self.spans[-1][1] is None:
                            self.spans[-1][1] = mark["end"] / audio_mod.SAMPLING_RATE
                if self.iterator.model.prob >= self.iterator.threshold:
                    self.voiced_until = ((self.consumed + offset + SILERO_WINDOW_SAMPLES_AT_16K)
                                         / audio_mod.SAMPLING_RATE)
                offset += SILERO_WINDOW_SAMPLES_AT_16K
            self.consumed += offset
            self.buffer = self.buffer[offset:]
        except Exception as e:  # noqa: BLE001
            log.warning("[VAD-ERROR] disabled for this item: %s", e)
            self.disabled = True
            self.spans = []
            self.voiced_until = None
        return ended

