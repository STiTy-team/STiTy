import numpy as np

from core.errors import ConfigError
from core.utils import audio as audio_mod
from core.utils import logging

from . import detectors
from .base import Detector, Speech

logger = logging.getLogger("bench")

SILERO_WINDOW_SAMPLES_AT_16K = 512


@detectors.register('silero')
class SileroVad(Detector):
    SETTINGS = {
        "min_silence_ms": ("min_silence_ms", int),
        "threshold": ("threshold", float),
        "speech_pad_ms": ("speech_pad_ms", int),
    }
    model_bytes = None

    async def load(self) -> None:
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
        self.model_bytes = buffer.getvalue()
        logger.info("silero VAD loaded (min_silence=%dms threshold=%.2f)",
                    self.settings.get("min_silence_ms", 800),
                    self.settings.get("threshold", 0.5))

    def start(self, **_) -> None:
        import io

        import torch
        from silero_vad import VADIterator

        self.iterator = VADIterator(
            model=torch.jit.load(io.BytesIO(self.model_bytes)),
            threshold=float(self.settings.get("threshold", 0.5)),
            sampling_rate=audio_mod.SAMPLING_RATE,
            min_silence_duration_ms=int(self.settings.get("min_silence_ms", 800)),
            speech_pad_ms=int(self.settings.get("speech_pad_ms", 160)),
        )
        self.buffer = np.empty(0, dtype=np.float32)
        self.disabled = False
        self.consumed = 0
        self.speech_started = 0.0

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
                    if "start" in mark:
                        self.speech_started = max(0.0, at)
                        logging.emit("vad_speech_start", at=round(at, 3))
                    if "end" in mark:
                        silence = float(self.settings.get("min_silence_ms", 800)) / 1000
                        ended = Speech(started_at=self.speech_started,
                                       ended_at=max(0.0, at), silence_waited_out_sec=silence)
                        logging.emit("vad_speech_end", at=round(at, 3),
                                     started_at=round(self.speech_started, 3))
                offset += SILERO_WINDOW_SAMPLES_AT_16K
            self.consumed += offset
            self.buffer = self.buffer[offset:]
        except Exception as e:  # noqa: BLE001
            logging.emit("vad_error", detail=str(e))
            logger.warning("VAD failed, disabled for this item: %s", e)
            self.disabled = True
        return ended

