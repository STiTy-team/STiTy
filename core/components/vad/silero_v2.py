from collections import deque

import numpy as np

from core.utils import audio as audio_mod
from core.utils import logging

from . import detectors
from .base import Detector, Speech
from .silero import SILERO_WINDOW_SAMPLES_AT_16K, SileroVad
from .tuners import SMART_TURN_SECONDS, build_tuner, parse_tuner

log = logging.getLogger(__name__)

NEGATIVE_MARGIN = 0.15


@detectors.register("silero:v2")
class SileroVadV2(SileroVad):
    SETTINGS = {
        **SileroVad.SETTINGS,
        "tuner": ("tuner", parse_tuner),
    }

    async def load(self) -> None:
        await super().load()
        self.tuner = build_tuner(
            self.settings.get("tuner"),
            threshold=float(self.settings.get("threshold", 0.5)),
            min_silence_ms=int(self.settings.get("min_silence_ms", 800)))
        await self.tuner.load()
        log.info("[LOAD] silero:v2 tuner=%s", type(self.tuner).__name__)

    def start(self, **_) -> None:
        import io

        import torch

        Detector.start(self, **_)
        self.model = torch.jit.load(io.BytesIO(self.model_bytes))
        self.model.reset_states()
        self.tuner.start()
        self.buffer = np.empty(0, dtype=np.float32)
        self.recent = deque(maxlen=SMART_TURN_SECONDS * audio_mod.SAMPLING_RATE
                            // SILERO_WINDOW_SAMPLES_AT_16K)
        self.disabled = False
        self.consumed = 0
        self.current = 0
        self.triggered = False
        self.temp_end = 0
        self.segment_start = 0
        self.speech_started = 0.0

    def detect(self, audio: bytes) -> Speech | None:
        if self.disabled:
            return None
        self.buffer = np.concatenate([self.buffer, audio_mod.from_pcm_bytes(audio)])
        ended = None
        try:
            offset = 0
            while offset + SILERO_WINDOW_SAMPLES_AT_16K <= self.buffer.size:
                window = self.buffer[offset:offset + SILERO_WINDOW_SAMPLES_AT_16K]
                ended = self._step(window) or ended
                offset += SILERO_WINDOW_SAMPLES_AT_16K
            self.consumed += offset
            self.buffer = self.buffer[offset:]
        except Exception as e:  # noqa: BLE001
            log.warning("[VAD-ERROR] disabled for this item: %s", e)
            self.disabled = True
            self.spans = []
        return ended

    def _probability(self, window: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            return float(self.model(torch.from_numpy(window), audio_mod.SAMPLING_RATE).item())

    def _step(self, window: np.ndarray) -> Speech | None:
        self.recent.append(window)
        self.current += window.size
        prob = self._probability(window)
        threshold = self.tuner.threshold()
        self.tuner.observe(prob, triggered=self.triggered)
        at = self.current / audio_mod.SAMPLING_RATE

        if prob >= threshold and self.temp_end:
            self.tuner.on_pause(self._ms(self.current - self.temp_end))
            self.temp_end = 0

        if prob >= threshold and not self.triggered:
            self.triggered = True
            self.segment_start = self.current
            self.speech_started = at
            self.spans.append([at, None])
            self.tuner.on_speech_start()
            return None

        if prob < threshold - NEGATIVE_MARGIN and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current
            silence_ms = self._ms(self.current - self.temp_end)
            segment_sec = (self.current - self.segment_start) / audio_mod.SAMPLING_RATE
            waited = self.tuner.should_end(silence_ms, segment_sec, self._recent_audio)
            if waited is None:
                return None
            self.temp_end = 0
            self.triggered = False
            if self.spans and self.spans[-1][1] is None:
                self.spans[-1][1] = at
            log.debug("[VAD-END] segment_sec=%.2f waited_ms=%.0f threshold=%.2f",
                      segment_sec, waited, threshold)
            return Speech(started_at=self.speech_started, ended_at=at,
                          silence_waited_out_sec=waited / 1000)
        return None

    def _recent_audio(self) -> np.ndarray:
        return np.concatenate(list(self.recent)) if self.recent else np.zeros(0, np.float32)

    @staticmethod
    def _ms(samples: int) -> float:
        return samples * 1000 / audio_mod.SAMPLING_RATE
