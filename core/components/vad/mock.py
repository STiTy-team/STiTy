from core.utils import audio as audio_mod

from . import detectors
from .base import Detector, Speech


def _at_least_one(value) -> int:
    count = int(value)
    if count < 1:
        raise ValueError(f"speech_chunks must be at least 1, got {count}")
    return count


@detectors.register("mock")
class MockVad(Detector):
    SETTINGS = {"speech_chunks": ("speech_chunks", _at_least_one)}

    def start(self, **_) -> None:
        super().start(**_)
        self.chunks = 0
        self.heard_samples = 0
        self.spans.append([0.0, None])

    def detect(self, audio: bytes) -> Speech | None:
        self.heard_samples += len(audio) // 2
        self.chunks += 1
        if self.chunks % self.settings.get("speech_chunks", 5):
            return None
        heard_sec = self.heard_samples / audio_mod.SAMPLING_RATE
        span = self.spans[-1]
        span[1] = heard_sec
        self.spans.append([heard_sec, None])
        return Speech(started_at=span[0], ended_at=heard_sec, silence_waited_out_sec=0.0)
