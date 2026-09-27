from core.utils import audio as audio_mod
from core.utils import timing

from . import transcribers
from ..registry import Partial, Speech, Transcribed
from .base import Transcriber


@transcribers.register("mock")
class MockTranscriber(Transcriber):

    def start(self, languages: list[str] | None = None, **_) -> None:
        self.language = (languages or ["en"])[0]
        self.words: list[str] = []
        self.count = 0
        self.seq = 0
        self.heard_samples = 0

    async def transcribe(self, audio: bytes) -> list:
        self.heard_samples += len(audio) // 2
        self.count += 1
        self.seq += 1
        self.words.append(f"word{self.count}")
        return [Partial(text=" ".join(self.words), language=self.language, seq=self.seq)]

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        if not self.words:
            return []
        original = " ".join(self.words)
        self.words = []
        self.seq += 1
        return [Transcribed(original=original, language=self.language,
                            commit_reason=reason,
                            decision_audio_sec=round(self.heard_samples / audio_mod.SAMPLING_RATE, 3),
                            committed_elapsed_sec=round(timing.elapsed() or 0.0, 4)),
                Partial(text="", language="", seq=self.seq)]

    async def finish(self, reason: str = "finish", speech: Speech | None = None) -> list:
        return await self.flush(reason, speech)
