from ..registry import Component, Partial, Speech, Transcribed

__all__ = ["Transcriber", "Partial", "Speech", "Transcribed"]


class Transcriber(Component):

    def start(self, language: str | None = None, **_) -> None:
        pass

    async def transcribe(self, audio: bytes) -> list:
        raise NotImplementedError

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        raise NotImplementedError

    async def finish(self, reason: str = "finish",
                     speech: Speech | None = None) -> list:
        raise NotImplementedError
