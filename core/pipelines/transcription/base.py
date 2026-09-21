from ..registry import Component, Speech


class Transcriber(Component):

    def start(self, language: str | None = None, **_) -> None:
        pass

    async def transcribe(self, audio: bytes) -> None:
        raise NotImplementedError

    async def flush(self, reason: str, speech: Speech | None = None) -> None:
        raise NotImplementedError

    async def finish(self, reason: str = "finish",
                     speech: Speech | None = None) -> None:
        raise NotImplementedError
