from ..registry import Component, Transcribed


class LanguageDetector(Component):

    def hear(self, audio: bytes) -> None:
        pass

    async def detect(self, item: Transcribed) -> str | None:
        raise NotImplementedError
