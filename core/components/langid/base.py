from ..registry import Component


class LanguageDetector(Component):

    async def detect(self, text: str) -> str | None:
        raise NotImplementedError
