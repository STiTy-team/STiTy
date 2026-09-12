from ..registry import Component


class Corrector(Component):

    async def correct(self, text: str, language: str | None = None) -> str:
        raise NotImplementedError
