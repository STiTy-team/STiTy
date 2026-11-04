from . import correctors
from .base import Corrector


@correctors.register("mock")
class MockCorrector(Corrector):

    async def correct(self, text: str, language: str | None = None) -> str:
        return text
