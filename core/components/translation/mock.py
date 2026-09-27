from . import translators
from .base import Translator


@translators.register("mock")
class MockTranslator(Translator):

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[str] | None = None) -> tuple[str, str]:
        return f"[{target_lang}] {text}", source_lang or ""
