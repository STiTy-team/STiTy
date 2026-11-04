import asyncio

from . import translators
from .base import Translator


def _not_negative(value) -> float:
    seconds = float(value)
    if seconds < 0:
        raise ValueError(f"delay_sec must not be negative, got {seconds}")
    return seconds


@translators.register("mock")
class MockTranslator(Translator):
    SETTINGS = {"delay_sec": ("delay_sec", _not_negative)}

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[str] | None = None) -> tuple[str, str]:
        await asyncio.sleep(self.settings.get("delay_sec", 0.0))
        return f"[{target_lang}] {text}", source_lang or ""
