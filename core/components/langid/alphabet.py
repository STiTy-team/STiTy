from core.utils import langs

from . import langids
from .base import LanguageDetector


@langids.register("alphabet")
class AlphabetLanguageDetector(LanguageDetector):

    SETTINGS = {"threshold": ("threshold", float), "langs": ("langs", tuple)}

    async def detect(self, text: str) -> str | None:
        candidates = self.settings.get("langs") or langs.ALPHABET_LANGS
        threshold = self.settings.get("threshold", langs.ALPHABET_THRESHOLD)
        return langs.detect_by_alphabet(text, langs=candidates, threshold=threshold)
