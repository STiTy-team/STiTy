from core.utils import langs

from . import langids
from .base import LanguageDetector, Transcribed


@langids.register("alphabet")
class AlphabetLanguageDetector(LanguageDetector):

    SETTINGS = {"threshold": ("threshold", float), "langs": ("langs", tuple)}

    async def detect(self, item: Transcribed) -> str | None:
        candidates = self.settings.get("langs") or langs.ALPHABET_LANGS
        threshold = self.settings.get("threshold", langs.ALPHABET_THRESHOLD)
        return langs.detect_by_alphabet(item.original, langs=candidates, threshold=threshold)
