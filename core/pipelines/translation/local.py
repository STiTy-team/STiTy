from core.utils import logging

from . import translators
from .base import Translator

logger = logging.getLogger("bench")

DEFAULT_MODEL = "google/madlad400-3b-mt"


@translators.register('local')
class LocalTranslation(Translator):
    SETTINGS = {"model": ("model", str), "device": ("device", str)}

    translator = None
    calls = 0
    failed = 0

    async def load(self) -> None:
        from core.translator.local_translator import make_translator

        self.model = self.settings.get("model", DEFAULT_MODEL)
        self.device = self.settings.get("device")
        logger.info("loading translation model: %s", self.model)
        self.translator = make_translator(model_name=self.model, device=self.device)
        self.translator.load()

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[str] | None = None) -> tuple[str, str]:
        if not text.strip() or not target_lang:
            return "", ""
        self.calls += 1
        try:
            return await self.translator.translate(
                text, target_lang, source_lang, context=context)
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            logging.emit("translate_error", detail=str(e))
            logger.warning("translation failed: %s", e)
            return "", ""

