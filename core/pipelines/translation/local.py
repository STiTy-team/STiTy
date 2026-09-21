from core.errors import ConfigError
from core.utils import logging

from . import translators
from .base import Translator
from .dialogue import DEFAULT_COUNT, DEFAULT_MAX_CHARS, recent

logger = logging.getLogger("bench")

DEFAULT_MODEL = "google/madlad400-3b-mt"


def takes_context(model: str) -> bool:
    from core.translator.local_translator import make_translator

    return bool(getattr(make_translator(model_name=model), "SUPPORTS_CONTEXT", False))


@translators.register('local')
class LocalTranslation(Translator):
    SETTINGS = {"model": ("model", str), "device": ("device", str), "quant": ("quant", str),
                "context": ("context", int), "context_chars": ("context_chars", int)}

    translator = None
    calls = 0
    failed = 0

    @classmethod
    def validate(cls, options: dict, *, kind: str) -> dict:
        resolved = super().validate(options, kind=kind)
        model = resolved.get("model", DEFAULT_MODEL)
        if resolved.get("context") and not takes_context(model):
            raise ConfigError(
                f"stity.{kind}: {model!r} has no place for context, so it cannot be given "
                f"the same earlier speech as the other translators; set context: 0")
        return resolved

    async def load(self) -> None:
        from core.translator.local_translator import make_translator

        self.model = self.settings.get("model", DEFAULT_MODEL)
        self.device = self.settings.get("device")
        self.context_count = self.settings.get(
            "context", DEFAULT_COUNT if takes_context(self.model) else 0)
        self.context_chars = self.settings.get("context_chars", DEFAULT_MAX_CHARS)
        extra = {"quant": self.settings["quant"]} if "quant" in self.settings else {}
        extra.update(context_window=self.context_count, always_use_context=True)
        logger.info("loading translation model: %s %s (context=%d finals / %d chars)",
                    self.model, extra, self.context_count, self.context_chars)
        self.translator = make_translator(model_name=self.model, device=self.device, **extra)
        self.translator.load()

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[dict] | None = None,
                        speaker: str = "") -> tuple[str, str]:
        if not text.strip() or not target_lang:
            return "", ""
        self.calls += 1
        try:
            earlier = recent(context, target_lang, self.context_count, self.context_chars)
            return await self.translator.translate(
                text, target_lang, source_lang, context=earlier)
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            logging.emit("translate_error", detail=str(e))
            logger.warning("translation failed: %s", e)
            return "", ""

    async def close(self) -> None:
        """Release model memory before a later metric model is loaded."""
        backend, self.translator = self.translator, None
        if backend is None:
            return
        close = getattr(backend, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
        del backend

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

