from core.errors import ConfigError
from core.utils import logging

from . import translators
from .api import ApiError, ApiTranslator, choice, per_million
from . import prompt

logger = logging.getLogger("bench")

MODEL_TYPES = ("latency_optimized", "quality_optimized", "prefer_quality_optimized")
FORMALITIES = ("default", "more", "less", "prefer_more", "prefer_less")
TARGET_CODES = {"en": "EN-US", "pt": "PT-BR", "zh": "ZH-HANS"}
PRO_URL = "https://api.deepl.com"
FREE_URL = "https://api-free.deepl.com"
DEFAULT_USD_PER_MILLION_CHARACTERS = 25.0


def transcript(context: list[dict], source_lang: str, target_lang: str) -> str:
    if any(entry.get("speaker") for entry in context):
        return prompt.speaker_lines(context, target_lang)
    text, _ = prompt.said(context, source_lang)
    shown = prompt.seen(context, target_lang)
    return text if shown is None else f"{text}\n{shown}"


@translators.register("deepl")
class DeepLTranslation(ApiTranslator):
    KEY_ENV = "DEEPL_API_KEY"
    SETTINGS = {
        **ApiTranslator.SETTINGS,
        "model_type": ("model_type", choice("model_type", MODEL_TYPES)),
        "formality": ("formality", choice("formality", FORMALITIES)),
        "usd_per_million_characters": ("usd_per_million_characters", float),
    }

    @classmethod
    def validate(cls, options: dict, *, kind: str) -> dict:
        resolved = super().validate(options, kind=kind)
        if "model_type" not in resolved:
            raise ConfigError(
                f"stity.{kind}: 'deepl' needs model_type, one of {list(MODEL_TYPES)}")
        return resolved

    def resolve_model(self) -> str:
        return self.settings["model_type"]

    async def check(self) -> None:
        self.free = self.key.endswith(":fx")
        self.url = FREE_URL if self.free else PRO_URL
        self.headers = {"Authorization": f"DeepL-Auth-Key {self.key}"}
        self.usd_per_million = (0.0 if self.free else self.settings.get(
            "usd_per_million_characters", DEFAULT_USD_PER_MILLION_CHARACTERS))
        quota, _ = await self.call("GET", f"{self.url}/v2/usage", headers=self.headers)
        logger.info("deepl %s plan: %s of %s characters used this period",
                    "free" if self.free else "pro",
                    quota.get("character_count"), quota.get("character_limit"))

    async def request(self, text: str, target_lang: str, source_lang: str,
                      context: list[dict], speaker: str) -> tuple[str, str]:
        body = {
            "text": [text],
            "source_lang": source_lang.upper(),
            "target_lang": TARGET_CODES.get(target_lang, target_lang.upper()),
            "model_type": self.model,
            "show_billed_characters": True,
        }
        if context:
            body["context"] = transcript(context, source_lang, target_lang)
        if "formality" in self.settings:
            body["formality"] = self.settings["formality"]
        data, attempts = await self.call(
            "POST", f"{self.url}/v2/translate", headers=self.headers, body=body)
        translations = data.get("translations") or []
        result = translations[0] if translations else {}
        characters = int(result.get("billed_characters") or 0)
        self.record(cost=per_million(characters, self.usd_per_million), attempts=attempts,
                    counts={"billed_characters": characters},
                    model_type_used=result.get("model_type_used"))
        if not translations:
            raise ApiError(f"deepl returned no translation: {data!r}"[:500])
        detected = (result.get("detected_source_language") or "").lower()
        return (result.get("text") or "").strip(), detected
