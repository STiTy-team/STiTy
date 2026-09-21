from . import translators
from .api import ApiError, ChatTranslator, choice, token_cost

URL = "https://generativelanguage.googleapis.com/v1beta"
THINKING_LEVELS = ("minimal", "low", "medium", "high")
PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-3.1-flash-lite": (0.25, 0.025, 1.50),
}


def price_of(model: str) -> tuple[float, float, float] | None:
    matches = [name for name in PRICES if model.startswith(name)]
    return PRICES[max(matches, key=len)] if matches else None


@translators.register("gemini")
class GeminiTranslation(ChatTranslator):
    KEY_ENV = "GEMINI_API_KEY"
    DEFAULT_MODEL = "gemini-3.1-flash-lite"
    SETTINGS = {
        **ChatTranslator.SETTINGS,
        "thinking_level": ("thinking_level", choice("thinking_level", THINKING_LEVELS)),
        "temperature": ("temperature", float),
    }

    def price_known(self) -> bool:
        return price_of(self.model) is not None

    async def check(self) -> None:
        self.price = price_of(self.model)
        self.headers = {"x-goog-api-key": self.key}
        self.thinking_level = self.settings.get("thinking_level", "minimal")
        await self.call("GET", f"{URL}/models/{self.model}", headers=self.headers)

    async def complete(self, system: str, user: str) -> str:
        generation = {"thinkingConfig": {"thinkingLevel": self.thinking_level},
                      "responseMimeType": "application/json"}
        if "temperature" in self.settings:
            generation["temperature"] = self.settings["temperature"]
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        data, attempts = await self.call(
            "POST", f"{URL}/models/{self.model}:generateContent",
            headers=self.headers, body=body)

        usage = data.get("usageMetadata") or {}
        thoughts = usage.get("thoughtsTokenCount", 0) or 0
        counts = {
            "input_tokens": usage.get("promptTokenCount", 0) or 0,
            "cached_tokens": usage.get("cachedContentTokenCount", 0) or 0,
            "output_tokens": (usage.get("candidatesTokenCount", 0) or 0) + thoughts,
            "reasoning_tokens": thoughts,
        }
        candidate = (data.get("candidates") or [{}])[0]
        finish = candidate.get("finishReason")
        self.record(cost=token_cost(self.price, input_tokens=counts["input_tokens"],
                                    cached_tokens=counts["cached_tokens"],
                                    output_tokens=counts["output_tokens"]),
                    attempts=attempts, counts=counts, finish_reason=finish)

        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        if not text.strip():
            blocked = (data.get("promptFeedback") or {}).get("blockReason")
            raise ApiError(f"gemini returned no text (finishReason={finish}, "
                           f"blockReason={blocked})")
        return text
