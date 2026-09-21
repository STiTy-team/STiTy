from . import translators
from .api import ApiError, ChatTranslator, choice, token_cost

URL = "https://api.openai.com/v1"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def price_of(model: str) -> tuple[float, float, float] | None:
    from core.meaning_segmentator.autoseg.infra.gateway import _price_of

    return _price_of(model)


@translators.register("gpt")
class GPTTranslation(ChatTranslator):
    KEY_ENV = "OPENAI_API_KEY"
    DEFAULT_MODEL = "gpt-5.4-nano"
    SETTINGS = {
        **ChatTranslator.SETTINGS,
        "reasoning_effort": ("reasoning_effort", choice("reasoning_effort", REASONING_EFFORTS)),
    }

    def price_known(self) -> bool:
        return price_of(self.model) is not None

    async def check(self) -> None:
        self.price = price_of(self.model)
        self.headers = {"Authorization": f"Bearer {self.key}"}
        self.reasoning_effort = self.settings.get("reasoning_effort", "none")
        await self.call("GET", f"{URL}/models/{self.model}", headers=self.headers)

    async def complete(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "reasoning_effort": self.reasoning_effort,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort == "none":
            body["temperature"] = 0
        data, attempts = await self.call(
            "POST", f"{URL}/chat/completions", headers=self.headers, body=body)

        usage = data.get("usage") or {}
        counts = {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "cached_tokens": (usage.get("prompt_tokens_details") or {})
            .get("cached_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {})
            .get("reasoning_tokens", 0) or 0,
        }
        first = (data.get("choices") or [{}])[0]
        finish = first.get("finish_reason")
        self.record(cost=token_cost(self.price, input_tokens=counts["input_tokens"],
                                    cached_tokens=counts["cached_tokens"],
                                    output_tokens=counts["output_tokens"]),
                    attempts=attempts, counts=counts, finish_reason=finish)

        text = (first.get("message") or {}).get("content") or ""
        if not text.strip():
            raise ApiError(f"gpt returned no text (finish_reason={finish})")
        return text
