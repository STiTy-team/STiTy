import asyncio
import json
from datetime import datetime, timezone

from core.errors import ConfigError
from core.translator.local_translator import guess_lang_code
from core.utils import env, logging

from . import dialogue, prompt
from .base import Translator

logger = logging.getLogger("bench")

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
REPORT_EVERY = 25
COUNTS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens",
          "billed_characters")


class ApiError(RuntimeError):
    pass


class Spend:

    def __init__(self):
        self.calls = 0
        self.cost = 0.0
        self.counts = dict.fromkeys(COUNTS, 0)

    def add(self, cost: float, counts: dict) -> None:
        self.calls += 1
        self.cost += cost
        for key, value in counts.items():
            self.counts[key] += int(value or 0)

    def snapshot(self) -> dict:
        cost = round(self.cost, 6)
        return {"calls": self.calls, "cost": cost, **self.counts,
                "by_purpose": {"translation": {"calls": self.calls, "cost": cost}}}


def choice(name: str, allowed: tuple[str, ...]):
    def read(value) -> str:
        value = str(value)
        if value not in allowed:
            raise ConfigError(
                f"stity.translation: {name} must be one of {list(allowed)}, got {value!r}")
        return value

    return read


def per_million(tokens: int, usd: float) -> float:
    return tokens * usd / 1_000_000


def token_cost(price: tuple[float, float, float] | None, *, input_tokens: int,
               cached_tokens: int, output_tokens: int) -> float:
    if price is None:
        return 0.0
    fresh, cached, output = price
    return (per_million(input_tokens - cached_tokens, fresh)
            + per_million(cached_tokens, cached)
            + per_million(output_tokens, output))


def _wait_sec(attempt: int, retry_after: str | None) -> float:
    try:
        return min(max(float(retry_after), 0.5), 30.0)
    except (TypeError, ValueError):
        return float(min(2 ** attempt, 8))


class ApiTranslator(Translator):
    KEY_ENV = ""
    SETTINGS = {
        "context": ("context", int),
        "context_chars": ("context_chars", int),
        "timeout_sec": ("timeout_sec", float),
        "max_retries": ("max_retries", int),
        "budget_usd": ("budget_usd", float),
    }

    client = None
    spend = None
    calls = 0
    failed = 0

    def resolve_model(self) -> str:
        raise NotImplementedError

    def price_known(self) -> bool:
        return True

    async def check(self) -> None:
        pass

    async def request(self, text: str, target_lang: str, source_lang: str,
                      context: list[dict], speaker: str) -> tuple[str, str]:
        raise NotImplementedError

    async def load(self) -> None:
        import httpx

        self.key = env.get(self.KEY_ENV)
        if not self.key:
            raise ConfigError(
                f"translation {self.NAME!r} needs {self.KEY_ENV} in the environment or .env")
        self.model = self.resolve_model()
        self.context_finals = self.settings.get("context", dialogue.DEFAULT_COUNT)
        self.context_chars = self.settings.get("context_chars", dialogue.DEFAULT_MAX_CHARS)
        self.max_retries = self.settings.get("max_retries", 3)
        self.budget = self.settings.get("budget_usd")
        if not self.price_known():
            if self.budget is not None:
                raise ConfigError(
                    f"translation {self.NAME!r}: no price for {self.model!r}, "
                    f"so budget_usd cannot be enforced")
            logger.warning("translation %s: no price for %r; cost will be recorded as 0",
                           self.NAME, self.model)
        self.spend = Spend()
        self.budget_reported = False
        self.client = httpx.AsyncClient(timeout=self.settings.get("timeout_sec", 30.0))
        await self.check()
        logger.info("translation %s ready: model=%s context=%d finals / %d chars usage_log=%s",
                    self.NAME, self.model, self.context_finals, self.context_chars,
                    self.usage_log)

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[dict] | None = None,
                        speaker: str = "") -> tuple[str, str]:
        if not text.strip() or not target_lang:
            return "", ""
        source = (source_lang or "").strip().lower() or guess_lang_code(text)
        if source == target_lang:
            return text, source
        if self.budget is not None and self.spend.cost >= self.budget:
            return self._refuse_over_budget(source)
        earlier = dialogue.recent(context, target_lang, self.context_finals, self.context_chars)
        self.calls += 1
        try:
            translation, detected = await self.request(
                text, target_lang, source, earlier, speaker)
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            logging.emit("translate_error", detail=str(e))
            logger.warning("translation failed: %s", e)
            return "", source
        return translation, detected or source

    def _refuse_over_budget(self, source: str) -> tuple[str, str]:
        self.failed += 1
        detail = (f"budget_usd {self.budget} reached "
                  f"(estimated ${self.spend.cost:.4f}); not calling {self.NAME}")
        if not self.budget_reported:
            self.budget_reported = True
            logger.error("translation %s", detail)
        logging.emit("translate_error", detail=detail)
        return "", source

    async def call(self, method: str, url: str, *, headers: dict,
                   body: dict | None = None) -> tuple[dict, int]:
        import httpx

        attempt = 0
        while True:
            retry_after = None
            try:
                response = await self.client.request(method, url, headers=headers, json=body)
            except httpx.TransportError as e:
                error = f"{type(e).__name__}: {e}"
            else:
                if response.status_code < 400:
                    return response.json(), attempt + 1
                error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in RETRYABLE_STATUS:
                    raise ApiError(error)
                retry_after = response.headers.get("retry-after")
            attempt += 1
            if attempt > self.max_retries:
                raise ApiError(f"{error} (gave up after {attempt} attempts)")
            await asyncio.sleep(_wait_sec(attempt, retry_after))

    def record(self, *, cost: float, attempts: int, counts: dict, **detail) -> None:
        self.spend.add(cost, counts)
        line = {
            "at": datetime.now(timezone.utc).isoformat(),
            "backend": self.NAME,
            "model": self.model,
            "attempts": attempts,
            **counts,
            **detail,
            "cost": cost,
            "cumulative_cost": self.spend.cost,
        }
        if self.usage_log is None:
            logging.emit("translation_usage", **line)
        else:
            with open(self.usage_log, "a", encoding="utf-8") as output:
                output.write(json.dumps(line, ensure_ascii=False) + "\n")
        if self.spend.calls % REPORT_EVERY == 0:
            logger.info("translation %s: %d calls, estimated cost $%.4f",
                        self.NAME, self.spend.calls, self.spend.cost)

    def usage(self) -> dict | None:
        if self.spend is None:
            return None
        return {"backend": self.NAME, "model": self.model,
                "requests": self.calls, "failed": self.failed, **self.spend.snapshot()}

    async def close(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            await client.aclose()
        if self.spend is not None:
            logger.info("translation %s: %d calls, estimated cost $%.4f",
                        self.NAME, self.spend.calls, self.spend.cost)


class ChatTranslator(ApiTranslator):
    DEFAULT_MODEL = ""
    SETTINGS = {**ApiTranslator.SETTINGS, "model": ("model", str)}

    def resolve_model(self) -> str:
        return self.settings.get("model", self.DEFAULT_MODEL)

    async def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    async def request(self, text: str, target_lang: str, source_lang: str,
                      context: list[dict], speaker: str) -> tuple[str, str]:
        system, user = prompt.messages(text, target_lang, source_lang, context, speaker)
        reply = await self.complete(system, user)
        translation = prompt.extract(reply, speaker)
        if not translation:
            raise ApiError(f"{self.NAME} reply has no translation field: {reply[:300]!r}")
        return translation, source_lang
