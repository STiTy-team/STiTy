import json
import os
import re
import time
from pathlib import Path

from core.errors import ConfigError
from core.utils import logging

from . import translators
from .base import Translator
from .local_qwen_3_5_4b.translator import (SYSTEM_PROMPT, ends_a_sentence,
                                           language_name, without_reasoning)

log = logging.getLogger(__name__)

DEFAULT_URL = "https://gw.letsur.ai/v1"
DEFAULT_KEY_ENV = "LETSUR_API_KEY"
DEFAULT_USAGE_PATH = "translation_usage.jsonl"
USAGE_LOG_EVERY = 20
PUNCTUATION = re.compile(r"[.,!?;:…。、？！]+")

SPEECH_NOTE = (
    "The line comes from live speech recognition, so it may contain misheard words that sound "
    "like the intended ones. Translate what the speaker most plausibly meant, using the "
    "conversation for clues.")
NO_PUNCTUATION_NOTE = (
    "Punctuation has been removed from the speech. Infer whether each line is a question, a "
    "statement or a request from its wording and the conversation, and punctuate your "
    "translation accordingly.")


class EmptyReply(Exception):
    pass


@translators.register("api")
class ApiTranslation(Translator):
    SETTINGS = {
        "url": ("url", str),
        "model": ("model", str),
        "key_env": ("key_env", str),
        "reasoning_effort": ("reasoning_effort", str),
        "context_turns": ("context_turns", int),
        "max_tokens": ("max_tokens", int),
        "timeout_sec": ("timeout_sec", float),
        "usage_path": ("usage_path", str),
        "extra_body": ("extra_body", dict),
        "fallback_model": ("fallback_model", str),
        "strip_punctuation": ("strip_punctuation", bool),
    }

    async def load(self) -> None:
        import httpx

        self.url = self.settings.get("url", DEFAULT_URL).rstrip("/")
        self.model = self.settings.get("model")
        if not self.model:
            raise ConfigError("translation.api needs a model")
        key_env = self.settings.get("key_env", DEFAULT_KEY_ENV)
        key = os.environ.get(key_env, "")
        if not key:
            raise ConfigError(f"translation.api needs {key_env} in the environment")
        self.reasoning_effort = self.settings.get("reasoning_effort", "low")
        self.context_turns = self.settings.get("context_turns", 1)
        self.max_tokens = self.settings.get("max_tokens", 1024)
        self.extra_body = self.settings.get("extra_body", {})
        self.fallback_model = self.settings.get("fallback_model", self.model)
        self.strip_punctuation = self.settings.get("strip_punctuation", False)
        self.usage_path = Path(self.settings.get("usage_path", DEFAULT_USAGE_PATH))
        self.calls = 0
        self.failed = 0
        self.cost_usd = 0.0
        self.client = httpx.AsyncClient(
            timeout=self.settings.get("timeout_sec", 20.0),
            headers={"Authorization": f"Bearer {key}"})
        log.info("[LOAD] api translation %s at %s", self.model, self.url)

    async def close(self) -> None:
        log.info("[USAGE] calls=%d failed=%d cost_usd=%.4f", self.calls, self.failed, self.cost_usd)
        try:
            await self.client.aclose()
        except RuntimeError:
            pass

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[str] | None = None) -> tuple[str, str]:
        if not text.strip() or not target_lang:
            return "", ""
        if source_lang == target_lang:
            return text, source_lang
        prompt = self._prompt(text.strip(), target_lang, source_lang, context or [])
        try:
            reply = await self._ask(prompt)
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            log.warning("[TRANS-ERROR] %s", e)
            return "", ""
        return without_reasoning(reply), source_lang or ""

    def _system_prompt(self, base: str) -> str:
        notes = [SPEECH_NOTE, NO_PUNCTUATION_NOTE] if self.strip_punctuation else [SPEECH_NOTE]
        return " ".join([base, *notes])

    def _for_model(self, text: str) -> str:
        if not self.strip_punctuation:
            return text
        return " ".join(PUNCTUATION.sub(" ", text).split())

    def _prompt(self, text: str, target_lang: str, source_lang: str | None,
                context: list[str]) -> list[dict]:
        source, target = language_name(source_lang), language_name(target_lang)
        earlier_turns = self._turns_to_show(text, context)
        text = self._for_model(text)
        if earlier_turns:
            listed = "\n".join(f"- {self._for_model(turn)}" for turn in earlier_turns)
            request = (f"Earlier turns of the conversation, oldest first (context only; do NOT "
                       f"translate these):\n{listed}\n\n"
                       f"Translate ONLY the line below, which is in {source}, into {target}.\n\n{text}")
        else:
            request = f"Translate the following {source} sentence into {target}.\n\n{text}"
        return [{"role": "system", "content": self._system_prompt(SYSTEM_PROMPT)},
                {"role": "user", "content": request}]

    def _turns_to_show(self, text: str, context: list[str]) -> list[str]:
        if not self.context_turns or not ends_a_sentence(text):
            return []
        return context[-self.context_turns:]

    async def _ask(self, messages: list[dict]) -> str:
        try:
            return await self._chat(messages, self.model)
        except Exception as e:  # noqa: BLE001
            log.warning("[TRANS-RETRY] %s, asking %s", e, self.fallback_model)
            return await self._chat(messages, self.fallback_model)

    async def _chat(self, messages: list[dict], model: str) -> str:
        body = {"model": model, "messages": messages, "max_tokens": self.max_tokens,
                **self.extra_body}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        response = await self.client.post(f"{self.url}/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        self._record_usage(data, model, choice.get("finish_reason"))
        content = choice["message"].get("content") or ""
        if not without_reasoning(content):
            raise EmptyReply(f"empty reply from {model} (finish_reason={choice.get('finish_reason')})")
        return content

    def _record_usage(self, data: dict, model: str, finish_reason: str | None) -> None:
        self.calls += 1
        usage = data.get("usage") or {}
        cost = float((data.get("estimated_cost") or {}).get("amount") or usage.get("cost") or 0.0)
        self.cost_usd += cost
        line = {"t": time.time(), "model": model,
                "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
                "cost": cost, "finish": finish_reason}
        with self.usage_path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        log.info("[USAGE-CALL] %s", json.dumps(line))
        if self.calls % USAGE_LOG_EVERY == 0:
            log.info("[USAGE] calls=%d cost_usd=%.4f", self.calls, self.cost_usd)
