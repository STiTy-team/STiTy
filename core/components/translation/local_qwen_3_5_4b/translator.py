import re
from pathlib import Path
from urllib.parse import urlparse

from core.errors import ConfigError
from core.utils import cache
from core.utils import logging
from core.utils.langs import CODE_TO_NAME
from core.utils.process import ManagedProcess, uv_command

from .. import translators
from ..base import Translator

log = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8100/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
VLLM_PROJECT = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are a translation engine for a live conversation. The earlier turns are context "
    "only: never translate them, and never repeat their translation. Use them only to "
    "resolve pronouns, omitted subjects, gender agreement and politeness level in the "
    "CURRENT line. Translate the CURRENT line and nothing else, even if it looks garbled "
    "or incomplete. Output only that translation, with no explanation and no quotes."
)

SENTENCE_END = re.compile(r"[.!?。？！]\s*[\"'」』)]*\s*$")


def language_name(code: str | None) -> str:
    return CODE_TO_NAME.get(code or "", code or "the source language")


def ends_a_sentence(text: str) -> bool:
    return bool(SENTENCE_END.search(text))


def without_reasoning(reply: str) -> str:
    return reply.split("</think>")[-1].strip().strip('"')


@translators.register("qwen3.5")
class Qwen35Translation(Translator):
    SETTINGS = {
        "url": ("url", str),
        "model": ("model", str),
        "gpu_memory_utilization": ("gpu_memory_utilization", float),
        "max_model_len": ("max_model_len", int),
        "startup_timeout_sec": ("startup_timeout_sec", float),
        "context_turns": ("context_turns", int),
        "max_tokens": ("max_tokens", int),
        "timeout_sec": ("timeout_sec", float),
    }

    async def load(self) -> None:
        import httpx

        self.url = self.settings.get("url", DEFAULT_URL).rstrip("/")
        self.model = self.settings.get("model", DEFAULT_MODEL)
        self.context_turns = self.settings.get("context_turns", 1)
        self.max_tokens = self.settings.get("max_tokens", 256)
        self.client = httpx.AsyncClient(timeout=self.settings.get("timeout_sec", 10.0))
        await cache.load(("qwen3.5-server", self.url, self.model), self._start_server_unless_running)
        log.info("[LOAD] qwen3.5 translation %s at %s", self.model, self.url)

    async def close(self) -> None:
        await self.client.aclose()

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[str] | None = None) -> tuple[str, str]:
        if not text.strip() or not target_lang:
            return "", ""
        if source_lang == target_lang:
            return text, source_lang
        prompt = self._prompt(text.strip(), target_lang, source_lang, context or [])
        try:
            reply = await self._chat(prompt)
        except Exception as e:  # noqa: BLE001
            log.warning("[TRANS-ERROR] %s", e)
            return "", ""
        return without_reasoning(reply), source_lang or ""

    def _prompt(self, text: str, target_lang: str, source_lang: str | None,
                context: list[str]) -> list[dict]:
        source, target = language_name(source_lang), language_name(target_lang)
        earlier_turns = self._turns_to_show(text, context)
        if earlier_turns:
            listed = "\n".join(f"- {turn}" for turn in earlier_turns)
            request = (f"Earlier turns of the conversation, oldest first (context only; do NOT "
                       f"translate these):\n{listed}\n\n"
                       f"Translate ONLY the line below, which is in {source}, into {target}.\n\n{text}")
        else:
            request = f"Translate the following {source} sentence into {target}.\n\n{text}"
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request}]

    def _turns_to_show(self, text: str, context: list[str]) -> list[str]:
        if not self.context_turns or not ends_a_sentence(text):
            return []
        return context[-self.context_turns:]

    async def _chat(self, messages: list[dict]) -> str:
        response = await self.client.post(f"{self.url}/chat/completions", json={
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    async def _start_server_unless_running(self) -> None:
        if not await self._server_is_up():
            await self._start_server()
            return
        served = await self._served_models()
        if self.model not in served:
            raise ConfigError(f"{self.url} serves {served}, not {self.model!r}")
        log.info("[LOAD] reusing running server at %s", self.url)

    async def _start_server(self) -> None:
        address = urlparse(self.url)
        command = uv_command(
            VLLM_PROJECT, "vllm", "serve", self.model,
            host=address.hostname or "127.0.0.1",
            port=address.port or 8100,
            gpu_memory_utilization=self.settings.get("gpu_memory_utilization", 0.25),
            max_model_len=self.settings.get("max_model_len", 4096),
        )
        server = ManagedProcess("qwen3.5-vllm", command, ready=self._server_is_up,
                                startup_timeout_sec=self.settings.get("startup_timeout_sec", 600.0))
        await server.start()

    async def _server_is_up(self) -> bool:
        try:
            await self._served_models()
        except Exception:  # noqa: BLE001
            return False
        return True

    async def _served_models(self) -> list[str]:
        response = await self.client.get(f"{self.url}/models")
        response.raise_for_status()
        return [entry["id"] for entry in response.json()["data"]]
