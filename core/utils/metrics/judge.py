"""LLM JSON judge shared by the annotation steps, with a per-call cost log."""
from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


QUALITY_ANNOTATOR_VERSION = "2026-09-21.2"


class JsonJudge(Protocol):
    model: str

    async def ask(self, *, purpose: str, system: str, payload: dict) -> dict:
        ...


class OpenAIJsonJudge:
    retry_delay_sec = 1.0
    report_every = 25

    def __init__(self, *, model: str = "gpt-5.4-mini", api_key: str | None = None,
                 max_retries: int = 3, usage_log: str | Path | None = None, client=None):
        from core.meaning_segmentator.autoseg.infra.gateway import Usage, _price_of

        if client is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is required for automatic quality annotation")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=key)
        price = _price_of(model)
        if price is None:
            print(f"[judge] warning: no price for {model!r}; cost will be recorded as 0",
                  flush=True)
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.usage = Usage(price=price)
        self.usage_log = Path(usage_log) if usage_log else None

    def _record(self, purpose: str, payload: dict) -> None:
        before = self.usage.cost
        self.usage.add(payload, purpose)
        usage = payload.get("usage") or {}
        if self.usage_log is not None:
            line = {
                "at": datetime.now(timezone.utc).isoformat(),
                "purpose": purpose,
                "model": self.model,
                "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
                "cached_tokens": (usage.get("prompt_tokens_details") or {})
                .get("cached_tokens", 0) or 0,
                "cost": self.usage.cost - before,
                "cumulative_cost": self.usage.cost,
            }
            with open(self.usage_log, "a", encoding="utf-8") as output:
                output.write(json.dumps(line, ensure_ascii=False) + "\n")
        if self.usage.calls % self.report_every == 0:
            print(f"[judge] {self.usage.calls} calls, estimated cost "
                  f"${self.usage.cost:.4f}", flush=True)

    async def ask(self, *, purpose: str, system: str, payload: dict) -> dict:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                self._record(purpose, response.model_dump())
                value = json.loads(response.choices[0].message.content)
                if not isinstance(value, dict):
                    raise ValueError(f"{purpose} did not return a JSON object")
                return value
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(self.retry_delay_sec * 2 ** attempt, 8))
        raise RuntimeError(f"{purpose} failed after {self.max_retries} attempts: {last_error}")


def locate(text: str, surface: str, start, end) -> tuple[int, int] | None:
    if (isinstance(start, int) and isinstance(end, int)
            and 0 <= start < end <= len(text) and text[start:end] == surface):
        return start, end
    positions = [match.start() for match in re.finditer(re.escape(surface), text)]
    if not positions:
        return None
    anchor = start if isinstance(start, int) else 0
    best = min(positions, key=lambda position: (abs(position - anchor), position))
    return best, best + len(surface)


__all__ = ["QUALITY_ANNOTATOR_VERSION", "JsonJudge", "OpenAIJsonJudge", "locate"]
