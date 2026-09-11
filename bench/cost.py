"""API usage accounting.

Appended and flushed per call, never batched at the end: a run that dies mid-way
must still account for what it spent. GPTTranslator reads only
resp.choices[0].message.content and discards resp.usage, so the metered wrapper in
components/translation.py is what feeds this.

No prices are hardcoded. Tokens are reported as tokens; a dollar figure appears
only if the config supplies a rate, because an invented rate is worse than none.
"""
import json
import threading
from pathlib import Path


class UsageLog:
    def __init__(self, path: Path, *, price_per_mtok: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = open(self.path, "a", encoding="utf-8")
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.models: set[str] = set()
        self.price_per_mtok = price_per_mtok or {}

    def record(self, *, model: str, usage, item: str = "", kind: str = "translation") -> None:
        prompt = completion = 0
        if usage is not None:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
        with self._lock:
            self.calls += 1
            self.input_tokens += prompt
            self.output_tokens += completion
            if model:
                self.models.add(model)
            row = {"kind": kind, "model": model, "item": item,
                   "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
            self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._file.flush()

    def summary(self) -> dict:
        out = {
            "usage_log": str(self.path),
            "models": sorted(self.models),
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
        if self.price_per_mtok:
            rate_in = float(self.price_per_mtok.get("input", 0.0))
            rate_out = float(self.price_per_mtok.get("output", 0.0))
            out["estimated_usd"] = round(
                self.input_tokens / 1e6 * rate_in + self.output_tokens / 1e6 * rate_out, 4)
        return out

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()
