"""번역기 두 개: 로컬 Qwen3.5-4B bf16, gpt-6-luna.

gpt-6-luna 는 DialogueContext 의 OpenAIChat(스트리밍으로 TTFT·usage 기록)과 CostLedger 를 그대로 쓴다.
로컬은 여기서 bf16 으로 직접 올린다 — LLMTranslator 의 비양자화 경로는 fp16 이라서다.

translate(system, user) -> dict: raw_output, latency_ms, ttft_ms, generation_ms,
input_tokens, output_tokens, truncated, attempts, api_meta
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "DialogueContext" / "scripts"))

from dctx.backends import OpenAIChat, _FirstToken, _ms  # noqa: E402
from dctx.ledger import CostLedger  # noqa: E402

__all__ = ["LocalBf16", "OpenAIChat", "CostLedger"]


class LocalBf16:
    kind = "local"

    def __init__(self, name: str, model: str, max_new_tokens: int = 256):
        self.name, self.model_name, self.max_new_tokens = name, model, max_new_tokens
        self._tok = self._model = None

    def load(self) -> dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.bfloat16, device_map="cuda")
        self._model.eval()
        torch.cuda.empty_cache()
        ids = self._model.generation_config.eos_token_id
        self._stop = set(ids if isinstance(ids, (list, tuple)) else [ids])
        for i in (self._tok.eos_token_id, self._tok.pad_token_id):
            if i is not None:
                self._stop.add(i)
        return {"load_sec": round(time.perf_counter() - started, 1),
                "dtype": str(next(self._model.parameters()).dtype),
                "class": type(self._model).__name__,
                "vram_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}

    def chat_text(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    def translate(self, system: str, user: str, tag: dict | None = None,  # noqa: ARG002
                  gen_kwargs: dict | None = None) -> dict:
        import torch

        tok, model = self._tok, self._model
        torch.cuda.synchronize()
        started = time.perf_counter()
        enc = tok(self.chat_text(system, user), return_tensors="pt", add_special_tokens=False)
        n_prompt = enc["input_ids"].shape[1]
        enc = {k: v.to(model.device) for k, v in enc.items()}
        streamer = _FirstToken()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=tok.eos_token_id, streamer=streamer,
                                 **(gen_kwargs or {}))
        new = out[0][n_prompt:].tolist()
        raw = tok.decode(new, skip_special_tokens=True)
        torch.cuda.synchronize()
        ended = time.perf_counter()
        n_out = len(new)
        while n_out and new[n_out - 1] in self._stop:
            n_out -= 1
        ttft = streamer.first
        return {
            "raw_output": raw,
            "latency_ms": _ms(ended - started),
            "ttft_ms": _ms(ttft - started) if ttft else None,
            "generation_ms": _ms(ended - ttft) if ttft else None,
            "input_tokens": int(n_prompt),
            "output_tokens": n_out,
            "truncated": len(new) >= self.max_new_tokens and new[-1] not in self._stop,
            "attempts": 1,
            "api_meta": {"peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
        }


def _batch_generate(self, pairs: list[tuple[str, str]], gen_kwargs: dict | None = None) -> list[dict]:
    """여러 요청을 왼쪽 패딩으로 묶어 한 번에 생성한다. 지연은 재지 않는다(요청마다 None).
    순차 생성과 출력이 같은지는 TranslatorPrompt/scripts/check_batch_equiv.py 로 확인한다."""
    import torch

    tok, model = self._tok, self._model
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok([self.chat_text(s, u) for s, u in pairs], return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    width = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id, **(gen_kwargs or {}))
    res = []
    for k in range(len(pairs)):
        new = out[k][width:].tolist()
        n_out = len(new)
        while n_out and new[n_out - 1] in self._stop:
            n_out -= 1
        res.append({"raw_output": tok.decode(new[:n_out], skip_special_tokens=True),
                    "latency_ms": None, "ttft_ms": None, "generation_ms": None,
                    "input_tokens": int(enc["attention_mask"][k].sum()), "output_tokens": n_out,
                    "truncated": n_out >= self.max_new_tokens, "attempts": 1, "api_meta": None})
    return res


LocalBf16.translate_batch = _batch_generate


class LunaChat:
    """OpenAIChat 에 모델 이름과 tag 를 묶어 LocalBf16 과 같은 translate(system, user) 모양으로 맞춘다."""

    kind = "openai"

    def __init__(self, name: str, cfg: dict, key: str, price, ledger, timeout: float):
        self.name = name
        self._chat = OpenAIChat(name, cfg["model"], key, price, ledger,
                                reasoning_effort=cfg.get("reasoning_effort", "none"),
                                temperature=cfg.get("temperature", 0), timeout=timeout,
                                max_retries=cfg.get("max_retries", 3))

    def load(self) -> dict:
        return {"model": self._chat.model}

    def translate(self, system: str, user: str, tag: dict | None = None) -> dict:
        return self._chat.translate(system, user, {}, tag=tag or {})
