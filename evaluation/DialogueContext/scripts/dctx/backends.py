"""번역기 네 개를 같은 모양으로 부른다: translate(system, user, request) -> dict.

돌려주는 dict: raw_output, latency_ms, ttft_ms, generation_ms, input_tokens, output_tokens,
truncated, attempts, api_meta.

- latency_ms: 요청 시작~최종 출력. 로컬은 chat template 적용·토큰화·generate·디코드까지,
  API 는 네트워크 포함이며 재시도했으면 마지막 시도만 잰다(시도 횟수는 attempts).
- ttft_ms: 로컬은 첫 생성 토큰, GPT 는 첫 비어 있지 않은 content 조각, DeepL 은 None.
- generation_ms: 첫 토큰부터 끝까지(latency_ms - ttft_ms). DeepL 은 None.
"""
import json
import time

from core.pipelines.translation.api import RETRYABLE_STATUS, _wait_sec, token_cost
from core.pipelines.translation.deepl import FREE_URL, PRO_URL, TARGET_CODES
from core.translator.local_translator import LLMTranslator


class ApiFailure(RuntimeError):
    pass


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


class _FirstToken:
    """generate 의 streamer 자리에 꽂는다. 첫 put 은 프롬프트, 두 번째가 첫 생성 토큰이다.
    transformers 가 put 에 넘기기 전에 .cpu() 를 부르므로 이 시각은 GPU 동기화 이후다."""

    def __init__(self):
        self.puts = 0
        self.first = None

    def put(self, value):
        self.puts += 1
        if self.puts == 2 and self.first is None:
            self.first = time.perf_counter()

    def end(self):
        pass


class LocalModel(LLMTranslator):
    """LLMTranslator 의 로딩(bnb 4bit nf4, bf16)과 BOS 처리를 그대로 쓰고, 메시지는 공용 템플릿으로 만든다."""

    kind = "local"

    def __init__(self, name: str, model: str, quant: str = "4bit", max_new_tokens: int = 200):
        super().__init__(model_name=model, quant=quant, max_new_tokens=max_new_tokens)
        self.name = name

    def load(self) -> float:
        started = time.perf_counter()
        self._ensure_loaded()
        return time.perf_counter() - started

    def chat_text(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def _stop_ids(self) -> set:
        ids = self._model.generation_config.eos_token_id
        ids = set(ids if isinstance(ids, (list, tuple)) else [ids])
        if self._tokenizer.eos_token_id is not None:
            ids.add(self._tokenizer.eos_token_id)
        if self._tokenizer.pad_token_id is not None:
            ids.add(self._tokenizer.pad_token_id)
        return {i for i in ids if i is not None}

    def translate(self, system: str, user: str, request: dict) -> dict:  # noqa: ARG002
        import torch

        tok, model = self._tokenizer, self._model
        torch.cuda.synchronize()
        started = time.perf_counter()
        # chat template 이 BOS 를 이미 넣는다. 토크나이저가 또 붙이면 gemma 가 <bos><bos> 로 시작한다.
        enc = tok(self.chat_text(system, user), return_tensors="pt", add_special_tokens=False)
        n_prompt = enc["input_ids"].shape[1]
        enc = {k: v.to(self._device) for k, v in enc.items()}
        streamer = _FirstToken()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=tok.eos_token_id, streamer=streamer)
        new = out[0][n_prompt:].tolist()
        raw = tok.decode(new, skip_special_tokens=True)
        torch.cuda.synchronize()
        ended = time.perf_counter()
        stop = self._stop_ids()
        n_out = len(new)
        while n_out and new[n_out - 1] in stop:
            n_out -= 1
        ttft = streamer.first
        return {
            "raw_output": raw,
            "latency_ms": _ms(ended - started),
            "ttft_ms": _ms(ttft - started) if ttft else None,
            "generation_ms": _ms(ended - ttft) if ttft else None,
            "input_tokens": int(n_prompt),
            "output_tokens": n_out,
            "truncated": len(new) >= self.max_new_tokens and new[-1] not in stop,
            "attempts": 1,
            "api_meta": None,
        }


class _Http:
    kind = "api"

    def __init__(self, timeout: float, max_retries: int):
        import httpx

        self.client = httpx.Client(timeout=timeout)
        self.max_retries = max_retries

    def load(self) -> float:
        return 0.0

    def _retry(self, attempt: int, error: str, retry_after) -> None:
        if attempt > self.max_retries:
            raise ApiFailure(f"{error} (gave up after {attempt} attempts)")
        time.sleep(_wait_sec(attempt, retry_after))


class OpenAIChat(_Http):
    """chat completions 스트리밍. TTFT 를 재려고 stream + include_usage 를 쓴다."""

    kind = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, name: str, model: str, key: str, price: tuple, ledger, *,
                 reasoning_effort: str = "none", temperature: float | None = 0,
                 max_completion_tokens: int | None = None, timeout: float = 60.0,
                 max_retries: int = 3):
        super().__init__(timeout, max_retries)
        self.name, self.model, self.price, self.ledger = name, model, tuple(price), ledger
        self.headers = {"Authorization": f"Bearer {key}"}
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def _once(self, body: dict) -> dict:
        started = time.perf_counter()
        first, parts, usage, finish = None, [], {}, None
        with self.client.stream("POST", self.URL, headers=self.headers, json=body) as r:
            if r.status_code >= 400:
                r.read()
                return {"status": r.status_code, "error": r.text[:500],
                        "retry_after": r.headers.get("retry-after")}
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        if first is None and content.strip():
                            first = time.perf_counter()
                        parts.append(content)
                    finish = choice.get("finish_reason") or finish
        ended = time.perf_counter()
        return {"status": 200, "text": "".join(parts), "usage": usage, "finish": finish,
                "latency_ms": _ms(ended - started),
                "ttft_ms": _ms(first - started) if first else None,
                "generation_ms": _ms(ended - first) if first else None}

    def translate(self, system: str, user: str, request: dict, *, tag: dict) -> dict:  # noqa: ARG002
        import httpx

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "reasoning_effort": self.reasoning_effort,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_completion_tokens:
            body["max_completion_tokens"] = self.max_completion_tokens
        attempt = 0
        while True:
            attempt += 1
            try:
                res = self._once(body)
            except (httpx.TransportError, json.JSONDecodeError) as e:
                self._retry(attempt, f"{type(e).__name__}: {e}", None)
                continue
            if res["status"] == 200:
                break
            error = f"HTTP {res['status']}: {res['error']}"
            if res["status"] not in RETRYABLE_STATUS:
                raise ApiFailure(error)
            self._retry(attempt, error, res["retry_after"])
        usage = res["usage"]
        counts = {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {})
            .get("reasoning_tokens", 0) or 0,
        }
        cost = token_cost(self.price, input_tokens=counts["input_tokens"],
                          cached_tokens=counts["cached_tokens"],
                          output_tokens=counts["output_tokens"])
        self.ledger.record(cost=cost, backend="openai", model=self.model, attempts=attempt,
                           usage_missing=not usage, finish_reason=res["finish"], **counts, **tag)
        return {
            "raw_output": res["text"],
            "latency_ms": res["latency_ms"],
            "ttft_ms": res["ttft_ms"],
            "generation_ms": res["generation_ms"],
            "input_tokens": counts["input_tokens"],
            "output_tokens": counts["output_tokens"],
            "truncated": res["finish"] == "length",
            "attempts": attempt,
            "api_meta": {"finish_reason": res["finish"], "cached_tokens": counts["cached_tokens"],
                         "reasoning_tokens": counts["reasoning_tokens"], "cost": cost},
        }


class DeepL(_Http):
    """text = 현재 발화, context = CONTEXT 블록 본문. 현재 발화 화자 표시는 넣을 자리가 없다."""

    kind = "deepl"

    def __init__(self, name: str, key: str, ledger, *, model_type: str = "quality_optimized",
                 usd_per_million_characters: float = 25.0, timeout: float = 60.0,
                 max_retries: int = 5):
        super().__init__(timeout, max_retries)
        self.name, self.ledger, self.model = name, ledger, model_type
        self.free = key.endswith(":fx")
        self.url = (FREE_URL if self.free else PRO_URL) + "/v2/translate"
        self.headers = {"Authorization": f"DeepL-Auth-Key {key}"}
        self.usd_per_million = 0.0 if self.free else usd_per_million_characters

    def translate(self, system: str, user: str, request: dict, *, tag: dict,  # noqa: ARG002
                  context: str) -> dict:
        import httpx

        body = {
            "text": [request["current_utterance"]],
            "source_lang": request["source_language"].upper(),
            "target_lang": TARGET_CODES.get(request["target_language"],
                                            request["target_language"].upper()),
            "model_type": self.model,
            "show_billed_characters": True,
        }
        if context:
            body["context"] = context
        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                r = self.client.post(self.url, headers=self.headers, json=body)
            except httpx.TransportError as e:
                self._retry(attempt, f"{type(e).__name__}: {e}", None)
                continue
            ended = time.perf_counter()
            if r.status_code < 400:
                break
            error = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code not in RETRYABLE_STATUS:
                raise ApiFailure(error)
            self._retry(attempt, error, r.headers.get("retry-after"))
        data = r.json()
        result = (data.get("translations") or [{}])[0]
        billed = int(result.get("billed_characters") or 0)
        cost = billed * self.usd_per_million / 1_000_000
        self.ledger.record(cost=cost, backend="deepl", model=self.model, attempts=attempt,
                           billed_characters=billed, free_key=self.free,
                           model_type_used=result.get("model_type_used"), **tag)
        if "text" not in result:
            raise ApiFailure(f"deepl returned no translation: {data!r}"[:500])
        return {
            "raw_output": result["text"],
            "latency_ms": _ms(ended - started),
            "ttft_ms": None,
            "generation_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "truncated": False,
            "attempts": attempt,
            "api_meta": {"billed_characters": billed,
                         "model_type_used": result.get("model_type_used"),
                         "detected_source_language": result.get("detected_source_language"),
                         "cost": cost},
        }
