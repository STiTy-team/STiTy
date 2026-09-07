"""nvidia/nemotron-3.5-asr-streaming-0.6b 엔진 (in-process, transformers).

요구: transformers >= 5.13.0  → stity env(4.57.6)와 충돌하므로 asr-nemotron env 전용.

Cache-Aware FastConformer + RNN-T. att_context_size = [left, right] 이고 단위는 80ms 프레임:
    [56, 0]  →  80ms      [56, 1]  → 160ms     [56, 3]  → 320ms
    [56, 6]  → 560ms      [56, 13] → 1.12s

RNN-T 는 단조 증가라 이전 출력을 수정하지 않는다. Qwen3-ASR(LLM 재디코딩)과 달리
중복/수정 가드가 필요 없다 — feed() 가 돌려주는 텍스트는 항상 "새로 붙은 것"이다.

probe 모드:
    python engine_nemotron.py --probe /path/to.wav --lang ko-KR
API 를 실행해본 적 없이 작성했으므로, 실패해도 무엇이 실제 시그니처인지 남기고 끝낸다.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
SAMPLING_RATE = 16000

# right-context -> 청크 길이(초)
CHUNK_BY_RIGHT = {0: 0.08, 1: 0.16, 3: 0.32, 6: 0.56, 13: 1.12}


class NemotronEngine:
    def __init__(self, model_id: str = MODEL_ID, right_context: int = 13,
                 lang: str = "ko-KR", lookahead_tokens: int = 6, device: str = "cuda"):
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.right_context = right_context
        self.att_context_size = [56, right_context]
        self.default_lang = lang
        self.lookahead_tokens = lookahead_tokens

        logger.info("loading %s (att_context_size=%s)", model_id, self.att_context_size)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForRNNT.from_pretrained(model_id, device_map=device)
        self.model.eval()

        if hasattr(self.processor, "set_num_lookahead_tokens"):
            self.processor.set_num_lookahead_tokens(lookahead_tokens)
        if hasattr(self.model, "set_att_context_size"):
            self.model.set_att_context_size(self.att_context_size)
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "set_default_att_context_size"):
            self.model.encoder.set_default_att_context_size(self.att_context_size)
        else:
            logger.warning("att_context_size 를 설정할 훅을 못 찾음 - 모델 기본값으로 진행")

        # 청크 길이는 right context 가 정한다. 클라이언트가 200ms 로 밀어 넣으므로
        # 여기서 다시 모아 정확한 경계로 잘라 넣는다.
        self.chunk_sec = CHUNK_BY_RIGHT.get(right_context, 1.12)
        self.chunk_samples = int(self.chunk_sec * SAMPLING_RATE)

        self._buf = np.zeros(0, dtype=np.float32)
        self._first = True
        self._lang = lang
        self._prev_text = ""
        self._state = None

    # -- StreamingEngine ---------------------------------------------------
    def start(self, lang: str) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._first = True
        self._prev_text = ""
        self._state = None
        if lang and lang != "auto":
            self._lang = {"ko": "ko-KR", "en": "en-US"}.get(lang, lang)
        else:
            self._lang = self.default_lang

    def feed(self, pcm: np.ndarray) -> Optional[str]:
        self._buf = np.concatenate([self._buf, pcm.astype(np.float32)])
        out = []
        while len(self._buf) >= self.chunk_samples:
            chunk = self._buf[: self.chunk_samples]
            self._buf = self._buf[self.chunk_samples:]
            text = self._infer(chunk, last=False)
            if text:
                out.append(text)
        return " ".join(out) if out else None

    def finish(self) -> Optional[str]:
        text = None
        if len(self._buf) > 0:
            pad = self.chunk_samples - len(self._buf)
            chunk = np.concatenate([self._buf, np.zeros(max(0, pad), dtype=np.float32)])
            text = self._infer(chunk, last=True)
            self._buf = np.zeros(0, dtype=np.float32)
        self._first = True
        return text

    def close(self) -> None:
        try:
            del self.model
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    # -- 내부 --------------------------------------------------------------
    def _infer(self, chunk: np.ndarray, last: bool) -> Optional[str]:
        kwargs = dict(
            sampling_rate=SAMPLING_RATE,
            language=self._lang,
            is_streaming=True,
            is_first_audio_chunk=self._first,
            return_tensors="pt",
        )
        if last:
            kwargs["is_last_audio_chunk"] = True
        try:
            inputs = self.processor(chunk, **kwargs)
        except TypeError:
            # is_last_audio_chunk 를 안 받는 버전
            kwargs.pop("is_last_audio_chunk", None)
            inputs = self.processor(chunk, **kwargs)

        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        self._first = False

        with self.torch.no_grad():
            gen_kwargs = {}
            if self._state is not None:
                gen_kwargs["state"] = self._state
            try:
                out = self.model.generate(**inputs, **gen_kwargs)
            except TypeError:
                out = self.model.generate(**inputs)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            out, self._state = out

        try:
            text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        except Exception:
            text = self.processor.batch_decode(out)[0]

        text = (text or "").strip()
        if not text:
            return None
        # RNN-T 는 수정하지 않지만, 구현에 따라 누적 텍스트를 돌려줄 수 있다.
        # 접두사면 새로 붙은 부분만 취한다.
        if self._prev_text and text.startswith(self._prev_text):
            new = text[len(self._prev_text):].strip()
            self._prev_text = text
            return new or None
        self._prev_text = text
        return text


def _probe(path: str, lang: str, right_context: int) -> int:
    """어댑터가 틀렸을 때 한 번에 고칠 수 있게 실제 API 표면을 남긴다."""
    report = {"model_id": MODEL_ID, "audio": path, "lang": lang,
              "right_context": right_context, "steps": []}

    def step(name, fn):
        try:
            v = fn()
            report["steps"].append({"step": name, "ok": True, "value": v})
            return v
        except Exception as e:
            report["steps"].append({"step": name, "ok": False,
                                    "error": f"{type(e).__name__}: {e}"})
            return None

    import transformers
    step("transformers_version", lambda: transformers.__version__)

    from transformers import AutoModelForRNNT, AutoProcessor
    proc = step("load_processor", lambda: AutoProcessor.from_pretrained(MODEL_ID))
    if proc is not None:
        step("processor_class", lambda: type(proc).__name__)
        step("processor_call_signature", lambda: str(inspect.signature(proc.__call__)))
        step("processor_public_methods",
             lambda: [m for m in dir(proc) if not m.startswith("_")][:60])

    model = step("load_model", lambda: AutoModelForRNNT.from_pretrained(MODEL_ID, device_map="cuda"))
    if model is not None:
        step("model_class", lambda: type(model).__name__)
        step("generate_signature", lambda: str(inspect.signature(model.generate)))
        step("att_context_hooks", lambda: [m for m in dir(model) if "att_context" in m.lower()]
             + [f"encoder.{m}" for m in dir(getattr(model, "encoder", object())) if "att_context" in m.lower()])

    # 실제 전사 1회
    def _transcribe():
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        eng = NemotronEngine(right_context=right_context, lang=lang)
        eng.start(lang)
        t0 = time.perf_counter()
        parts = []
        step_n = int(0.2 * SAMPLING_RATE)
        for i in range(0, len(audio), step_n):
            t = eng.feed(audio[i:i + step_n])
            if t:
                parts.append(t)
        t = eng.finish()
        if t:
            parts.append(t)
        dur = len(audio) / sr
        el = time.perf_counter() - t0
        return {"transcript": " ".join(parts), "audio_sec": round(dur, 2),
                "wall_sec": round(el, 2), "rtf": round(el / dur, 3) if dur else None}

    step("streaming_transcribe", _transcribe)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["steps"] and report["steps"][-1].get("ok") else 1


def main():
    logging.basicConfig(format="%(levelname)s\t%(message)s", level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", metavar="WAV")
    ap.add_argument("--lang", default="ko-KR")
    ap.add_argument("--right-context", type=int, default=13)
    args = ap.parse_args()
    if args.probe:
        sys.exit(_probe(args.probe, args.lang, args.right_context))
    ap.error("--probe 없이는 할 일이 없다. 서버는 server.py 로 띄운다.")


if __name__ == "__main__":
    main()
