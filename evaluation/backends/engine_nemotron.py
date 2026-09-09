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
    """cache-aware streaming (nemotron-3.5-asr-streaming).

    정식 스트리밍 API 는 청크마다 generate() 를 부르는 것이 **아니다**.
    mel 청크를 내놓는 제너레이터를 input_features 로 한 번 넘기면 generate() 가
    스트림을 소비하며 encoder/decoder 캐시를 내부에서 이어간다
    (generation_nemotron_asr_streaming.py:212-225).

    청크마다 generate() 를 부르면 캐시가 매번 리셋돼 경계에서 단어가 통째로 빠진다.
    2026-09-09 실측: 재입국 충격은 신혼 단계가 없기 때문에 -> 매입국 충격 신혼단계가 때문에.
    제너레이터로 바꾼 뒤 같은 문장이 참조와 거의 일치했다.

    청크 크기는 샘플이 아니라 **mel 프레임 수**로 못박혀 있다:
        first = 1 + subsampling_factor * right,  subsequent = subsampling_factor * (right + 1)
    프로세서의 num_samples_first_audio_chunk 는 첫 청크에서 1 프레임 과다하므로
    (105 요구 / 106 산출) 프레임 수에서 역산한다. per 쪽 프로퍼티는 정확하다.

    generate() 는 스트림을 다 소비한 뒤에야 반환하므로 feed() 는 부분 결과를 내지
    않는다. 전체 전사는 finish() 가 돌려준다.
    """

    HOP = 160
    SUPPORTED_LOOKAHEAD = (0, 3, 6, 13)

    def __init__(self, model_id: str = MODEL_ID, right_context: int = 13,
                 lang: str = "ko-KR", device: str = "cuda", **_ignored):
        import queue as _queue
        import threading as _threading
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        self.torch = torch
        self._queue_mod = _queue
        self._threading = _threading
        self.model_id = model_id
        self.default_lang = lang

        if right_context not in self.SUPPORTED_LOOKAHEAD:
            logger.warning("right_context=%s 미지원 - 13 으로 진행 (지원: %s)",
                           right_context, self.SUPPORTED_LOOKAHEAD)
            right_context = 13
        self.lookahead = right_context

        logger.info("loading %s (num_lookahead_tokens=%s)", model_id, self.lookahead)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.set_num_lookahead_tokens(self.lookahead)
        self.model = AutoModelForRNNT.from_pretrained(model_id, device_map=device)
        self.model.eval()

        self.latency_ms = self.processor.streaming_latency_ms
        self.first_samples = (self.processor.num_mel_frames_first_audio_chunk - 1) * self.HOP
        self.chunk_samples = self.processor.num_samples_per_audio_chunk

        self._lang = lang
        self._q = None
        self._thread = None
        self._error = None
        self._text = ""
        self._buf = np.zeros(0, dtype=np.float32)
        self._first = True

    # -- StreamingEngine ---------------------------------------------------
    def start(self, lang: str) -> None:
        self._abort()
        if lang and lang != "auto":
            self._lang = {"ko": "ko-KR", "en": "en-US"}.get(lang, lang)
        else:
            self._lang = self.default_lang
        self._q = self._queue_mod.Queue(maxsize=64)
        self._thread = None
        self._error = None
        self._text = ""
        self._buf = np.zeros(0, dtype=np.float32)
        self._first = True

    def feed(self, pcm: np.ndarray) -> Optional[str]:
        if self._q is None:
            self.start(self._lang)
        self._buf = np.concatenate([self._buf, np.asarray(pcm, dtype=np.float32)])
        while True:
            n = self.first_samples if self._first else self.chunk_samples
            if len(self._buf) < n:
                break
            chunk = self._buf[:n]
            self._buf = self._buf[n:]
            self._push(chunk, first=self._first)
            self._first = False
        return None  # generate() 는 스트림을 다 먹은 뒤에야 텍스트를 준다

    def finish(self) -> Optional[str]:
        if self._q is None:
            return None
        if len(self._buf) > 0:
            n = self.first_samples if self._first else self.chunk_samples
            tail = np.concatenate([self._buf, np.zeros(max(0, n - len(self._buf)), dtype=np.float32)])[:n]
            self._push(tail, first=self._first)
            self._first = False
        self._buf = np.zeros(0, dtype=np.float32)
        if self._thread is not None:
            self._q.put(None)              # 스트림 끝 -> 제너레이터 종료 -> generate 반환
            self._thread.join(timeout=120)
            if self._thread.is_alive():
                logger.error("generate 스레드가 120s 안에 끝나지 않았다")
        self._q = None
        self._thread = None
        if self._error is not None:
            raise self._error
        return self._text or None

    def close(self) -> None:
        self._abort()
        try:
            del self.model
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    # -- 내부 --------------------------------------------------------------
    def _prompt_ids(self):
        probe = self.processor(np.zeros(self.first_samples, dtype=np.float32),
                               sampling_rate=SAMPLING_RATE, language=self._lang,
                               is_streaming=True, is_first_audio_chunk=True,
                               return_tensors="pt")
        return probe["prompt_ids"].to(self.model.device)

    def _features(self, chunk: np.ndarray, first: bool):
        out = self.processor(chunk, sampling_rate=SAMPLING_RATE, language=self._lang,
                             is_streaming=True, is_first_audio_chunk=first,
                             return_tensors="pt")
        return out["input_features"].to(self.model.device)

    def _push(self, chunk: np.ndarray, first: bool) -> None:
        feats = self._features(chunk, first)
        self._ensure_thread()
        self._q.put(feats)

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        prompt_ids = self._prompt_ids()
        q = self._q

        def _stream():
            while True:
                item = q.get()
                if item is None:
                    return
                yield item

        def _run():
            try:
                with self.torch.no_grad():
                    out = self.model.generate(input_features=_stream(),
                                              num_lookahead_tokens=self.lookahead,
                                              prompt_ids=prompt_ids)
                text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
                self._text = (text or "").strip()
            except Exception as e:                      # noqa: BLE001
                logger.exception("nemotron generate 실패")
                self._error = e

        self._thread = self._threading.Thread(target=_run, name="nemotron-generate", daemon=True)
        self._thread.start()

    def _abort(self) -> None:
        if self._q is not None and self._thread is not None:
            try:
                self._q.put_nowait(None)
            except Exception:
                pass
            self._thread.join(timeout=10)
        self._q = None
        self._thread = None


def _probe(path: str, lang: str, right_context: int) -> int:
    """어댑터가 틀렸을 때 한 번에 고칠 수 있게 실제 API 표면을 남긴다."""
    report = {"model_id": MODEL_ID, "audio": path, "lang": lang,
              "right_context": right_context, "steps": []}

    def _json_safe(v):
        # load_processor/load_model 은 객체를 반환한다. 그대로 report 에 넣으면
        # 마지막 json.dumps 에서 통째로 죽는다(2026-09-09 런이 여기서 날아갔다).
        try:
            json.dumps(v, ensure_ascii=False)
            return v
        except TypeError:
            return "<%s>" % type(v).__name__

    def step(name, fn):
        try:
            v = fn()
            report["steps"].append({"step": name, "ok": True, "value": _json_safe(v)})
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
