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

    generate() 자체는 스트림을 다 소비한 뒤에야 반환하지만, 토큰은 그 전에
    이미 나온다. 베이스 GenerationMixin._sample 이 스텝마다 streamer.put(next_tokens)
    을 부르므로(transformers/generation/utils.py:2933) streamer 를 끼우면
    feed() 가 증분 텍스트를 돌려줄 수 있다. 2026-09-11 이전 구현은 streamer 없이
    finish() 에서 한 번에 뱉어 클립당 조각이 1 개였고, 그래서 이 백엔드만
    "지연 꼴찌" 로 찍혔다 - 모델이 아니라 어댑터 문제였다.

    RNNT 라 토큰 열에 blank 가 대량으로 섞인다. 델타는 토큰을 따로 디코드해
    이어붙이지 않고 **누적 디코드 후 접두사 차이**로 뽑는다(blank/특수토큰이
    skip_special_tokens 로 사라지고, 부분 토큰 경계에서 깨지지 않는다).
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
        self._ids = []
        self._emitted = ""
        self._ids_lock = _threading.Lock()

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
        self._ids = []
        self._emitted = ""
        self._ids_lock = self._threading.Lock()

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
        return self._drain()

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
        tail = self._drain(flush=True)
        if not self._emitted and self._text:
            # 스트리머가 한 번도 안 불린 경우에만 최종 시퀀스로 대체한다.
            # 여기서 self._text 와 self._emitted 를 '조정' 하려 들면 안 된다:
            # 둘은 같은 greedy 디코드 결과지만 공백 처리가 달라(strip vs lstrip)
            # 문자열 비교가 어긋나고, 그러면 전사 전체가 한 번 더 붙는다
            # (2026-09-11 실측: CER 1.245, HYP 가 정확히 두 번 반복됐다).
            tail = self._text
            self._emitted = self._text
        return tail or None

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

        streamer = _TokenStreamer(self)

        def _run():
            try:
                with self.torch.no_grad():
                    out = self.model.generate(input_features=_stream(),
                                              num_lookahead_tokens=self.lookahead,
                                              prompt_ids=prompt_ids,
                                              streamer=streamer)
                text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
                self._text = (text or "").strip()
            except Exception as e:                      # noqa: BLE001
                logger.exception("nemotron generate 실패")
                self._error = e

        self._thread = self._threading.Thread(target=_run, name="nemotron-generate", daemon=True)
        self._thread.start()

    def _drain(self, flush: bool = False) -> Optional[str]:
        """streamer 가 쌓은 토큰을 디코드해 아직 안 내보낸 부분만 돌려준다.

        **단어 경계까지만** 내보낸다. 소비자(smoke_client)는 final 들을
        " ".join 으로 잇는데, 단어 중간에서 끊으면 그 자리에 공백이 생겨
        멀쩡한 전사가 깨진다(2026-09-11: "lag be hind", "ma de", "depen ding"
        만으로 en WER 0.0959 -> 0.1892). 반 단어는 자막으로 띄울 수도 없다.
        flush=True(finish 경로)면 꼬리까지 전부 내보낸다.
        """
        with self._ids_lock:
            ids = list(self._ids)
        if not ids:
            return None
        try:
            text = self.processor.batch_decode([ids], skip_special_tokens=True)[0]
        except Exception:                                   # noqa: BLE001
            return None
        text = (text or "").lstrip()
        if not text.startswith(self._emitted):
            # RNNT 는 단조 증가라 보통 여기 안 온다. 와도 조용히 리셋하지 않고
            # 전체를 새 기준으로 삼는다(중복 출력이 침묵보다 낫다).
            self._emitted = text
            return text.strip() or None
        pending = text[len(self._emitted):]
        if flush:
            self._emitted = text
            return pending.strip() or None
        cut = pending.rfind(" ")
        if cut < 0:
            return None                     # 아직 단어 하나도 안 끝났다
        self._emitted += pending[:cut + 1]  # 공백까지 소비 - 다음 델타는 단어 첫 글자부터
        return pending[:cut].strip() or None

    def _put_ids(self, ids) -> None:
        with self._ids_lock:
            self._ids.extend(ids)

    def _abort(self) -> None:
        if self._q is not None and self._thread is not None:
            try:
                self._q.put_nowait(None)
            except Exception:
                pass
            self._thread.join(timeout=10)
        self._q = None
        self._thread = None


class _TokenStreamer:
    """generate() 가 스텝마다 부르는 훅. BaseStreamer 상속은 필요 없다 - put/end 면 된다."""

    def __init__(self, engine: "NemotronEngine"):
        self.engine = engine

    def put(self, value) -> None:
        try:
            ids = value.reshape(-1).tolist() if hasattr(value, "reshape") else list(value)
        except Exception:                                   # noqa: BLE001
            return
        self.engine._put_ids(int(i) for i in ids)

    def end(self) -> None:
        pass


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
