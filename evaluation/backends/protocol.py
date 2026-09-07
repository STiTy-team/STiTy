"""STiTy 평가 WebSocket 프로토콜의 백엔드 중립 구현.

`evaluation/harness/stream.py` 가 기대하는 계약만 담는다. ASR 엔진은 주입받는다 —
이 파일은 vllm/transformers/nemo 중 무엇도 import 하지 않는다. 그래야 백엔드마다
다른 conda env 에서 같은 서버를 띄울 수 있다.

하네스가 실제로 쓰는 계약(harness/stream.py 기준):

  연결          → 서버가 {"type":"hello"}
  {"type":"start", lang, targetLang}  → 서버가 {"type":"ready"}
  바이너리 int16 PCM 16kHz 청크 (기본 200ms, 실시간 속도로 들어옴)
  {"type":"finish"} → 남은 final 들 → **{"type":"finish_done"}**
  {"type":"stop"}   → 종료

`finish_done` 은 선택이 아니다. 안 보내면 클라이언트가 파일마다
POST_FINISH_GRACE_SEC(60초)를 통째로 기다린다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional, Protocol

import numpy as np
import websockets

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class StreamingEngine(Protocol):
    """백엔드가 구현해야 하는 전부."""

    def start(self, lang: str) -> None:
        """새 스트림 시작. 내부 상태 초기화."""

    def feed(self, pcm: np.ndarray) -> Optional[str]:
        """float32 mono 16kHz 청크 → 이번에 새로 확정된 텍스트(없으면 None)."""

    def finish(self) -> Optional[str]:
        """스트림 종료. 남은 텍스트 플러시(없으면 None)."""

    def close(self) -> None:
        """자원 해제."""


def _fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


class BackendServer:
    """엔진 하나를 STiTy 평가 프로토콜로 감싸 서빙한다."""

    def __init__(self, engine_factory, host: str = "0.0.0.0", port: int = 8765,
                 backend_name: str = "unknown", config_note: str = ""):
        self.engine_factory = engine_factory
        self.host = host
        self.port = port
        self.backend_name = backend_name
        self.config_note = config_note
        self._engine = None

    # ── 세션 ────────────────────────────────────────────────────────────────
    async def _handle(self, websocket):
        peer = getattr(websocket, "remote_address", None)
        logger.info("client connected: %s", peer)
        await websocket.send(json.dumps({
            "type": "hello",
            "message": f"{self.backend_name} ready",
            "serverConfig": {"backend": self.backend_name, "config": self.config_note},
        }))

        # 엔진은 프로세스당 하나(모델 1회 로드). 세션마다 start()로 상태만 리셋한다.
        if self._engine is None:
            logger.info("loading engine ...")
            t0 = time.perf_counter()
            self._engine = self.engine_factory()
            logger.info("engine loaded in %.1fs", time.perf_counter() - t0)

        state = None
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if state is None:
                        continue
                    await self._on_audio(websocket, state, message)
                    continue

                data = json.loads(message)
                mtype = data.get("type", "")

                if mtype == "start":
                    state = self._new_state(data.get("lang") or "auto")
                    self._engine.start(state["lang"])
                    await websocket.send(json.dumps({"type": "ready", "message": "streaming"}))

                elif mtype == "finish":
                    if state is not None:
                        await self._on_finish(websocket, state)
                        # 같은 연결에서 다음 파일이 이어진다. 상태만 새로 판다.
                        state = None
                    await websocket.send(json.dumps({"type": "finish_done"}))

                elif mtype == "stop":
                    break

                elif mtype in ("log", "tts_log"):
                    pass
        except websockets.exceptions.ConnectionClosed:
            logger.info("client disconnected: %s", peer)
        except Exception:
            logger.exception("session error")
        finally:
            logger.info("session end: %s", peer)

    def _new_state(self, lang: str) -> dict:
        return {
            "lang": lang,
            "t0": time.perf_counter(),
            "samples": 0,
            "segment_id": 0,
            "emitted": [],
        }

    async def _emit_final(self, websocket, state, text: str, reason: str):
        text = (text or "").strip()
        if not text:
            return
        state["segment_id"] += 1
        audio_end = state["samples"] / SAMPLING_RATE
        # 클라이언트가 실시간 속도로 밀어 넣으므로, 벽시계 경과에서 오디오 길이를 빼면
        # "이 확정이 실시간 대비 얼마나 뒤처졌는가" 가 된다. FSL 과 같은 축이다.
        lag = (time.perf_counter() - state["t0"]) - audio_end
        audio_start = state.get("last_audio_end", 0.0)
        state["last_audio_end"] = audio_end
        state["emitted"].append(text)

        await websocket.send(json.dumps({
            "type": "final",
            "original": text,
            "translation": "",
            "language": state["lang"],
            "commitReason": reason,
            "segmentId": state["segment_id"],
            "audioStartSec": audio_start,
            "audioEndSec": audio_end,
            "start": _fmt_hms(audio_start),
            "end": _fmt_hms(audio_end),
            "fsl_sec": max(0.0, lag),
            "seg_audio_sec": max(0.0, audio_end - audio_start),
        }, ensure_ascii=False))

    async def _on_audio(self, websocket, state, payload: bytes):
        pcm = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
        state["samples"] += len(pcm)
        loop = asyncio.get_running_loop()
        # 추론은 블로킹이다. 이벤트 루프를 막으면 수신 버퍼가 밀려 지연 측정이 오염된다.
        text = await loop.run_in_executor(None, self._engine.feed, pcm)
        if text:
            await self._emit_final(websocket, state, text, "seg")

    async def _on_finish(self, websocket, state):
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._engine.finish)
        if text:
            await self._emit_final(websocket, state, text, "finish")

    # ── 기동 ────────────────────────────────────────────────────────────────
    async def _serve(self):
        async with websockets.serve(self._handle, self.host, self.port,
                                    ping_interval=None, ping_timeout=None,
                                    max_size=10 * 1024 * 1024):
            logger.info("%s listening on ws://%s:%d  (%s)",
                        self.backend_name, self.host, self.port, self.config_note)
            await asyncio.Future()

    def run(self):
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass
        finally:
            if self._engine is not None:
                try:
                    self._engine.close()
                except Exception:
                    pass
