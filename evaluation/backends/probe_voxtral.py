"""Voxtral-Mini-4B-Realtime 프로브 + 브리지 시도.

vLLM 이 `/v1/realtime` WebSocket 을 직접 제공한다. 그 이벤트 스키마를 실행해본 적이
없으므로, 이 스크립트는 **먼저 사실을 수집하고** 그 다음에 전사를 시도한다.

수집하는 것:
  1. /v1/models          - 모델이 실제로 떴는지, 이름이 무엇인지
  2. /openapi.json       - realtime 라우트의 정확한 스키마 (가장 중요)
  3. realtime 세션 이벤트 전문 덤프 - 무엇이 오는지 있는 그대로

전사가 실패해도 1~3 이 남으면 브리지를 한 번에 정확히 짤 수 있다.

    python probe_voxtral.py --audio a.wav --base-url http://127.0.0.1:8000 --out probe.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time

import numpy as np

SAMPLING_RATE = 16000


def _load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLING_RATE:
        import math
        # 의존성을 늘리지 않으려고 선형 보간으로 리샘플한다. 프로브 용도라 충분하다.
        n = int(round(len(audio) * SAMPLING_RATE / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


async def _http_json(url: str):
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            txt = await r.text()
            try:
                return json.loads(txt)
            except Exception:
                return {"_raw": txt[:4000]}


def _summarize_openapi(spec: dict) -> dict:
    """realtime 관련 경로/스키마만 추린다. 전문은 너무 크다."""
    out = {"realtime_paths": {}, "realtime_schemas": []}
    for path, item in (spec.get("paths") or {}).items():
        if "realtime" in path.lower() or "transcription" in path.lower():
            out["realtime_paths"][path] = list(item.keys())
    comps = (spec.get("components") or {}).get("schemas") or {}
    for name in comps:
        low = name.lower()
        if "realtime" in low or "transcription" in low or "audio" in low:
            out["realtime_schemas"].append(name)
    out["all_paths"] = sorted((spec.get("paths") or {}).keys())
    return out


async def _realtime_session(ws_url: str, audio: np.ndarray, model: str,
                            lang: str, max_events: int = 400) -> dict:
    """OpenAI Realtime 형태를 가정하고 붙어본다. 오는 이벤트는 전부 기록한다."""
    import websockets

    rec = {"ws_url": ws_url, "sent": [], "events": [], "error": None,
           "transcript_guess": ""}
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    try:
        async with websockets.connect(ws_url, ping_interval=None,
                                      open_timeout=60, max_size=20 * 1024 * 1024) as ws:
            async def _reader():
                try:
                    while len(rec["events"]) < max_events:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        if isinstance(raw, bytes):
                            rec["events"].append({"binary_len": len(raw)})
                            continue
                        try:
                            ev = json.loads(raw)
                        except Exception:
                            rec["events"].append({"unparsed": raw[:500]})
                            continue
                        # 전문을 다 남기면 커진다. 타입 + 텍스트로 보이는 필드만.
                        slim = {"type": ev.get("type")}
                        for k in ("delta", "text", "transcript", "error", "item_id"):
                            if k in ev:
                                slim[k] = ev[k]
                        if len(rec["events"]) < 15:
                            slim["_full_keys"] = sorted(ev.keys())
                        rec["events"].append(slim)
                        for k in ("delta", "transcript", "text"):
                            v = ev.get(k)
                            if isinstance(v, str) and v:
                                rec["transcript_guess"] += v
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    rec["events"].append({"reader_error": f"{type(e).__name__}: {e}"})

            reader = asyncio.create_task(_reader())

            async def send(obj):
                rec["sent"].append(obj.get("type"))
                await ws.send(json.dumps(obj))

            # 세션 설정 - 필드명이 틀려도 서버가 error 이벤트로 알려줄 것이고,
            # 그 error 가 곧 정답 스키마다.
            await send({
                "type": "session.update",
                "session": {
                    "model": model,
                    "input_audio_format": "pcm16",
                    "language": lang,
                    "temperature": 0.0,
                },
            })
            await asyncio.sleep(1.0)

            # 200ms 씩 실시간 속도로
            step = int(0.2 * SAMPLING_RATE)
            t0 = time.perf_counter()
            for i in range(0, len(pcm16), step):
                chunk = pcm16[i:i + step]
                target = t0 + (i + len(chunk)) / SAMPLING_RATE
                while True:
                    left = target - time.perf_counter()
                    if left <= 0:
                        break
                    await asyncio.sleep(min(left, 0.02))
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk.tobytes()).decode(),
                }))
            rec["sent"].append("input_audio_buffer.append xN")
            await send({"type": "input_audio_buffer.commit"})
            await asyncio.sleep(8.0)
            reader.cancel()
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


async def _run(args) -> int:
    report = {"base_url": args.base_url, "audio": args.audio, "lang": args.lang}

    report["models"] = await _http_json(f"{args.base_url}/v1/models")
    try:
        model_name = report["models"]["data"][0]["id"]
    except Exception:
        model_name = args.model
    report["model_name"] = model_name

    spec = await _http_json(f"{args.base_url}/openapi.json")
    report["openapi"] = _summarize_openapi(spec) if isinstance(spec, dict) else spec

    audio = _load_audio(args.audio)
    report["audio_sec"] = round(len(audio) / SAMPLING_RATE, 2)

    ws_base = args.base_url.replace("http://", "ws://").replace("https://", "wss://")
    # 후보 경로를 순서대로 시도한다. openapi 에서 찾은 게 있으면 그걸 먼저.
    candidates = []
    for p in (report.get("openapi") or {}).get("realtime_paths", {}):
        candidates.append(p)
    for p in ("/v1/realtime", "/v1/realtime/transcription", "/realtime"):
        if p not in candidates:
            candidates.append(p)

    report["attempts"] = []
    for path in candidates[:4]:
        url = f"{ws_base}{path}?model={model_name}"
        res = await _realtime_session(url, audio, model_name, args.lang)
        report["attempts"].append(res)
        if res.get("transcript_guess"):
            report["transcript"] = res["transcript_guess"]
            break

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "attempts"},
                     ensure_ascii=False, indent=2)[:4000])
    print(f"\n[probe] 전문 저장: {args.out}")
    return 0 if report.get("transcript") else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="mistralai/Voxtral-Mini-4B-Realtime-2602")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--out", default="voxtral_probe.json")
    args = ap.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
