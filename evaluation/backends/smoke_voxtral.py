"""Voxtral Realtime(OpenAI 호환 WS) 스모크 클라이언트.

채점/데이터 로딩은 smoke_client 를 그대로 쓴다 - 요약 JSON 형식이 같아야
드라이버의 요약 표가 세 백엔드를 한 줄로 비교할 수 있다.

프로토콜(2026-09-09 실측으로 확정):
    connect ws://host:port/v1/realtime?model=<id>
    -> {"type":"session.created"}
    <- {"type":"session.update", "model":<id>, "session":{...}}
       model 은 **최상위**여야 한다. session 안에 넣으면
       "Missing required field: model" 로 거절당한다. 성공해도
       session.updated 같은 확인 이벤트는 오지 않는다(무응답이 정상).
    <- {"type":"input_audio_buffer.append","audio":<base64 pcm16>} xN
    <- {"type":"input_audio_buffer.commit"}
    -> {"type":"transcription.delta","delta":"..."} xN
    종료 이벤트가 없으므로 정적(quiet) 타임아웃으로 끊는다.
    delta 는 디코드 스텝마다 오고 대부분 빈 문자열이다 - 정상 클립도 그렇다.
    입력 피크가 대략 0.005 아래면 **전부** 빈 문자열로 온다(2026-09-11).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_client import SAMPLING_RATE, load_audio, load_fleurs, score  # noqa: E402

QUIET_SEC = 3.0      # 마지막 delta 이후 이만큼 조용하면 발화 종료로 본다
MAX_WAIT_SEC = 30.0  # 그래도 안 끝나면 포기


async def run_one(url, model, audio, lang, trailing_ms=1000, commit_interval=0.0):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    out = {"transcript": "", "first_token_latency": None, "avg_fsl_sec": None,
           "num_finals": 0}
    async with websockets.connect(url, max_size=None, open_timeout=60,
                                  ping_interval=None) as ws:
        state = {"first": None, "last": None, "text": "", "n": 0, "err": None}

        async def reader():
            try:
                while True:
                    ev = json.loads(await ws.recv())
                    t = ev.get("type")
                    if t == "error":
                        state["err"] = ev.get("error")
                        continue
                    piece = None
                    for k in ("delta", "transcript", "text"):
                        v = ev.get(k)
                        if isinstance(v, str) and v:
                            piece = v
                            break
                    if piece is None:
                        continue
                    now = time.perf_counter()
                    if state["first"] is None:
                        state["first"] = now
                    state["last"] = now
                    state["text"] += piece
                    state["n"] += 1
            except Exception:
                pass

        rd = asyncio.create_task(reader())
        await ws.send(json.dumps({
            "type": "session.update",
            "model": model,                       # 최상위여야 한다
            "session": {"input_audio_format": "pcm16", "language": lang},
        }))
        await asyncio.sleep(0.5)
        if state["err"]:
            rd.cancel()
            raise RuntimeError("session.update 거절: %s" % state["err"])

        step = int(0.2 * SAMPLING_RATE)
        t0 = time.perf_counter()
        since_commit = 0.0
        for i in range(0, len(pcm), step):
            chunk = pcm[i:i + step]
            target = t0 + (i + len(chunk)) / SAMPLING_RATE   # 실시간 속도로 흘린다
            while True:
                left = target - time.perf_counter()
                if left <= 0:
                    break
                await asyncio.sleep(min(left, 0.02))
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk.tobytes()).decode(),
            }))
            # 이 프로토콜에서 확정을 유도하는 건 commit 이다. 주기적으로 보내지
            # 않으면 발화가 끝날 때까지 출력이 하나도 오지 않는다(2026-09-10 실측).
            if commit_interval > 0:
                since_commit += len(chunk) / SAMPLING_RATE
                if since_commit >= commit_interval:
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    since_commit = 0.0
        t_audio_end = time.perf_counter()
        # 다른 백엔드와 같은 오디오를 보내기 위해 무음도 같이 흘린다.
        # 지연 기준(t_audio_end)은 무음 앞에서 잡았으므로 비교 축은 유지된다.
        if trailing_ms > 0:
            sil = np.zeros(int(SAMPLING_RATE * trailing_ms / 1000), dtype='<i2')
            for i in range(0, len(sil), step):
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(sil[i:i + step].tobytes()).decode(),
                }))
                await asyncio.sleep(0.2)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        deadline = t_audio_end + MAX_WAIT_SEC
        while time.perf_counter() < deadline:
            await asyncio.sleep(0.25)
            if state["last"] is not None and time.perf_counter() - state["last"] > QUIET_SEC:
                break
        rd.cancel()

    out["transcript"] = state["text"].strip()
    out["num_finals"] = state["n"]
    audio_sec = len(pcm) / SAMPLING_RATE
    if state["first"] is not None:
        out["first_token_latency"] = round(state["first"] - t0, 3)
        out["ttfo_sec"] = round(state["first"] - t_audio_end, 3)
    if state["last"] is not None:
        out["avg_fsl_sec"] = round(state["last"] - t_audio_end, 3)  # 레거시(= completion_lag)
        out["completion_lag_sec"] = round(state["last"] - t_audio_end, 3)
        out["xrt"] = round((state["last"] - t0) / audio_sec, 3)
    return out


async def main_async(a) -> int:
    unit = "cer" if a.lang == "ko" else "wer"
    rows = load_fleurs(a.lang, a.limit)
    if not rows:
        print(json.dumps({"error": "no FLEURS rows", "lang": a.lang}))
        return 1
    ws_base = a.base_url.replace("http://", "ws://").replace("https://", "wss://")
    url = "%s/v1/realtime?model=%s" % (ws_base, a.model)
    print("[smoke] voxtral %s %d clips -> %s (unit=%s)" % (a.lang, len(rows), url, unit),
          flush=True)

    results, t_start = [], time.perf_counter()
    for i, r in enumerate(rows, 1):
        audio = load_audio(r["path"], a.peak_normalize)
        try:
            out = await run_one(url, a.model, audio, a.lang, a.trailing_ms,
                                a.commit_interval_sec)
        except Exception as e:
            print("  [%d] %s FAILED %s: %s" % (i, r["file_id"], type(e).__name__, e),
                  flush=True)
            continue
        s = score(r["reference"], out["transcript"], unit)
        results.append({**r, **out, "audio_sec": len(audio) / SAMPLING_RATE, unit: s})
        print("  [%d/%d] %s=%s" % (i, len(rows), unit,
                                   ("%.3f" % s) if s is not None else "NA"), flush=True)
        print("      REF %s" % r["reference"][:80], flush=True)
        print("      HYP %s" % out["transcript"][:80], flush=True)

    vals = [r[unit] for r in results if r.get(unit) is not None]
    lat = [r["first_token_latency"] for r in results if r.get("first_token_latency")]
    fsl = [r["avg_fsl_sec"] for r in results if r.get("avg_fsl_sec") is not None]
    ttfo = [r["ttfo_sec"] for r in results if r.get("ttfo_sec") is not None]
    comp = [r["completion_lag_sec"] for r in results if r.get("completion_lag_sec") is not None]
    xrt = [r["xrt"] for r in results if r.get("xrt") is not None]
    audio_total = sum(r["audio_sec"] for r in results) or 1
    summary = {
        "backend": a.tag, "lang": a.lang, "unit": unit,
        "commit_interval_sec": a.commit_interval_sec,
        "peak_normalize": a.peak_normalize,
        "n_ok": len(results), "n_total": len(rows),
        "avg_" + unit: round(mean(vals), 4) if vals else None,
        "avg_first_token_latency_sec": round(mean(lat), 3) if lat else None,
        "avg_fsl_sec": round(mean(fsl), 3) if fsl else None,
        "avg_ttfo_sec": round(mean(ttfo), 3) if ttfo else None,
        "avg_completion_lag_sec": round(mean(comp), 3) if comp else None,
        "avg_xrt": round(mean(xrt), 3) if xrt else None,
        "rtf": round((time.perf_counter() - t_start) / audio_total, 3),
        "empty_transcripts": sum(1 for r in results if not r["transcript"]),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": results}, f, ensure_ascii=False, indent=2)
    print("\n[smoke] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if vals else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default="mistralai/Voxtral-Mini-4B-Realtime-2602")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--trailing-ms", type=int, default=1000)
    ap.add_argument("--peak-normalize", type=float, default=0.0, metavar="PEAK",
                    help="클립별 피크 정규화. Voxtral 은 피크가 대략 0.005 아래면 "
                         "델타는 오는데 텍스트가 전부 빈 문자열이다(smoke_client.load_audio 주석).")
    ap.add_argument("--commit-interval-sec", type=float, default=0.0,
                    help="0 이면 끝에 한 번만 commit(증분 출력 없음)")
    ap.add_argument("--tag", default="voxtral")
    ap.add_argument("--out", default="voxtral_smoke.json")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
