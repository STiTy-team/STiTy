"""Voxtral Realtime(vLLM /v1/realtime) 스모크 클라이언트.

채점·데이터 로딩·지연 지표는 smoke_client 를 그대로 쓴다 - 세 백엔드가 한 표에
들어가려면 같은 정의여야 한다.

프로토콜은 vLLM 소스가 계약이다
(`entrypoints/speech_to_text/realtime/{api_router,connection,protocol}.py`):

    connect ws://host:port/v1/realtime?model=<id>
    -> {"type":"session.created"}
    <- {"type":"session.update","model":<id>}        model 은 **최상위**
    <- {"type":"input_audio_buffer.commit"}          생성 시작
    <- {"type":"input_audio_buffer.append","audio":<b64 pcm16>} xN
    <- {"type":"input_audio_buffer.commit","final":true}   오디오 입력 종료
    -> {"type":"transcription.delta","delta":"..."} xN
    -> {"type":"transcription.done","text":...,"usage":...}

확정된 사실 셋:
  1. **commit 은 주기가 아니라 스위치다.** final=False 면 start_generation() 을
     부르고, 이미 돌고 있으면 "Generation already in progress" 로 무시된다.
     주기적으로 보내는 건 의미가 없다 - 바뀌는 건 *생성을 언제 시작하느냐* 뿐이다.
  2. **final=true 를 보내야 끝난다.** 이게 오디오 큐에 종료 센티널을 넣는 유일한
     수단이고(connection.py:157), 안 보내면 스트림이 안 끝나 transcription.done
     이 영영 오지 않는다. 예전 구현이 정적 타임아웃으로 끊던 이유가 이것이다.
  3. **언어 지정이 불가능하다.** session.update 스키마는 type/model 뿐이고
     realtime 프롬프트는 start() + encode_streaming_tokens() 라 언어 토큰이 없다.
     항상 자동 감지다 - 다른 두 백엔드도 auto 로 맞춰야 비교가 성립한다.

delta 는 디코드 스텝마다 오고 대부분 빈 문자열이다(정상 클립도 그렇다).
입력 피크가 대략 0.005 아래면 전부 빈 문자열로 온다 - smoke_client.load_audio 주석.
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
from smoke_client import (SAMPLING_RATE, LAAL_UNIT, laal_pair, load_audio,  # noqa: E402
                          load_fleurs, score, score_pair)

QUIET_SEC = 3.0      # 마지막 delta 이후 이만큼 조용하면 발화 종료로 본다
MAX_WAIT_SEC = 30.0  # 그래도 안 끝나면 포기


async def run_one(url, model, audio, lang, trailing_ms=0, start_delay_sec=0.0):
    """문서 흐름으로 한 클립. start_delay_sec > 0 이면 생성 시작만 그만큼 늦춘다."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    async with websockets.connect(url, max_size=None, open_timeout=60,
                                  ping_interval=None) as ws:
        st = {"text": "", "n": 0, "err": None, "done_at": None,
              "segs_ca": [], "first": None, "last": None, "done_text": None}
        origin = None

        async def reader():
            while True:
                try:
                    ev = json.loads(await ws.recv())
                except Exception:
                    return
                t = ev.get("type")
                if t == "error":
                    st["err"] = ev.get("error")
                    continue
                if t == "transcription.done":
                    st["done_at"] = time.perf_counter()
                    st["done_text"] = ev.get("text")
                    return
                if t != "transcription.delta":
                    continue
                piece = ev.get("delta") or ""
                if not piece:
                    continue            # 스텝마다 오는 빈 델타. 정상이다.
                now = time.perf_counter()
                if st["first"] is None:
                    st["first"] = now
                st["last"] = now
                st["text"] += piece
                st["n"] += 1
                if origin is not None:
                    st["segs_ca"].append((piece, (now - origin) * 1000.0))

        rd = asyncio.create_task(reader())
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await asyncio.sleep(0.3)
        if st["err"]:
            rd.cancel()
            raise RuntimeError("session.update 거절: %s" % st["err"])

        origin = time.perf_counter()
        if start_delay_sec <= 0:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            started = True
        else:
            started = False

        step = int(0.2 * SAMPLING_RATE)
        for i in range(0, len(pcm), step):
            chunk = pcm[i:i + step]
            target = origin + (i + len(chunk)) / SAMPLING_RATE
            while True:
                left = target - time.perf_counter()
                if left <= 0:
                    break
                await asyncio.sleep(min(left, 0.02))
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk.tobytes()).decode(),
            }))
            if not started and (time.perf_counter() - origin) >= start_delay_sec:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                started = True
        t_audio_end = time.perf_counter()
        if not started:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        # 오디오 입력 종료. 이걸 보내야 transcription.done 이 온다.
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        try:
            await asyncio.wait_for(rd, timeout=MAX_WAIT_SEC)
        except asyncio.TimeoutError:
            rd.cancel()

    text = (st["done_text"] if st["done_text"] is not None else st["text"]).strip()
    audio_sec = len(pcm) / SAMPLING_RATE
    return {
        "transcript": text,
        "num_finals": st["n"],
        "segs_policy": [],   # 이 서버는 결정 시점을 주지 않는다 - 정책축 없음
        "segs_ca": list(st["segs_ca"]),
        "n_decision": 0,
        "avg_fsl_sec": None,
        "finalization_lag_sec": round(st["last"] - t_audio_end, 3) if st["last"] else None,
        "done_lag_sec": round(st["done_at"] - t_audio_end, 3) if st["done_at"] else None,
        "rtf": round((st["last"] - origin) / audio_sec, 3) if st["last"] else None,
    }


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
            out = await run_one(url, a.model, audio, a.lang,
                                start_delay_sec=a.start_delay_sec)
        except Exception as e:
            print("  [%d] %s FAILED %s: %s" % (i, r["file_id"], type(e).__name__, e),
                  flush=True)
            continue
        s, s_raw = score_pair(r["reference"], out["transcript"], unit, a.lang)
        _sec = len(audio) / SAMPLING_RATE
        # delta 는 단어 중간에서 끊긴다 - fragments=True (units_with_delays 주석)
        _laal, _laal_ca = laal_pair(out.pop("segs_policy"), out.pop("segs_ca"),
                                    _sec, r["reference"], a.lang, fragments=True)
        results.append({**r, **out, "audio_sec": _sec,
                        "laal_ms": _laal, "laal_ca_ms": _laal_ca,
                        unit: s, unit + "_raw": s_raw})
        print("  [%d/%d] %s=%s" % (i, len(rows), unit,
                                   ("%.3f" % s) if s is not None else "NA"), flush=True)
        print("      REF %s" % r["reference"][:80], flush=True)
        print("      HYP %s" % out["transcript"][:80], flush=True)

    vals = [r[unit] for r in results if r.get(unit) is not None]
    fin = [r["finalization_lag_sec"] for r in results
           if r.get("finalization_lag_sec") is not None]
    dn = [r["done_lag_sec"] for r in results if r.get("done_lag_sec") is not None]
    laal = [r["laal_ms"] for r in results if r.get("laal_ms") is not None]
    laal_ca = [r["laal_ca_ms"] for r in results if r.get("laal_ca_ms") is not None]
    rtfs = [r["rtf"] for r in results if r.get("rtf") is not None]
    audio_total = sum(r["audio_sec"] for r in results) or 1
    summary = {
        "backend": a.tag, "lang": a.lang, "unit": unit,
        "peak_normalize": a.peak_normalize,
        "start_delay_sec": a.start_delay_sec,
        "laal_unit": LAAL_UNIT.get(a.lang, "word"),
        "n_ok": len(results), "n_total": len(rows),
        "avg_" + unit: round(mean(vals), 4) if vals else None,
        "laal_ms": round(mean(laal), 1) if laal else None,
        "laal_ca_ms": round(mean(laal_ca), 1) if laal_ca else None,
        "finalization_lag_sec": round(mean(fin), 3) if fin else None,
        "done_lag_sec": round(mean(dn), 3) if dn else None,
        "rtf": round(mean(rtfs), 3) if rtfs else None,
        "wall_rtf": round((time.perf_counter() - t_start) / audio_total, 3),
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
    ap.add_argument("--trailing-ms", type=int, default=0,
                    help="이 백엔드는 VAD 가 없어 무음 패딩이 필요 없다. 0 이 맞다.")
    ap.add_argument("--peak-normalize", type=float, default=0.0, metavar="PEAK",
                    help="클립별 피크 정규화. Voxtral 은 피크가 대략 0.005 아래면 "
                         "델타는 오는데 텍스트가 전부 빈 문자열이다(smoke_client.load_audio 주석).")
    ap.add_argument("--start-delay-sec", type=float, default=0.0,
                    help="생성 시작(첫 commit)을 이만큼 늦춘다. 0=오디오 전송 전에 시작(문서 흐름). "
                         "commit 은 주기가 아니라 스위치라 '주기' 노브는 존재하지 않는다.")
    ap.add_argument("--tag", default="voxtral")
    ap.add_argument("--out", default="voxtral_smoke.json")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
