"""FLEURS 로 백엔드 서버를 때려보는 최소 클라이언트.

`evaluation/harness/stream.py` 와 같은 프로토콜을 말한다. 하네스를 그대로 쓰지 않는
이유는 하나다 - 하네스의 데이터셋 클라이언트는 LibriSpeech/DailyTalk/KsponSpeech 용이고
FLEURS 는 AST 트랙에서 manifest 로만 쓰인다. 서버 계약을 검증하는 게 목적이므로
의존성 없는 얇은 클라이언트가 낫다. 서버가 검증되면 정식 데이터셋 클라이언트로 승격한다.

채점은 외부 의존성 없이 편집거리로 직접 한다(jiwer 미설치 env 에서도 돌아야 한다).
한국어는 CER, 영어는 WER.

    python smoke_client.py --ws ws://127.0.0.1:8765 --lang ko --limit 20 --out r.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf
import websockets

SAMPLING_RATE = 16000
FLEURS_ROOT = Path.home() / "STiTy-team" / "datasets" / "fleurs" / "data"
LANG_DIR = {"ko": "ko_kr", "en": "en_us"}


# -- 채점 -----------------------------------------------------------------
def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", (text or "").lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(ref: str, hyp: str, unit: str) -> float | None:
    r, h = _norm(ref), _norm(hyp)
    if unit == "cer":
        r, h = r.replace(" ", ""), h.replace(" ", "")
        seq_r, seq_h = list(r), list(h)
    else:
        seq_r, seq_h = r.split(), h.split()
    if not seq_r:
        return None
    return _edit_distance(seq_r, seq_h) / len(seq_r)


# -- 데이터 ---------------------------------------------------------------
def load_fleurs(lang: str, limit: int):
    d = FLEURS_ROOT / LANG_DIR[lang]
    tsv, audio_dir = d / "test.tsv", d / "audio" / "test"
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            wav = audio_dir / p[1]
            if not wav.exists():
                continue
            rows.append({"file_id": p[1].replace(".wav", ""),
                         "path": str(wav), "reference": p[2]})
    rows.sort(key=lambda r: r["file_id"])
    # 결정적 부분집합 - 실행 간 같은 클립을 쓴다.
    return rows[:limit]


# -- 스트리밍 -------------------------------------------------------------
async def run_one(ws, audio: np.ndarray, target_lang: str, trailing_ms: int):
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "start", "lang": "auto", "targetLang": target_lang}))
    # ready 대기
    deadline = time.perf_counter() + 30
    while time.perf_counter() < deadline:
        m = await asyncio.wait_for(ws.recv(), timeout=30)
        if isinstance(m, str) and json.loads(m).get("type") == "ready":
            break

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    step = int(0.2 * SAMPLING_RATE)
    finals, lags, first_at = [], [], None
    first_arr = last_arr = None      # 도착 시각(벽시계). 무음 제외 기준으로 쓴다.
    done = asyncio.Event()

    async def reader():
        nonlocal first_at, first_arr, last_arr
        idle = 0.0
        while True:
            try:
                m = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                if done.is_set():
                    idle += 1.0
                    if idle >= 30:
                        return
                continue
            except Exception:
                return
            idle = 0.0
            if not isinstance(m, str):
                continue
            d = json.loads(m)
            t = d.get("type")
            if t == "finish_done":
                if done.is_set():
                    return
            elif t == "final":
                txt = (d.get("original") or "").strip()
                if txt:
                    _now = time.perf_counter()
                    if first_at is None:
                        first_at = _now - t0
                        first_arr = _now
                    last_arr = _now
                    finals.append(txt)
                    if d.get("fsl_sec") is not None:
                        lags.append(d["fsl_sec"])

    task = asyncio.create_task(reader())
    origin = time.perf_counter()
    for i in range(0, len(pcm), step):
        chunk = pcm[i:i + step]
        target = origin + (i + len(chunk)) / SAMPLING_RATE
        while True:
            left = target - time.perf_counter()
            if left <= 0:
                break
            await asyncio.sleep(min(left, 0.02))
        await ws.send(chunk.tobytes())
    # 실제 오디오가 끝난 시각. 지연은 전부 이 기준으로 잰다 - 뒤에 붙는 무음은
    # VAD 커밋을 유도하려고 우리가 보내는 것이지 발화의 일부가 아니다.
    t_audio_end = time.perf_counter()
    if trailing_ms > 0:
        sil = np.zeros(int(SAMPLING_RATE * trailing_ms / 1000), dtype=np.int16)
        for i in range(0, len(sil), step):
            await ws.send(sil[i:i + step].tobytes())
            await asyncio.sleep(0.2)
    await ws.send(json.dumps({"type": "finish"}))
    done.set()
    await task

    audio_sec = len(pcm) / SAMPLING_RATE
    return {"transcript": " ".join(finals).strip(),
            "num_finals": len(finals),
            "first_token_latency": first_at,
            "avg_fsl_sec": mean(lags) if lags else None,
            "ttfo_sec": round(first_arr - t_audio_end, 3) if first_arr else None,
            "completion_lag_sec": round(last_arr - t_audio_end, 3) if last_arr else None,
            "xrt": round((last_arr - origin) / audio_sec, 3) if last_arr else None,
            "wall_sec": time.perf_counter() - t0}


async def main_async(a) -> int:
    unit = "cer" if a.lang == "ko" else "wer"
    rows = load_fleurs(a.lang, a.limit)
    if not rows:
        print(json.dumps({"error": "no FLEURS rows", "lang": a.lang}))
        return 1
    print(f"[smoke] {a.lang} {len(rows)} clips -> {a.ws} (unit={unit})", flush=True)

    results, t_start = [], time.perf_counter()
    ws = await websockets.connect(a.ws, ping_interval=None, open_timeout=60,
                                  max_size=10 * 1024 * 1024)
    # hello
    try:
        await asyncio.wait_for(ws.recv(), timeout=15)
    except Exception:
        pass

    try:
        for i, r in enumerate(rows, 1):
            audio, sr = sf.read(r["path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLING_RATE:
                n = int(round(len(audio) * SAMPLING_RATE / sr))
                audio = np.interp(np.linspace(0, len(audio) - 1, n),
                                  np.arange(len(audio)), audio).astype(np.float32)
            try:
                out = await run_one(ws, audio, a.lang, a.trailing_ms)
            except Exception as e:
                print(f"  [{i}] {r['file_id']} FAILED {type(e).__name__}: {e}", flush=True)
                try:
                    await ws.close()
                except Exception:
                    pass
                ws = await websockets.connect(a.ws, ping_interval=None, open_timeout=60,
                                              max_size=10 * 1024 * 1024)
                try:
                    await asyncio.wait_for(ws.recv(), timeout=15)
                except Exception:
                    pass
                continue
            s = score(r["reference"], out["transcript"], unit)
            results.append({**r, **out, "audio_sec": len(audio) / SAMPLING_RATE,
                            unit: s})
            print(f"  [{i}/{len(rows)}] {unit}={s:.3f} " if s is not None
                  else f"  [{i}/{len(rows)}] {unit}=NA ", flush=True)
            print(f"      REF {r['reference'][:80]}", flush=True)
            print(f"      HYP {out['transcript'][:80]}", flush=True)
    finally:
        try:
            await ws.send(json.dumps({"type": "stop"}))
            await ws.close()
        except Exception:
            pass

    vals = [r[unit] for r in results if r.get(unit) is not None]
    lat = [r["first_token_latency"] for r in results if r.get("first_token_latency")]
    fsl = [r["avg_fsl_sec"] for r in results if r.get("avg_fsl_sec") is not None]
    ttfo = [r["ttfo_sec"] for r in results if r.get("ttfo_sec") is not None]
    comp = [r["completion_lag_sec"] for r in results if r.get("completion_lag_sec") is not None]
    xrt = [r["xrt"] for r in results if r.get("xrt") is not None]
    audio_total = sum(r["audio_sec"] for r in results) or 1
    summary = {
        "backend": a.tag, "lang": a.lang, "unit": unit,
        "n_ok": len(results), "n_total": len(rows),
        f"avg_{unit}": round(mean(vals), 4) if vals else None,
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
    ap.add_argument("--ws", default="ws://127.0.0.1:8765")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--trailing-ms", type=int, default=1000)
    ap.add_argument("--tag", default="unknown")
    ap.add_argument("--out", default="smoke.json")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
