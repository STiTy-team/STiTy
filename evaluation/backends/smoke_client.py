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


# -- 지연 지표 ------------------------------------------------------------
# 자체 지표(ttfo/completion_lag)를 쓰지 않는다. 레포에 이미 LAAL 구현이 있고
# AST 서버의 `decisionAudioSec` 이 그 d_i 로 쓰라고 붙어 있다
# (streaming_websocket_server_ast.py 헤더). 자체 지표는 클립 길이에 지배돼
# 시스템 지연이 아니라 오디오 길이를 재는 값이었다.
#
#   LAAL     d = 커밋을 결정한 시점까지 읽은 소스 오디오 길이 (정책, 하드웨어 무관)
#   LAAL_CA  d = 클라이언트가 그 출력을 받은 실시간 경과       (계산 비용 포함)
#
# Qwen3 는 서버가 decisionAudioSec 을 주므로 LAAL 에 그대로 쓴다. Voxtral/Nemotron
# 은 그 필드가 없지만 **클립을 1배속으로 흘리므로** 도착 경과가 곧 읽은 오디오
# 길이다 — 같은 축이다. 이 경우 LAAL 과 LAAL_CA 의 차이는 계산 지연뿐이다.
sys.path.insert(0, "/home/skkai/STiTy-team/STiTy")
try:
    from evaluation.ast.metrics_ast import laal_for_utterance
except Exception:                                       # noqa: BLE001
    laal_for_utterance = None

LAAL_UNIT = {"ko": "char", "en": "word"}     # ko 는 어절 개념이 약해 char


def units_with_delays(pieces, unit="word", fragments=False):
    """[(조각, 지연ms)] -> [(완성된 단위들, 지연ms)]

    `fragments=True` 는 조각이 **단어 중간에서 끊길 수 있다**는 뜻이다(Voxtral
    델타). 조각을 그대로 세면 구두점만 있는 조각이나 파편이 각각 한 단위로
    잡혀 |Y_hyp| 가 부푼다 — 실측: 26조각 -> 24단위, 실제 문장은 19단어.
    그러면 gamma = T/max(|Y_hyp|,|Y_ref|) 와 tau 가 어긋나 이 백엔드만 LAAL 이
    왜곡된다. 누적 텍스트에서 **완성된 단위**에만 그 시점의 지연을 붙인다.

    Qwen3/Nemotron 의 final 은 그 자체로 완성된 세그먼트이므로 변환하지 않는다
    (`fragments=False`). 거기에 경계 로직을 걸면 매 세그먼트의 마지막 단어가
    부당하게 다음으로 밀린다.
    """
    if not fragments or unit != "word":
        return list(pieces)
    acc, delays = "", []          # delays[i] = 이 단어에 마지막으로 글자를 보탠 조각의 지연
    for text, d in pieces:
        prev = acc.split()
        # 직전 단어가 이어지는 건 누적이 공백으로 안 끝나고 **이번 조각도 공백으로
        # 시작하지 않을 때**뿐이다. 앞 조건만 보면 " due" 같은 조각이 직전 단어를
        # 이어받은 것으로 처리돼 그 단어의 시각을 덮어쓴다.
        open_tail = (bool(acc) and not acc[-1].isspace() and len(prev) > 0
                     and bool(text) and not text[0].isspace())
        acc += text
        words = acc.split()
        start = (len(prev) - 1) if open_tail else len(prev)
        for k in range(start, len(words)):
            if k < len(delays):
                delays[k] = d          # 이어 붙었으면 시각을 갱신
            else:
                delays.append(d)
    words = acc.split()
    # 지연은 **그 단어의 마지막 글자가 온 시각**이다. 경계를 확인한 다음 조각의
    # 시각을 쓰면 한 조각만큼 늦게 잡혀 이 백엔드만 불리해진다.
    return [(w, delays[k]) for k, w in enumerate(words) if k < len(delays)]


def partial_axes(snaps, final_text, unit):
    """partial 스냅숏들 -> (first_display, stable) 단위 리스트.

    partial 은 누적 텍스트를 **통째로 교체**한다(서버 주석: 모델이 롤백 후
    재디코딩하므로 append 로는 정합성을 못 맞춘다). 그래서 단위 i 의 시각은
      first  = i 번째 단위가 처음 존재하게 된 스냅숏의 시각
      stable = i 번째 단위가 최종본과 같은 값이 된 **마지막** 시각
                (그 뒤로 다시 바뀌면 굳지 않은 것으로 본다)
    로 잡는다. stable 은 "화면에 뜬 글자를 믿어도 되는 시각"이고, first 는
    "무엇이든 떴다"는 시각이다. 둘 다 내야 수정 가능한 출력(Qwen3 partial)과
    append-only 출력(Voxtral 델타, Nemotron RNN-T)을 공정하게 견준다.
    """
    if not snaps or not final_text:
        return [], []
    seq = (lambda s: list(s.replace(" ", ""))) if unit == "char" else (lambda s: s.split())
    fin = seq(final_text)
    first = []                      # first[i] = 처음 뜬 시각
    stable = [None] * len(fin)
    for txt, d in snaps:
        cur = seq(txt)
        while len(first) < len(cur):
            first.append(d)
        for i in range(min(len(cur), len(fin))):
            if cur[i] == fin[i]:
                if stable[i] is None:
                    stable[i] = d
            else:
                stable[i] = None    # 다시 틀어졌다
    last = snaps[-1][1]
    segs_first = [(u, first[i]) for i, u in enumerate(fin) if i < len(first)]
    segs_stable = [(u, stable[i] if stable[i] is not None else last)
                   for i, u in enumerate(fin)]
    return segs_first, segs_stable


def laal_pair(segments_policy, segments_ca, src_sec, ref_text, lang,
              fragments=False):
    """(LAAL, LAAL_CA) ms. segments = [(text, delay_ms)].

    LAAL    d = 커밋 결정 시점까지 읽은 소스 오디오 길이. **서버가 그 값을 주는
            백엔드에서만 잰다** (지금은 Qwen3 뿐). 없으면 None.
    LAAL_CA d = 그 출력을 받은 실시간 경과. 세 백엔드 모두 같은 정의라
            **백엔드 비교는 이 축으로 한다.**
    """
    if laal_for_utterance is None or src_sec <= 0:
        return None, None
    unit = LAAL_UNIT.get(lang, "word")
    ca = units_with_delays(segments_ca, unit, fragments)
    b = laal_for_utterance(ca, src_sec * 1000.0, ref_text, unit)
    a = None
    if segments_policy:
        pol = units_with_delays(segments_policy, unit, fragments)
        a = laal_for_utterance(pol, src_sec * 1000.0, ref_text, unit)
    return (round(a, 1) if a is not None else None,
            round(b, 1) if b is not None else None)


# -- 오디오 로딩 ----------------------------------------------------------
def load_audio(path: str, peak: float = 0.0) -> np.ndarray:
    """wav -> 16k mono float32. peak>0 이면 클립별 피크 정규화.

    FLEURS 는 클립별 녹음 레벨이 50dB 넘게 벌어져 있다(en test 기준 피크
    0.002 ~ 0.75). Qwen3/Nemotron 은 특징 추출 단계에서 발화 단위 정규화를
    하므로 영향이 없지만, Voxtral Realtime 은 입력 게인을 그대로 받아
    피크가 대략 0.005 아래면 **아무 것도 내놓지 않는다**(2026-09-11 실측:
    빈 전사 4건이 전부 en 최저 레벨 클립이었고, 게인만 올리면 4건 모두
    정상 전사됐다. 피크 0.1/0.5/0.95 에서 전사가 완전히 동일해 임계값
    문제이지 게인 민감도가 아니다).

    따라서 레벨은 백엔드별 특성이 아니라 **입력 조건**으로 취급하고, 세
    백엔드에 같은 정규화를 적용한 오디오를 먹인다.
    """
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLING_RATE:
        n = int(round(len(audio) * SAMPLING_RATE / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    if peak > 0:
        cur = float(np.abs(audio).max())
        if cur > 1e-6:
            audio = (audio * (peak / cur)).astype(np.float32)
    return audio


# -- 스트리밍 -------------------------------------------------------------
async def run_one(ws, audio: np.ndarray, target_lang: str, trailing_ms: int,
                  utt_id: str = ""):
    t0 = time.perf_counter()
    # uttId: 서버가 발화 경계를 넘은 final 을 어느 발화에 귀속할지 판단하는 값
    # (streaming_websocket_server_ast.py `_StartSniffingWS._sniff`). 안 보내면 None 이
    # 되어 [AST-LATE] 추적이 무력화된다.
    await ws.send(json.dumps({"type": "start", "lang": "auto",
                              "targetLang": target_lang, "uttId": utt_id}))
    # ready 대기. 서버가 첫 연결에서 엔진을 lazy 로 올리므로 첫 클립은 모델 로딩
    # 시간을 그대로 기다린다(Nemotron 약 30초). 30초로 잡으면 첫 클립이
    # TimeoutError 로 빠져 이 백엔드만 49클립이 되고 비교 조건이 깨진다.
    deadline = time.perf_counter() + 180
    while time.perf_counter() < deadline:
        m = await asyncio.wait_for(ws.recv(), timeout=30)
        if isinstance(m, str) and json.loads(m).get("type") == "ready":
            break

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    step = int(0.2 * SAMPLING_RATE)
    finals, lags, first_at = [], [], None
    first_arr = last_arr = None      # 도착 시각(벽시계). 확정 지연 기준.
    segs_policy = []   # [(text, d_ms)] d = 커밋 결정 시점까지 읽은 소스 오디오
    segs_ca = []       # [(text, d_ms)] d = 그 출력을 받은 실시간 경과
    snaps = []         # [(누적 텍스트, d_ms)] partial - 미확정 가설(통째 교체)
    n_decision = 0     # decisionAudioSec 을 실제로 받은 final 수
    done = asyncio.Event()

    async def reader():
        nonlocal first_at, first_arr, last_arr, n_decision
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
            elif t == "partial":
                # 미확정 가설. 채점에는 절대 쓰지 않고 지연 축에만 쓴다.
                ptxt = (d.get("text") or "").strip()
                if ptxt:
                    snaps.append((ptxt, (time.perf_counter() - origin) * 1000.0))
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
                    # 체감(computation-aware): 스트리밍 시작부터 이 출력을 받기까지
                    d_ca = (_now - origin) * 1000.0
                    segs_ca.append((txt, d_ca))
                    # 정책: 서버가 주면 그 값, 없으면 1배속이라 체감과 같은 축
                    # 정책 축은 서버가 결정 시점을 줄 때만 성립한다. 없으면
                    # 비워 둔다 - 도착 시각으로 채우면 계산 비용이 섞인 값을
                    # "정책" 이라 부르는 셈이고, 그 값만 다른 백엔드와 비교하면
                    # 결정 시점을 주는 쪽만 유리해진다.
                    _dec = d.get("decisionAudioSec")
                    if _dec is not None:
                        n_decision += 1
                        segs_policy.append((txt, float(_dec) * 1000.0))

    origin = time.perf_counter()
    task = asyncio.create_task(reader())
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
            "segs_policy": segs_policy,
            "segs_ca": segs_ca,
            "snaps": snaps,
            "n_decision": n_decision,
            "avg_fsl_sec": mean(lags) if lags else None,
            # 확정 지연: 마지막 출력 − 실제 발화 끝(뒤에 붙인 무음 제외)
            "finalization_lag_sec": round(last_arr - t_audio_end, 3) if last_arr else None,
            "rtf": round((last_arr - origin) / audio_sec, 3) if last_arr else None,
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
            audio = load_audio(r["path"], a.peak_normalize)
            try:
                out = await run_one(ws, audio, a.lang, a.trailing_ms,
                                    utt_id=r["file_id"])
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
            _sec = len(audio) / SAMPLING_RATE
            _laal, _laal_ca = laal_pair(out.pop("segs_policy"), out.pop("segs_ca"),
                                        _sec, r["reference"], a.lang)
            _snaps = out.pop("snaps")
            _lu = LAAL_UNIT.get(a.lang, "word")
            _sf, _ss = partial_axes(_snaps, out["transcript"], _lu)
            _lf = _ls = None
            if _sf and laal_for_utterance is not None and _sec > 0:
                _lf = round(laal_for_utterance(_sf, _sec * 1000.0, r["reference"], _lu), 1)
                _ls = round(laal_for_utterance(_ss, _sec * 1000.0, r["reference"], _lu), 1)
            results.append({**r, **out, "audio_sec": _sec,
                            "laal_ms": _laal, "laal_ca_ms": _laal_ca,
                            "laal_first_ms": _lf, "laal_stable_ms": _ls,
                            "n_partial": len(_snaps), unit: s})
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
    fsl = [r["avg_fsl_sec"] for r in results if r.get("avg_fsl_sec") is not None]
    fin = [r["finalization_lag_sec"] for r in results
           if r.get("finalization_lag_sec") is not None]
    laal = [r["laal_ms"] for r in results if r.get("laal_ms") is not None]
    laal_ca = [r["laal_ca_ms"] for r in results if r.get("laal_ca_ms") is not None]
    laal_f = [r["laal_first_ms"] for r in results if r.get("laal_first_ms") is not None]
    laal_s = [r["laal_stable_ms"] for r in results if r.get("laal_stable_ms") is not None]
    n_par = [r.get("n_partial", 0) for r in results]
    rtfs = [r["rtf"] for r in results if r.get("rtf") is not None]
    n_dec = sum(r.get("n_decision", 0) for r in results)
    audio_total = sum(r["audio_sec"] for r in results) or 1
    summary = {
        "backend": a.tag, "lang": a.lang, "unit": unit,
        "peak_normalize": a.peak_normalize,
        "laal_unit": LAAL_UNIT.get(a.lang, "word"),
        "n_ok": len(results), "n_total": len(rows),
        f"avg_{unit}": round(mean(vals), 4) if vals else None,
        # LAAL: 정책(결정 시점) / LAAL_CA: 체감(도착 시점). 낮을수록 좋다.
        "laal_ms": round(mean(laal), 1) if laal else None,
        "laal_ca_ms": round(mean(laal_ca), 1) if laal_ca else None,
        # partial 채널(있는 백엔드만). first = 처음 뜬 시각, stable = 굳은 시각
        "laal_first_ms": round(mean(laal_f), 1) if laal_f else None,
        "laal_stable_ms": round(mean(laal_s), 1) if laal_s else None,
        "partials_per_clip": round(mean(n_par), 1) if n_par else None,
        "finalization_lag_sec": round(mean(fin), 3) if fin else None,
        "avg_fsl_sec": round(mean(fsl), 3) if fsl else None,
        "rtf": round(mean(rtfs), 3) if rtfs else None,
        "wall_rtf": round((time.perf_counter() - t_start) / audio_total, 3),
        "finals_with_decision_sec": n_dec,
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
    ap.add_argument("--peak-normalize", type=float, default=0.0,
                    metavar="PEAK",
                    help="클립별 피크를 이 값으로 맞춘다(0=원본 그대로). "
                         "세 백엔드에 같은 값을 줘야 비교가 성립한다.")
    ap.add_argument("--tag", default="unknown")
    ap.add_argument("--out", default="smoke.json")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
