"""dedup 라우팅 시뮬 — 번역 커밋을 시각이 아니라 내용으로 거른다.

규칙: 어댑터 번역 커밋의 텍스트가 전사 스트림(koen ASR 커밋)의 근처 창 텍스트와
겹치면(문자 3-gram 포함률 >= 문턱) verbatim 중복으로 보고 번역 pane 에서 접는다.
입력==타깃 구간의 통과 출력만 걸리고, 진짜 번역(다른 언어 원문)은 전사와 안 겹쳐 남는다.
LID 불필요·인과적(창 = 커밋 시각 기준 [t-8, t+2], +2 는 전사가 번역보다 빠른 경우 약간 허용).

bench venv, cwd=STiTy-bench.
"""
import argparse, copy, json, os, re, sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from bench.metrics.latency import longyaal
from bench.metrics.translation import translation_sentences

KOEN_RUN = "/home/mobility/STiTy-bench/bench/runs/northstar_ko+en-ko@v2/asr.qwen-seg-koen+mt.qwen3.5-4b"

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--thresholds", default="0.5,0.7")
a = ap.parse_args()


def norm(text):
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def trigrams(s):
    return {s[i:i + 3] for i in range(len(s) - 2)}


transcript = {}
for line in open(Path(KOEN_RUN) / "items.jsonl"):
    row = json.loads(line)
    transcript[row["id"]] = [(float(c["decision_audio_sec"]), norm(c.get("original") or ""))
                             for c in row["records"]
                             if c.get("type") == "transcribed" and c.get("decision_audio_sec") is not None]


def containment(scene, t, text):
    x = norm(text)
    if len(x) < 4:
        window_has = any(abs(ct - t) <= 5.0 and x and x in ctx
                         for ct, ctx in transcript.get(scene, []))
        return 1.0 if window_has else 0.0
    window = "".join(ctx for ct, ctx in transcript.get(scene, []) if t - 8.0 <= ct <= t + 2.0)
    tg = trigrams(x)
    if not tg:
        return 0.0
    wg = trigrams(window)
    return len(tg & wg) / len(tg)


rows = [json.loads(l) for l in open(Path(a.run_dir) / "items.jsonl")]

for thr in [float(x) for x in a.thresholds.split(",")]:
    crows = copy.deepcopy(rows)
    kept = dropped = 0
    for row in crows:
        scene = row["id"]
        recs = []
        for r in row["records"]:
            if r.get("type") != "translated":
                recs.append(r)
                continue
            t = float(r.get("decision_audio_sec") or 0.0)
            if containment(scene, t, r.get("translation") or "") >= thr:
                dropped += 1
                continue
            kept += 1
            recs.append(r)
        row["records"] = recs

    ly = longyaal(crows, a.target)
    lyca = longyaal(crows, a.target, computation_aware=True)
    sentences = translation_sentences(crows, a.target)
    empty = sum(1 for s in sentences if s["src"] and not s["mt"])
    out = Path(a.outdir) / f"{a.target}-dedup{thr:g}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comet_inputs.jsonl", "w") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    json.dump({"metrics": {"longyaal_ms": ly, "longyaal_ca_ms": lyca, "kept": kept,
                           "dropped": dropped, "empty_mt": empty,
                           "n_sentences": sum(1 for s in sentences if s["src"])},
               "unavailable": {}},
              open(out / "summary.json", "w"), indent=1)
    print(f"{a.target}-dedup{thr:g}: longyaal={ly and round(ly)} ca={lyca and round(lyca)} "
          f"kept={kept} dropped={dropped} empty_mt={empty}", flush=True)
print("DEDUP_SIM_DONE", flush=True)
