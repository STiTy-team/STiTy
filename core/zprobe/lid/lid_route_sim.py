"""LID 라우팅 시뮬레이션 — grpocer25 dual 런의 번역 스트림을 LID 판정으로 필터해 재채점.

하이브리드 표시 규칙: 타깃 L 자막의 번역 칸에는 "입력 언어 != L" 로 판정된 커밋만 나간다.
입력이 L 로 판정된 구간은 전사 칸이 전담한다 (bench COMET 은 어차피 교차 언어 문장만
채점하므로, 오판으로 번역 커밋이 빠지면 그 문장 mt 가 비어 0점 — 자막 증발 그대로).

조건: none(현행, 필터 없음) / ac(음향 단독 = 기존 lid_router) / twostage / oracle(정답 스팬).
판정 시점: 커밋의 decision_audio_sec 이전에 나온 마지막 타임라인 증거 (인과적).
증거가 아직 없으면 번역으로 내보낸다 (기본 경로).

--lags: 커밋 내용은 audio 창보다 디코드 룩어헤드만큼 과거라, 판정 창을 L 초 뒤로
밀어 귀속한다 (ns_score.py dual-wer 의 --lag 과 같은 보정). 0 = 보정 없음.

bench venv 로, cwd=STiTy-bench 에서 실행.
"""
import argparse, copy, json, os, sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from bench.metrics.latency import longyaal
from bench.metrics.translation import translation_sentences

NS = "/home/mobility/stity-datasets/northstar-260928"

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--timeline", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--lags", default="0", help="쉼표 구분 lag 초 목록")
a = ap.parse_args()

timeline = json.load(open(a.timeline))
spans = {}
for line in open(f"{NS}/manifest.jsonl"):
    it = json.loads(line)
    scene = it["audio"].split("/")[-1].removesuffix(".wav")
    spans.setdefault(scene, []).append((it["offset"], it["offset"] + it["duration"], it["src_lang"]))


def truth_of(scene, lo, hi):
    best, lang = -1.0, None
    for s, e, l in spans.get(scene, []):
        d = min(hi, e) - max(lo, s)
        if d > best:
            best, lang = d, l
    return lang


def verdict_at(scene, t, kind):
    idx = 1 if kind == "ac" else 2
    v = None
    for entry in timeline.get(scene, []):
        if entry[0] <= t:
            v = entry[idx] or v
        else:
            break
    return v


rows = [json.loads(l) for l in open(Path(a.run_dir) / "items.jsonl")]

conds = [("none", 0.0)] + [(c, lag) for lag in (float(x) for x in a.lags.split(","))
                           for c in ("ac", "twostage", "oracle")]
for cond, lag in conds:
    crows = copy.deepcopy(rows)
    kept = dropped = default_kept = 0
    for row in crows:
        scene, prev_t = row["id"], 0.0
        recs = []
        for r in row["records"]:
            if r.get("type") != "translated":
                recs.append(r)
                continue
            t = float(r.get("decision_audio_sec") or prev_t)
            if cond == "none":
                v = None
            elif cond == "oracle":
                hi = (t if t > prev_t else prev_t + 0.5) - lag
                v = truth_of(scene, prev_t - lag, hi)
            else:
                v = verdict_at(scene, t - lag, cond)
            prev_t = t
            if cond != "none" and v is None:
                default_kept += 1
            if cond != "none" and v == a.target:
                dropped += 1
                continue
            kept += 1
            recs.append(r)
        row["records"] = recs

    ly = longyaal(crows, a.target)
    lyca = longyaal(crows, a.target, computation_aware=True)
    sentences = translation_sentences(crows, a.target)
    empty = sum(1 for s in sentences if s["src"] and not s["mt"])
    tag = cond if cond == "none" else f"{cond}-lag{lag:g}"
    out = Path(a.outdir) / f"{a.target}-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comet_inputs.jsonl", "w") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    json.dump({"metrics": {"longyaal_ms": ly, "longyaal_ca_ms": lyca,
                           "kept": kept, "dropped": dropped, "default_kept": default_kept,
                           "empty_mt": empty, "n_sentences": sum(1 for s in sentences if s["src"])},
               "unavailable": {}},
              open(out / "summary.json", "w"), indent=1)
    print(f"{a.target}-{tag}: longyaal={ly and round(ly)} ca={lyca and round(lyca)} "
          f"kept={kept} dropped={dropped} default={default_kept} empty_mt={empty}", flush=True)
print("ROUTE_SIM_DONE", flush=True)
