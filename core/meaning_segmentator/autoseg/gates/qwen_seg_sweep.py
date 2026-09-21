"""지연 대비 품질 곡선 — 스트리밍 임계값 `<SEG>` 절단을 문장 끝을 보는 절단들과 같은 자로 잰다.

x = 문장별 평균 조각 길이(어절)의 평균, y = `H_set` 평균 (run27 test 200문장 전부. 무절단 문장도
포함 — 그때 H_set 은 통째 번역의 QE 다). `qwen_seg_prob.py` 와 같은 채점기·번역 캐시.

곡선 두 부류:

  스트리밍 (단어가 들어오는 즉시 결정, 문장 끝 모름)
    qwen_stream_feed   P(<SEG>) ≥ θ 이면 자르고 ' <SEG>' 를 문맥에 넣는다 (ASR 서버와 같은 조건)
    qwen_stream        같은데 <SEG> 를 문맥에 안 넣는다. `qwen_scores_*` 의 p0 로 모사한다
    fixed_N            N 어절마다 자른다
    punct_stream       구두점으로 끝난 어절 뒤에서 자른다 (점 하나)
  문장 끝을 보는 절단 (상위 k 자리, 목표 조각 길이 T 에서 k = round(n/T) − 1)
    oracle / judge13 / punct_k

스트리밍 절단은 모두 `--min-words` (기본 2) 어절 이상 쌓였을 때만 자른다.

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.autoseg.gates.qwen_seg_sweep \
        --from-run en2x/en-multi/run27 --run-id en2x/en-multi/qwenseg01 --prompt-run en2x/en-multi/judge13
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

from ..paths import RUNS_DIR
from .qwen_seg_prob import DEFAULT_MODEL, PUNCT, load_thinker, log

THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
FIXED_N = (2, 3, 4, 5, 6, 8, 10, 14)
T_GRID = (2, 3, 4, 5, 6, 8, 10, 14)


def stream_cuts_from_p(ps: list[float], n: int, th: float, min_words: int) -> tuple[int, ...]:
    """경계 확률 ps[j−1] (j=1..n−1) 로 스트리밍 절단을 모사한다 (문맥에 <SEG> 를 안 넣는 경우)."""
    cuts, last = [], 0
    for j in range(1, n):
        if ps[j - 1] >= th and j - last >= min_words:
            cuts.append(j)
            last = j
    return tuple(cuts)


def fixed_cuts(n: int, N: int) -> tuple[int, ...]:
    return tuple(range(N, n, N))


def punct_stream_cuts(units: list[str], min_words: int) -> tuple[int, ...]:
    cuts, last = [], 0
    for j in range(1, len(units)):
        if PUNCT.search(units[j - 1]) and j - last >= min_words:
            cuts.append(j)
            last = j
    return tuple(cuts)


def k_for(n: int, T: int) -> int:
    return max(0, min(n - 1, round(n / T) - 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-run", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--prompt-run", default=None)
    ap.add_argument("--prompt-budget", type=float, default=0.5)
    ap.add_argument("--comet-batch-size", type=int, default=32)
    a = ap.parse_args()

    src, run_dir = RUNS_DIR / a.from_run, RUNS_DIR / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "cache").exists():
        (run_dir / "cache").symlink_to(Path("..") / src.name / "cache")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    targets = cfg["targets"]
    rows = json.loads((src / f"data/{a.split}.json").read_text(encoding="utf-8"))
    lab = json.loads((src / f"oracle_labels_{a.split}.json").read_text(encoding="utf-8"))
    texts = [r["text"] for r in rows]
    units = [t.split() for t in texts]
    tag = Path(a.model).name

    # ── 스트리밍 절단 (feed-seg) — 모델을 실제로 흘린다. 결과는 캐시
    feed_path = run_dir / f"stream_feed_cuts_{a.split}_{tag}_mw{a.min_words}.json"
    feed = json.loads(feed_path.read_text(encoding="utf-8")) if feed_path.exists() else {}
    todo = [th for th in THRESHOLDS if str(th) not in feed]
    if todo:
        import torch
        from .qwen_seg_stream import StreamSegmenter
        tok, thinker = load_thinker(Path(a.model))
        for th in todo:
            t0 = time.time()
            seg = StreamSegmenter(tok, thinker, "English", th, a.min_words, True, 500)
            per = []
            for u in units:
                seg.reset([])
                cuts = [i for i, w in enumerate(u, 1) if seg.feed(w)["cut"] and i < len(u)]
                per.append(cuts)
            feed[str(th)] = per
            feed_path.write_text(json.dumps(feed), encoding="utf-8")
            log(f"[stream feed θ={th}] 절단 {sum(map(len, per))} / {time.time() - t0:.0f}s")
        del thinker
        torch.cuda.empty_cache()

    qs = json.loads(next(run_dir.glob(f"qwen_scores_{a.split}_{tag}.json")).read_text(encoding="utf-8"))

    # ── 채점기
    from ..loop import target_is_spaced
    from ..runtime import data, hset, metrics
    from ..runtime import labels as L
    from ..runtime.pipeline import JsonCache, LocalTranslator, to_lang_code
    sents = [data.Sentence(**r) for r in rows]
    translators = {t: LocalTranslator(tgt_code=to_lang_code(t),
                                      cache=JsonCache(run_dir / "cache" / f"translate_{to_lang_code(t)}.json"))
                   for t in targets}
    scorer = hset.HsetScorer(translators=translators,
                             qe=metrics.make_adequacy_backend(cfg.get("adequacy_backend", "cometkiwi"),
                                                              batch_size=a.comet_batch_size),
                             spaced=True, target_spaced={t: target_is_spaced(t) for t in targets})

    def point(cutsets: list[tuple[int, ...]]) -> dict:
        vals = scorer.score(texts, list(enumerate(cutsets)),
                            lambda i, j: lab[targets[0]][i]["contra"][j - 1])
        for tr in translators.values():
            if tr.cache is not None:
                tr.cache.flush()
        return {"chunk": round(st.mean(len(u) / (len(c) + 1) for u, c in zip(units, cutsets)), 3),
                "h": round(st.mean(vals), 4), "cuts": sum(map(len, cutsets)),
                "per": [round(v, 4) for v in vals]}

    def k_policy(score_of):
        out = {}
        for T in T_GRID:
            cs = []
            for i, u in enumerate(units):
                sc = score_of(i, u)
                cs.append(hset.top_k_cuts(u, sc, k_for(len(u), T), 1) if k_for(len(u), T) else ())
            out[T] = cs
        return out

    curves: dict[str, list] = {}

    def run_curve(name: str, knob_sets: dict) -> None:
        t0 = time.time()
        pts = []
        for knob, cs in knob_sets.items():
            p = point(cs)
            p["knob"] = knob
            pts.append(p)
        curves[name] = pts
        log(f"[{name}] " + " ".join(f"{p['knob']}:({p['chunk']:.1f},{p['h']:.3f})" for p in pts)
            + f" / {time.time() - t0:.0f}s")

    run_curve("qwen_stream_feed", {th: [tuple(c) for c in feed[str(th)]] for th in THRESHOLDS})
    run_curve("qwen_stream", {th: [stream_cuts_from_p([math.exp(x) for x in qs[i]["p0"]], len(u), th,
                                                      a.min_words) for i, u in enumerate(units)]
                              for th in THRESHOLDS})
    run_curve("fixed_N", {N: [fixed_cuts(len(u), N) for u in units] for N in FIXED_N})
    run_curve("punct_stream", {"punct": [punct_stream_cuts(u, a.min_words) for u in units]})
    run_curve("oracle", k_policy(lambda i, u: {j: L.label_value(lab, i, j) for j in range(1, len(u))}))
    run_curve("punct_k", k_policy(lambda i, u: {j: float(bool(PUNCT.search(u[j - 1]))) for j in range(1, len(u))}))

    if a.prompt_run:
        from ..infra.gateway import Gateway
        from ..loop_distill import evaluate
        prun = RUNS_DIR / a.prompt_run
        gw = Gateway(provider="openai", model="gpt-5-mini", budget=a.prompt_budget,
                     reasoning_effort="medium", max_connections=16)
        rws, _m = evaluate(gw, (prun / "best_prompt.txt").read_text(encoding="utf-8"), sents, lab, True, 1,
                           cfg["t_grid"], JsonCache(prun / "cache" / "segment.json"), 16, 6, "medium",
                           k_samples=3)
        u = gw.usage.snapshot()
        (run_dir / "metrics.json").write_text(json.dumps({"usage": u, "run_total_cost": u["cost"]}, indent=1),
                                              encoding="utf-8")
        log(f"[judge13] 분절 캐시 재생 — 호출 {u['calls']} / ${u['cost']:.4f}")
        gw.close()
        by_id = {r["id"]: r for r in rws if r.get("scores")}

        def judge_score(i, u):
            r = by_id.get(sents[i].id)
            return {j: float(x) for j, x in zip(r["positions"], r["scores"])} if r else {}
        run_curve("judge13", k_policy(judge_score))

    out = run_dir / f"sweep_{a.split}_{tag}_mw{a.min_words}.json"
    out.write_text(json.dumps(curves, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[끝] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
