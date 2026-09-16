"""루프 밖 사이드 채점 — 어느 런의 후보 프롬프트를 한 분할에서 재고 기준 프롬프트와 짝 부트스트랩.

선별(50문장)에서 아깝게 떨어진 후보가 200문장에서는 어땠을지 보려는 것이다(judge12: 5이터 전부
후보 CI 가 겹쳤다). 루프의 채점 경로(`evaluate` → `policy_sets` → `HsetScorer`)를 그대로 쓰고
런의 분절·번역 캐시를 공유한다 — **루프가 도는 동안 돌리지 않는다** (캐시 파일을 서로 덮어쓴다).

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.tools.autoseg_en2x.run27.score_candidate \\
        --run-id en2x/en-multi/judge12 --from-run en2x/en-multi/run27 --split test_a \\
        --base prompt_v0.txt --candidates iter_05/candidate_3.txt iter_04/candidate_0.txt \\
        --max-k 99 --workers 128 --extra-key-envs OPENAI_API_KEY_2

비용: 결과 JSON(`side_eval_<tag>.json`)에 usage 를 싣고 채점마다 덮어 쓴다. `cost_report` 형식.
"""
from __future__ import annotations
import argparse, json, statistics as st, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from core.meaning_segmentator.autoseg.infra.gateway import Gateway, add_provider_args
from core.meaning_segmentator.autoseg.loop import target_is_spaced
from core.meaning_segmentator.autoseg.loop_distill import evaluate
from core.meaning_segmentator.autoseg.loop_judge import by_latency, policy_sets
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import data, hset, metrics
from core.meaning_segmentator.autoseg.runtime.pipeline import JsonCache, LocalTranslator, to_lang_code

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--run-id", required=True)
p.add_argument("--from-run", required=True)
p.add_argument("--split", default="test_a")
p.add_argument("--base", default="prompt_v0.txt", help="런 디렉토리 기준 상대 경로")
p.add_argument("--candidates", nargs="+", required=True, help="런 디렉토리 기준 상대 경로들")
p.add_argument("--tag", default=None)
p.add_argument("--model", default="gpt-5-mini")
p.add_argument("--seg-reasoning-effort", default="medium")
p.add_argument("--k-samples", type=int, default=3)
p.add_argument("--batch-size", type=int, default=6)
p.add_argument("--workers", type=int, default=64)
p.add_argument("--min-gap", type=int, default=1)
p.add_argument("--min-chunk", type=int, default=2)
p.add_argument("--max-k", type=int, default=99)
p.add_argument("--comet-batch-size", type=int, default=32)
p.add_argument("--budget", type=float, default=10.0)
add_provider_args(p)
a = p.parse_args()

run_dir, src = RUNS_DIR / a.run_id, RUNS_DIR / a.from_run
cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
spaced, targets, t_grid = cfg["spaced"], cfg["targets"], cfg["t_grid"]
sents = [data.Sentence(**r) for r in json.loads((src / f"data/{a.split}.json").read_text(encoding="utf-8"))]
lab = json.loads((src / f"oracle_labels_{a.split}.json").read_text(encoding="utf-8"))
tag = a.tag or f"{a.split}_{int(time.time())}"
out_path = run_dir / f"side_eval_{tag}.json"

gw = Gateway.from_args(a, model=a.model, budget=a.budget, max_connections=max(16, a.workers))
seg_cache = JsonCache(run_dir / "cache" / "segment.json")
translators = {t: LocalTranslator(tgt_code=to_lang_code(t),
                                  cache=JsonCache(run_dir / "cache" / f"translate_{to_lang_code(t)}.json"))
               for t in targets}
scorer = hset.HsetScorer(translators=translators,
                         qe=metrics.make_adequacy_backend(cfg.get("adequacy_backend", "cometkiwi"),
                                                          batch_size=a.comet_batch_size),
                         spaced=spaced, target_spaced={t: target_is_spaced(t) for t in targets})
seg_effort = None if a.seg_reasoning_effort == "none" else a.seg_reasoning_effort
result = {"run_id": a.run_id, "split": a.split, "base": a.base, "scores": {}, "usage": {}}


def save():
    result["usage"] = gw.usage.snapshot()
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def score(rel: str) -> tuple[dict, dict]:
    pr = (run_dir / rel).read_text(encoding="utf-8")
    t0 = time.time()
    rows, m = evaluate(gw, pr, sents, lab, spaced, a.min_gap, t_grid, seg_cache, a.workers,
                       a.batch_size, seg_effort, k_samples=a.k_samples)
    seg_cache.flush()
    sets = policy_sets(rows, sents, spaced, a.min_gap, a.min_chunk, a.max_k)
    keys = sorted(sets)
    vals = scorer.score([s.text for s in sents], [(i, sets[(i, T)]) for i, T in keys],
                        lambda i, j: lab[targets[0]][i]["contra"][j - 1])
    for tr in translators.values():
        tr.cache.flush()
    h = dict(zip(keys, vals))
    print(f"[{rel}] H_set {st.mean(h.values()):.4f} {by_latency(sents, h, spaced)} / fmt {m['format_pass_rate']} "
          f"(1차 {m['format_pass_rate_no_retry']}, 재정렬 {m['first_pass_violations'].get('realigned', 0)}) / "
          f"{time.time() - t0:.0f}s / 누적 ${gw.usage.snapshot()['cost']:.2f}", flush=True)
    result["scores"][rel] = {"mean": round(st.mean(h.values()), 4), "by_bin": by_latency(sents, h, spaced),
                             "fmt": m["format_pass_rate"], "n": len(h)}
    save()
    return h, m


h0, _ = score(a.base)
for rel in a.candidates:
    h, _ = score(rel)
    keys = sorted(set(h) & set(h0))
    boot = hset.paired_bootstrap([h[k] for k in keys], [h0[k] for k in keys], clusters=[i for i, _k in keys])
    print(f"[{rel}] Δ vs {a.base} {boot['mean']:+.4f} [{boot['lo']:+.4f}, {boot['hi']:+.4f}] "
          f"짝 {boot['n']} / 문장 {boot['n_clusters']}", flush=True)
    result["scores"][rel]["delta_vs_base"] = boot
    save()
print(f"-> {out_path}")
