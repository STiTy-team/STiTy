"""같은 프롬프트를 dev 에서 한 번 더 채점한다 — 잡음 바닥 측정.

루프의 모든 Δ 는 "프롬프트 A 한 번 vs 프롬프트 B 한 번" 이다. 분절기가 비결정론이면
(gpt-5-mini 는 temperature 를 거부한다) A 와 A 를 두 번 재도 Δ 가 0 이 아니다. 그
크기를 모르면 채택 문턱(1 se)이 무엇을 거르는지 알 수 없다.

run22 실측으로 관문 통과 규칙은 dev 3,399 경계 중 3 자리에 걸리는데, 그 개정은 남긴
경계의 37% 를 옮겼다. 그 37% 가 샘플링 잡음인지 문안 교란인지는 **같은 프롬프트의
재채점**만이 가른다 — 재채점 churn 이 30% 대면 잡음, 한 자리면 문안.

기준(첫 채점)은 런의 `iter_00/dev_rows.json` 을 그대로 읽는다. 재채점은 **별도 캐시**
(`cache/segment_retest_<tag>.json`)를 쓴다 — 런 캐시를 쓰면 전부 적중해 아무것도 안 잰다.

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.tools.autoseg_en2x.retest \\
        --run-id en2x/en-multi/run22 --provider openai --tag v0a

비용 기록: 산출물 `retest_<tag>.json` 에 usage 를 싣고, 도는 동안 30초마다 같은 파일에
누적 usage 를 덮어 쓴다 (크래시해도 지출이 남는다). `cost_report` 가 그대로 집계한다.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
import threading
import time
from pathlib import Path

from core.meaning_segmentator.autoseg.infra.gateway import Gateway, add_provider_args
from core.meaning_segmentator.autoseg.loop_distill import evaluate, paired
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import data
from core.meaning_segmentator.autoseg.runtime.pipeline import JsonCache


def churn(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """절단집합이 바뀐 (문장,T) 비율과 옮겨간 경계 비율. 두 채점이 같으면 전부 0."""
    b = {r["id"]: r for r in rows_b}
    n = changed = up = down = kept = moved = 0
    for r in rows_a:
        s = b.get(r["id"])
        if not s or not r.get("by_T") or not s.get("by_T"):
            continue
        for T, x in r["by_T"].items():
            y = s["by_T"].get(T)
            if not y:
                continue
            n += 1
            kx, ky = set(x["kept_model"]), set(y["kept_model"])
            if kx != ky:
                changed += 1
                up += y["overlap"] > x["overlap"]
                down += y["overlap"] < x["overlap"]
            kept += len(kx)
            moved += len(kx - ky)
    return {"pairs": n, "changed": changed, "changed_rate": round(changed / max(1, n), 3),
            "up": up, "down": down, "kept": kept, "moved": moved,
            "moved_rate": round(moved / max(1, kept), 3)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--prompt", default=None, help="기본은 런의 prompt_v0.txt")
    p.add_argument("--base-rows", default=None, help="기본은 iter_00/dev_rows.json")
    p.add_argument("--tag", required=True, help="산출물·캐시 꼬리표. 재채점마다 달라야 한다")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--seg-reasoning-effort", default="medium")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--budget", type=float, default=2.0)
    add_provider_args(p)
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_id
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    prompt = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else \
        (run_dir / "prompt_v0.txt").read_text(encoding="utf-8")
    base_rows = json.loads(Path(args.base_rows or run_dir / "iter_00" / "dev_rows.json")
                           .read_text(encoding="utf-8"))
    sents = [data.Sentence(**r) for r in
             json.loads((run_dir / "data" / "dev.json").read_text(encoding="utf-8"))]
    labels = json.loads((run_dir / "oracle_labels_dev.json").read_text(encoding="utf-8"))

    gw = Gateway.from_args(args, budget=args.budget)
    cache = JsonCache(run_dir / "cache" / f"segment_retest_{args.tag}.json")
    effort = None if args.seg_reasoning_effort == "none" else args.seg_reasoning_effort
    out = run_dir / f"retest_{args.tag}.json"

    def dump(extra: dict | None = None) -> None:
        blob = {"tag": args.tag, "prompt": args.prompt or "prompt_v0.txt", "n": len(sents),
                "model": args.model, "seg_reasoning_effort": args.seg_reasoning_effort,
                "usage": gw.usage.snapshot(), **(extra or {})}
        out.write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")

    stop = threading.Event()

    def ticker() -> None:
        while not stop.wait(30):
            dump()
            u = gw.usage.snapshot()
            print(f"  [진행] 호출 {u['calls']} 누적 ${u['cost']:.4f}", flush=True)

    print(f"dev {len(sents)}문장 / T {cfg['t_grid']} / {args.model} effort={args.seg_reasoning_effort}")
    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    t0 = time.time()
    try:
        rows, m = evaluate(gw, prompt, sents, labels, cfg["spaced"], cfg["min_gap"],
                           cfg["t_grid"], cache, args.workers, args.batch_size, effort)
    finally:
        stop.set()
        cache.flush()
        dump()
    b_by = {r["id"]: r for r in base_rows}
    base_m = {"overlap": round(st.mean(r["overlap"] for r in base_rows if r.get("overlap") is not None), 4)}
    d = paired(rows, base_rows, key="overlap")
    per = [r["overlap"] - b_by[r["id"]]["overlap"] for r in rows
           if r.get("overlap") is not None and b_by.get(r["id"], {}).get("overlap") is not None]
    ch = churn(base_rows, rows)
    sp = {"base": None, "retest": m["spearman_within"]}
    res = {"base_overlap": base_m["overlap"], "retest_overlap": m["overlap"],
           "retest_by_T": m["overlap_by_T"], "paired": d,
           "delta_sd": round(st.stdev(per), 4) if len(per) > 1 else None,
           "abs_delta_mean": round(st.mean(abs(x) for x in per), 4) if per else None,
           "same_sentence_rate": round(sum(1 for x in per if x == 0) / max(1, len(per)), 3),
           "churn": ch, "spearman_within": sp, "format_pass_rate": m["format_pass_rate"],
           "first_pass_violations": m["first_pass_violations"]["counts"],
           "seconds": round(time.time() - t0)}
    dump({"result": res})
    (run_dir / f"retest_{args.tag}_rows.json").write_text(json.dumps(rows, ensure_ascii=False),
                                                          encoding="utf-8")
    u = gw.usage.snapshot()
    print(f"\n기준 overlap {base_m['overlap']} -> 재채점 {m['overlap']} {m['overlap_by_T']}")
    print(f"쌍체 Δ {d['mean_delta']:+.4f} ± {d['se_delta']:.4f} (n={d['n_pairs']}) / "
          f"문장별 Δ sd {res['delta_sd']} / |Δ| 평균 {res['abs_delta_mean']} / "
          f"Δ=0 문장 {res['same_sentence_rate']:.0%}")
    print(f"churn: (문장,T) {ch['changed']}/{ch['pairs']} ({ch['changed_rate']:.0%}) 바뀜, "
          f"오름 {ch['up']} 내림 {ch['down']}; 경계 {ch['moved']}/{ch['kept']} ({ch['moved_rate']:.0%}) 옮김")
    print(f"비용 ${u['cost']:.4f} / 호출 {u['calls']} / {res['seconds']}s -> {out}")
    gw.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
