"""프롬프트에 규칙 한 줄을 손으로 넣고 홀드아웃에서 재본다 — 표를 만들 값어치가 있나.

전수 집계로 찾은 부류(기능어 바로 뒤 절단을 모델이 체계적으로 과소평가, train 529 경계
t=-13.4, 증거에 없던 기능어만 따로 봐도 n=206 t=-7.31)가 **프롬프트로 고쳐지는지**를
가장 싸게 확인한다. 배관을 다 만들기 전에 이것부터 한다.

**라벨에서 어긋난다고 프롬프트로 고쳐진다는 보장이 없다.** run19 에서 관문을 통과한
감점 규칙(숫자|단위, t=-3.35)을 넣었더니 부검이 `removed_safe_boundary` 를 진단했다 —
통계적으로 유효한 규칙이 원래 잘 잡던 경계까지 깎았다. 그래서 재본다.

루프가 후보를 재는 `loop_distill.evaluate` 를 그대로 쓴다. 나오는 Δ 는 런 로그의
`[iter N 개정] ... Δ=` 와 같은 자다. 기준 프롬프트는 캐시에 있으므로 실제 비용은
변형 하나 분량(홀드아웃 100문장 ≈ 17콜)이다.

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.tools.autoseg_en2x.rule_probe \
        --run-id en2x/en-multi/run20 --rule-file <추가할 규칙.txt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.meaning_segmentator.autoseg.infra.gateway import Gateway, add_provider_args
from core.meaning_segmentator.autoseg.loop_distill import evaluate, paired
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import data
from core.meaning_segmentator.autoseg.runtime.pipeline import JsonCache


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="기준 런 (분할·라벨·캐시를 그대로 쓴다)")
    p.add_argument("--rule-file", required=True, help="[Scoring Rules] 끝에 붙일 줄들")
    p.add_argument("--prompt", default=None, help="기준 프롬프트. 기본은 런의 best_prompt.txt")
    p.add_argument("--train", type=int, default=100, help="배치 크기 — 홀드아웃은 그 뒤")
    p.add_argument("--section", default="[Decision Procedure]",
                   help="이 헤더 **앞**에 규칙을 끼운다 ([Scoring Rules] 의 끝)")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--seg-reasoning-effort", default="medium")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--budget", type=float, default=3.0)
    add_provider_args(p)
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_id
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    base = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else \
        (run_dir / "best_prompt.txt").read_text(encoding="utf-8")
    add = Path(args.rule_file).read_text(encoding="utf-8").strip()

    if args.section not in base:
        print(f"섹션 '{args.section}' 이 프롬프트에 없다", file=sys.stderr)
        return 1
    variant = base.replace(args.section, f"{add}\n\n{args.section}", 1)

    # 분할·라벨은 런 디렉토리 것을 그대로 읽는다 — 다시 나누면 문장이 달라진다.
    holdout = [data.Sentence(**r) for r in
               json.loads((run_dir / "data" / "train.json").read_text(encoding="utf-8"))
               ][args.train:]
    labels = json.loads((run_dir / "oracle_labels_train.json").read_text(encoding="utf-8"))
    # 라벨 인덱스는 train 분할 전체 기준이므로 오프셋을 맞춘다.
    labels = {t: per[args.train:] for t, per in labels.items()}

    gw = Gateway.from_args(args, budget=args.budget)
    cache = JsonCache(run_dir / "cache" / "segment.json")
    effort = None if args.seg_reasoning_effort == "none" else args.seg_reasoning_effort
    t_grid = cfg["t_grid"]

    def ev(pr):
        return evaluate(gw, pr, holdout, labels, cfg["spaced"], cfg["min_gap"], t_grid,
                        cache, args.workers, args.batch_size, effort)

    print(f"홀드아웃 {len(holdout)}문장 / T {t_grid} / min_gap {cfg['min_gap']}")
    print(f"기준 {len(base)}자 -> 변형 {len(variant)}자 (+{len(variant) - len(base)})")
    b_rows, b_m = ev(base)
    print(f"  기준: overlap {b_m['overlap']}  by_T {b_m['overlap_by_T']}  fmt {b_m['format_pass_rate']}")
    v_rows, v_m = ev(variant)
    print(f"  변형: overlap {v_m['overlap']}  by_T {v_m['overlap_by_T']}  fmt {v_m['format_pass_rate']}")
    d = paired(v_rows, b_rows, key="overlap")
    print(f"\n쌍체 Δ (변형 - 기준): {d['mean_delta']:+.4f} ± {d['se_delta']:.4f} "
          f"(n={d['n_pairs']}, {d['mean_delta'] / max(1e-9, d['se_delta']):+.1f} se)")
    print(f"비용 ${gw.usage.snapshot()['cost']:.4f}")
    cache.flush()
    out = run_dir / "rule_probe.json"
    out.write_text(json.dumps({"rule": add, "base_metrics": b_m, "variant_metrics": v_m,
                               "paired": d}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
