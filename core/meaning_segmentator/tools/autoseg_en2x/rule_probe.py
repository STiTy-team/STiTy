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

`--split dev` 는 홀드아웃 100문장 대신 dev 215문장에서 잰다. 홀드아웃 Δ 의 표준오차는
실측 ±0.028 이라 규칙 하나 크기의 효과를 못 가르고, dev 면 ±0.019 로 준다. 기준
프롬프트는 두 자리 모두 런에서 이미 재서 캐시에 있으므로 늘어나는 비용은 변형 쪽뿐이다.

**변형 여러 개를 견줄 때는 `--tag` 로 산출물 이름을 갈라야 한다.** 안 그러면 뒤 실행이
`rule_probe.json` 을 덮어쓴다.

가설은 **미리 하나로 정해 놓고** 재야 한다. 여러 변형을 만들어 그중 최고(또는 최저)를
고르면 그 값은 효과가 아니라 잡음이다 — 효과가 전부 0 인 변형 18개를 ±0.028 로 재면
최저값이 평균 −0.051 로 나온다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.meaning_segmentator.autoseg.infra.gateway import Gateway, add_provider_args
from core.meaning_segmentator.autoseg.loop_distill import evaluate, paired
from core.meaning_segmentator.tools.autoseg_en2x.retest import churn
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import data
from core.meaning_segmentator.autoseg.runtime.pipeline import JsonCache


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="기준 런 (분할·라벨·캐시를 그대로 쓴다)")
    p.add_argument("--rule-file", default=None, help="[Scoring Rules] 끝에 붙일 줄들")
    p.add_argument("--variant-prompt", default=None,
                   help="규칙을 끼우는 대신 이 파일을 통째로 변형으로 쓴다 — 순서만 바꾼 널 변형 같은, "
                        "삽입으로 못 만드는 변형용. --rule-file 과 둘 중 하나")
    p.add_argument("--k-samples", type=int, default=1,
                   help="문장당 채점 횟수 (loop_distill --k-samples 와 같은 캐시 규칙). 기준이 k 로 "
                        "재진 런이면 기준은 캐시 적중이고 변형만 k배 든다")
    p.add_argument("--prompt", default=None, help="기준 프롬프트. 기본은 런의 best_prompt.txt")
    p.add_argument("--train", type=int, default=100, help="배치 크기 — 홀드아웃은 그 뒤")
    p.add_argument("--split", choices=("holdout", "dev"), default="holdout",
                   help="재는 자리. holdout 은 train 분할의 배치 뒤쪽 100문장, dev 는 215문장 — "
                        "dev 쪽이 표준오차가 0.028 에서 0.019 로 준다 (기준은 어느 쪽도 캐시 적중)")
    p.add_argument("--tag", default=None, help="산출물 이름 꼬리표. 변형 여러 개를 같은 런에서 잴 때")
    p.add_argument("--save-rows", action="store_true",
                   help="변형의 문장별 행(점수·라벨·절단)을 rule_probe_<tag>_rows.json 에 남긴다 — "
                        "라벨 성분별(contra/adq) 상관 같은 후속 분석용")
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
    if bool(args.rule_file) == bool(args.variant_prompt):
        print("--rule-file 과 --variant-prompt 중 하나만", file=sys.stderr)
        return 1
    if args.variant_prompt:
        add = ""
        variant = Path(args.variant_prompt).read_text(encoding="utf-8")
    else:
        add = Path(args.rule_file).read_text(encoding="utf-8").strip()
        if args.section not in base:
            print(f"섹션 '{args.section}' 이 프롬프트에 없다", file=sys.stderr)
            return 1
        variant = base.replace(args.section, f"{add}\n\n{args.section}", 1)

    # 분할·라벨은 런 디렉토리 것을 그대로 읽는다 — 다시 나누면 문장이 달라진다.
    if args.split == "dev":
        sents = [data.Sentence(**r) for r in
                 json.loads((run_dir / "data" / "dev.json").read_text(encoding="utf-8"))]
        labels = json.loads((run_dir / "oracle_labels_dev.json").read_text(encoding="utf-8"))
    else:
        # 홀드아웃은 train 분할의 뒤쪽이다 — 앞 `--train` 문장은 Critic 이 사례를 뽑은 배치라
        # 거기서 재면 규칙을 만든 데이터로 그 규칙을 채점하게 된다.
        sents = [data.Sentence(**r) for r in
                 json.loads((run_dir / "data" / "train.json").read_text(encoding="utf-8"))
                 ][args.train:]
        labels = json.loads((run_dir / "oracle_labels_train.json").read_text(encoding="utf-8"))
        # 라벨 인덱스는 train 분할 전체 기준이므로 오프셋을 맞춘다.
        labels = {t: per[args.train:] for t, per in labels.items()}

    gw = Gateway.from_args(args, budget=args.budget)
    cache = JsonCache(run_dir / "cache" / "segment.json")
    effort = None if args.seg_reasoning_effort == "none" else args.seg_reasoning_effort
    t_grid = cfg["t_grid"]

    def ev(pr, what):
        r = evaluate(gw, pr, sents, labels, cfg["spaced"], cfg["min_gap"], t_grid,
                     cache, args.workers, args.batch_size, effort, k_samples=args.k_samples)
        u = gw.usage.snapshot()
        cache.flush()
        print(f"  [{what}] 누적 비용 ${u['cost']:.4f} / 호출 {u['calls']}", flush=True)
        return r

    print(f"{args.split} {len(sents)}문장 / T {t_grid} / min_gap {cfg['min_gap']}")
    print(f"기준 {len(base)}자 -> 변형 {len(variant)}자 (+{len(variant) - len(base)})")
    b_rows, b_m = ev(base, "기준")
    print(f"  기준: overlap {b_m['overlap']}  by_T {b_m['overlap_by_T']}  fmt {b_m['format_pass_rate']}")
    v_rows, v_m = ev(variant, "변형")
    print(f"  변형: overlap {v_m['overlap']}  by_T {v_m['overlap_by_T']}  fmt {v_m['format_pass_rate']}")
    d = paired(v_rows, b_rows, key="overlap")
    print(f"\n쌍체 Δ (변형 - 기준): {d['mean_delta']:+.4f} ± {d['se_delta']:.4f} "
          f"(n={d['n_pairs']}, {d['mean_delta'] / max(1e-9, d['se_delta']):+.1f} se)")
    # 변형이 기준의 절단을 얼마나 흔들었나 — 규칙이 겨눈 자리 수와 견줘 문안 교란을 읽는다.
    ch = churn(b_rows, v_rows)
    print(f"churn: (문장,T) {ch['changed']}/{ch['pairs']} ({ch['changed_rate']:.0%}) 바뀜, "
          f"오름 {ch['up']} 내림 {ch['down']}; 경계 {ch['moved']}/{ch['kept']} ({ch['moved_rate']:.0%}) 옮김")
    usage = gw.usage.snapshot()
    print(f"비용 ${usage['cost']:.4f}")
    cache.flush()
    out = run_dir / (f"rule_probe_{args.tag}.json" if args.tag else "rule_probe.json")
    out.write_text(json.dumps({"rule": add, "variant_prompt": args.variant_prompt,
                               "k_samples": args.k_samples, "split": args.split, "n": len(sents),
                               "base_prompt": args.prompt, "base_metrics": b_m,
                               "variant_metrics": v_m, "paired": d, "churn": ch, "usage": usage},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    if args.save_rows:
        rp = out.with_name(out.stem + "_rows.json")
        rp.write_text(json.dumps(v_rows, ensure_ascii=False), encoding="utf-8")
        print(f"-> {rp}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
