"""두 런의 제안 곡선(`auto_T*`)을 한 파일에 모아 `plot_tradeoff` 가 겹쳐 그리게 한다.

`bleu_eval` 은 런마다 `bleu/<tgt>.json` 을 따로 쓴다. 같은 데이터·같은 비교군 위에서
프롬프트만 다른 두 런(예: 판정 루프의 채택본과 v0)을 한 그림에 넣으려면 조건 이름이
한 파일 안에서 갈라져 있어야 한다. 이 스크립트가 기준 런을 그대로 두고, 변형 런의
`auto_*` 조건만 다른 접두사로 옮겨 담는다.

    python -m core.meaning_segmentator.autoseg.scoring.merge_variant \\
        --base en2x/covost2/full_judge13 \\
        --variant en2x/covost2/full_judge13v0:auto_v0 \\
        --out en2x/covost2/full_judge13_cmp --targets zh de ja

**산출은 작도 전용이다.** 문장별 값(`hyps`, `comet_seg`)은 옮기지 않는다 — 조건 하나에
15,530문장이라 원본이 타깃당 40MB 가 넘고, 쌍체 부트스트랩·COMET 재채점은 원본 런의
파일에서 해야 출처가 분명하다. 여기 남는 것은 곡선을 그리는 데 쓰는 요약값뿐이다.

비교군 조건은 기준 런 것을 쓴다. 두 런이 같은 라벨·같은 캐시를 쓰면 값이 같아야 하므로,
`laal_ms` 나 지표가 어긋나면 경고를 찍는다 — 어긋났다는 것은 두 런이 실제로는 같은
조건에서 비교되고 있지 않다는 뜻이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..paths import RUNS_DIR

# 문장별 원본. 작도에 안 쓰이고 파일을 수십 MB 로 불린다.
HEAVY = ("hyps", "comet_seg")
TOL = {"laal_ms": 1.0, "bleu": 0.01, "comet": 0.001, "chrf2": 0.01}


def slim(cell: dict) -> dict:
    return {k: v for k, v in cell.items() if k not in HEAVY}


def main() -> int:
    ap = argparse.ArgumentParser(description="두 런의 제안 곡선을 한 파일로 모은다")
    ap.add_argument("--base", required=True, help="기준 런 id (비교군·상한을 여기서 가져온다)")
    ap.add_argument("--variant", required=True, action="append", metavar="RUN_ID:PREFIX",
                    help="겹쳐 그릴 런과 그 곡선에 붙일 접두사. 여러 번 줄 수 있다. "
                         "접두사는 `plot_tradeoff --variant` 에 그대로 넘긴다")
    ap.add_argument("--out", required=True, help="산출 런 id")
    ap.add_argument("--targets", nargs="+", default=["de", "ja", "zh"])
    ap.add_argument("--from-prefix", default="auto",
                    help="변형 런에서 옮겨 올 조건 접두사 (기본 auto — 제안 곡선)")
    a = ap.parse_args()

    out_dir = RUNS_DIR / a.out / "bleu"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tgt in a.targets:
        base = json.loads((RUNS_DIR / a.base / "bleu" / f"{tgt}.json").read_text(encoding="utf-8"))
        conds = {k: slim(v) for k, v in base["conditions"].items()}
        sources = {"base": a.base}

        for spec in a.variant:
            run_id, _, prefix = spec.rpartition(":")
            if not run_id or not prefix:
                raise SystemExit(f"--variant 는 RUN_ID:PREFIX 꼴이어야 한다: {spec}")
            var = json.loads((RUNS_DIR / run_id / "bleu" / f"{tgt}.json").read_text(encoding="utf-8"))
            if var["n"] != base["n"]:
                raise SystemExit(f"[{tgt}] 문장 수가 다르다 — {a.base} {base['n']} vs "
                                 f"{run_id} {var['n']}. 같은 표본이 아니면 겹쳐 그리면 안 된다")

            moved = []
            for name, cell in var["conditions"].items():
                if name == a.from_prefix or name.startswith(a.from_prefix + "_"):
                    conds[prefix + name[len(a.from_prefix):]] = slim(cell)
                    moved.append(name)
                elif name in base["conditions"]:
                    # 비교군·상한은 두 런이 같은 라벨을 공유한다. 값이 어긋나면 비교가 성립 안 한다.
                    for field, tol in TOL.items():
                        x, y = base["conditions"][name].get(field), cell.get(field)
                        if x is not None and y is not None and abs(x - y) > tol:
                            print(f"  !! [{tgt}] {name}.{field}: {a.base} {x} vs {run_id} {y}")
            if not moved:
                raise SystemExit(f"[{tgt}] {run_id} 에 `{a.from_prefix}` 곡선이 없다")
            sources[prefix] = run_id
            print(f"[{tgt}] {run_id} 의 {len(moved)}조건 → `{prefix}_T*`")

        out = {k: v for k, v in base.items() if k != "conditions"}
        out["sources"] = sources
        out["conditions"] = conds
        (out_dir / f"{tgt}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        print(f"[{tgt}] 조건 {len(conds)}개 → {out_dir / f'{tgt}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
