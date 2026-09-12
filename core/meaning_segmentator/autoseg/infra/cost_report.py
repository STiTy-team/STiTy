"""런 하나가 실제로 쓴 LLM 비용을 모아 본다 — 예산 감시용.

게이트웨이 usage 스냅샷은 산출물마다 흩어져 있다(`prompt_eval/*.json`, `history.json`,
`iter_*/metrics.json`). 크래시한 실행은 스냅샷을 못 남기므로 **분절 캐시 증분으로 역산**한
추정치도 함께 낸다 — 그 차이가 곧 "기록되지 않은 지출"이다.

    python -m core.meaning_segmentator.autoseg.infra.cost_report --run-id en-multi/clean500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..paths import RUNS_DIR


def _loop_point(name: str) -> float | None:
    """루프 누적 스냅샷이면 그 시점(이터레이션 번호), 아니면 None.

    루프는 게이트웨이 usage 를 **런 누적값**으로 매 이터레이션 덮어 쓴다. 같은 시점이
    `history.json[i]` 와 `iter_i/metrics.json` 두 곳에, 마지막 시점이 `final_report.json`
    에 한 번 더 실린다 — 즉 한 번 쓴 돈이 산출물에 세 벌 있다. 그냥 더하면 run16 에서
    실지출 $7.18 이 $39.67 로 보고된다(3배 남짓). 계열 안에서는 **최댓값 하나만** 쓴다.
    """
    if name == "final_report.json":
        return float("inf")
    m = re.fullmatch(r"history\.json\[(\d+)\]", name)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"iter_(\d+)[/\\]metrics\.json", name)
    if m:
        return float(m.group(1))
    return None


def fold_cumulative(snaps: list[tuple[str, dict]]) -> tuple[float, set[str]]:
    """(집계 비용, 합계에 실제로 든 스냅샷 이름들).

    루프 누적 계열은 시점 순으로 걸으며 **비용이 줄어드는 자리를 실행 경계로** 본다
    (`--resume` 은 게이트웨이를 새로 만들어 0 부터 다시 센다). 구간마다 최댓값을 하나씩
    취해 더한다. 루프 밖 산출물(`prompt_eval/*` 등)은 각자 독립 실행이라 그대로 더한다.
    """
    loop: dict[float, tuple[str, float]] = {}
    total, counted = 0.0, set()
    for name, u in snaps:
        cost = float(u.get("cost", 0.0))
        t = _loop_point(name)
        if t is None:
            total += cost
            counted.add(name)
        elif t not in loop or cost > loop[t][1]:
            loop[t] = (name, cost)
    seg: list[tuple[str, float]] = []
    for t in sorted(loop):
        name, cost = loop[t]
        if seg and cost < seg[-1][1]:
            total += seg[-1][1]
            counted.add(seg[-1][0])
            seg = []
        seg.append((name, cost))
    if seg:
        total += seg[-1][1]
        counted.add(seg[-1][0])
    return total, counted


def usage_snapshots(run_dir: Path) -> list[tuple[str, dict]]:
    out = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name in ("config.json", "language_profile.json", "measured_profile.json"):
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(blob, dict) and isinstance(blob.get("usage"), dict):
            out.append((str(path.relative_to(run_dir)), blob["usage"]))
        elif isinstance(blob, list):
            for i, item in enumerate(blob):
                if isinstance(item, dict) and isinstance(item.get("usage"), dict):
                    out.append((f"{path.relative_to(run_dir)}[{i}]", item["usage"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="런의 LLM 비용 집계")
    p.add_argument("--run-id", required=True)
    p.add_argument("--budget", type=float, default=None, help="넘으면 비영 종료코드")
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_id
    snaps = usage_snapshots(run_dir)
    total, counted = fold_cumulative(snaps)
    print(f"== {args.run_id}")
    for name, u in snaps:
        mark = "*" if name in counted else " "
        print(f" {mark}{name:44s} 호출 {u.get('calls', 0):5d}  ${float(u.get('cost', 0.0)):8.4f}")
        for k, v in sorted(u.get("by_purpose", {}).items(), key=lambda x: -x[1]["cost"]):
            print(f"      {k:18s} {v['calls']:5d}콜  ${v['cost']:7.4f}")
    print(f"  {'* 표시가 합계에 든 것 (나머지는 같은 지출의 중간 스냅샷)':44s}")
    print(f"  {'기록된 합계':44s} {'':10s} ${total:8.4f}")

    seg = run_dir / "cache" / "segment.json"
    if seg.exists():
        n = len(json.loads(seg.read_text(encoding="utf-8")))
        rate = (total / n) if n else 0.0
        print(f"\n  분절 캐시 {n}건 → 기록 기준 문장당 ${rate:.5f}")
        print("  **캐시 건수가 기록된 분절 수보다 많으면 그 차이는 스냅샷을 못 남긴 "
              "(크래시한) 실행이 쓴 돈이다.**")
    if args.budget is not None and total > args.budget:
        print(f"\n[초과] ${total:.4f} > 예산 ${args.budget:.4f}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
