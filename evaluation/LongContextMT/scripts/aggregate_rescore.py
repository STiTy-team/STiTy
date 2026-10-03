"""rescore.py 결과 집계 → results/<run>/rescore/summary.{json,md}

정답(iwslt, sol) × 단위(sentence, block, doc) × 모델 × N. 문장·묶음은 평균과 N=0 대비 쌍체 차이
(같은 문장/묶음끼리)와 부트스트랩 95% 구간(단위를 다시 뽑는다). doc 는 값 하나라 차이만 낸다.
아직 없는 지표(GPU 단계 전)는 '-' 로 둔다.

    $METRICS_PY evaluation/LongContextMT/scripts/aggregate_rescore.py
"""
import argparse
import json
import random
from pathlib import Path
from statistics import mean

import yaml

HERE = Path(__file__).resolve().parents[1]
METRICS = {
    "sentence": ["comet", "xcomet", "metricx", "chrfpp", "spbleu"],
    "block": ["doc_comet", "metricx", "chrfpp", "spbleu"],
    "doc": ["chrfpp", "spbleu"],
}
LABEL = {"comet": "COMET", "xcomet": "XCOMET", "metricx": "MetricX↓", "chrfpp": "chrF++",
         "spbleu": "spBLEU", "doc_comet": "Doc-COMET"}
IDENT = {"sentence": "idx", "block": "start"}


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if path.exists() else []


def ci(d, n_boot=2000, seed=7):
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return [ms[int(.025 * n_boot)], ms[int(.975 * n_boot)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "configs" / "experiment.yml").read_text(encoding="utf-8"))
    out = (Path(args.run_dir) if args.run_dir else HERE / "results" / cfg["run_id"]) / "rescore"

    summary, md = {}, []
    for unit in ("sentence", "block", "doc"):
        rows = read_jsonl(out / f"scores_{unit}.jsonl")
        groups = {}
        for r in rows:
            groups.setdefault((r["ref"], r["model"]), {}).setdefault(r["n"], []).append(r)
        for (ref, model), by_n in sorted(groups.items()):
            res = {}
            base = by_n.get(0, [])
            for n in sorted(by_n):
                rs = by_n[n]
                e = {"units": len(rs)}
                for m in METRICS[unit]:
                    vals = [r.get(m) for r in rs]
                    if any(v is None for v in vals):
                        continue
                    e[m] = mean(vals)
                    if n == 0 or not base:
                        continue
                    if unit == "doc":
                        e[f"{m}_delta"] = e[m] - base[0][m]
                        continue
                    b = {r[IDENT[unit]]: r.get(m) for r in base}
                    d = [r[m] - b[r[IDENT[unit]]] for r in rs if b.get(r[IDENT[unit]]) is not None]
                    if d:
                        e[f"{m}_delta"] = mean(d)
                        e[f"{m}_ci95"] = ci(d)
                res[n] = e
            summary.setdefault(unit, {}).setdefault(ref, {})[model] = res

    for unit, by_ref in summary.items():
        md += [f"# {unit}", ""]
        for ref, by_model in by_ref.items():
            for model, res in by_model.items():
                ms = METRICS[unit]
                md += [f"## {unit} · ref={ref} · {model}", "",
                       "| N | 단위 수 | " + " | ".join(f"{LABEL[m]} | Δ{LABEL[m]}" for m in ms) + " |",
                       "|---|---|" + "---|---|" * len(ms)]
                for n, e in res.items():
                    cells = []
                    for m in ms:
                        p = 2 if m in ("metricx", "chrfpp", "spbleu") else 4
                        cells.append("-" if m not in e else f"{e[m]:.{p}f}")
                        if f"{m}_delta" not in e:
                            cells.append("-")
                        elif f"{m}_ci95" in e:
                            lo, hi = e[f"{m}_ci95"]
                            star = "*" if lo > 0 or hi < 0 else ""
                            cells.append(f"{e[m + '_delta']:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]{star}")
                        else:
                            cells.append(f"{e[m + '_delta']:+.{p}f}")
                    md.append(f"| {n} | {e['units']} | " + " | ".join(cells) + " |")
                md.append("")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
