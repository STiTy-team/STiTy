"""2.3 집계 → results/<run_id>/quality_summary.{json,md}

모델 × 조건 × 묶음(ted en→ko, wmt24 en→ko, wmt24 ko→en). 문장 평균 COMET·MetricX(낮을수록 좋음)·CometKiwi,
코퍼스 chrF++, 그리고 같은 문장끼리 S0 대비 차이와 부트스트랩 95% 구간. 한국어 타깃은 형식 위반률도 낸다.
"""
import hashlib
import json
import random
import sys
from pathlib import Path
from statistics import mean

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tp.violations import check  # noqa: E402

GROUPS = [("ted", "en", "ko"), ("wmt24", "en", "ko"), ("wmt24", "ko", "en")]
METRICS = ("comet", "metricx", "kiwi")


def read_jsonl(path: Path):
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def key(src, hyp, ref):
    return hashlib.sha1(json.dumps([src, hyp, ref], ensure_ascii=False).encode()).hexdigest()


def ci(d, n_boot=2000, seed=7):
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return [ms[int(.025 * n_boot)], ms[int(.975 * n_boot)]]


def main():
    from sacrebleu.metrics import CHRF

    cfg = yaml.safe_load((HERE / "configs" / "quality.yml").read_text(encoding="utf-8"))
    run_dir = HERE / "results" / (sys.argv[1] if len(sys.argv) > 1 else cfg["run_id"])
    cache = {c["key"]: c for c in read_jsonl(run_dir / "scores_cache.jsonl")}
    chrf = CHRF(word_order=2)
    out, md = {}, []
    for mdir in sorted(p for p in run_dir.iterdir() if (p / "quality.jsonl").exists()):
        rows = read_jsonl(mdir / "quality.jsonl")
        for r in rows:
            if r["tgt_lang"] == "ko":
                r["violations"] = check(r["raw_output"], r["hyp"], r["src"], [], r["truncated"], None)
        conds = sorted({r["cond"] for r in rows})
        res = {}
        for ds, s, t in GROUPS:
            g = f"{ds}:{s}-{t}"
            per = {c: {r["id"]: r for r in rows if r["cond"] == c and r["dataset"] == ds and r["src_lang"] == s}
                   for c in conds}
            score = {c: {m: {i: cache.get(key(r["src"], r["hyp"], r["ref"]), {}).get(m) for i, r in v.items()}
                         for m in METRICS} for c, v in per.items()}
            res[g] = {}
            for c in conds:
                v = per[c]
                if not v:
                    continue
                e = {"n": len(v),
                     "chrfpp": chrf.corpus_score([r["hyp"] for r in v.values()], [[r["ref"] for r in v.values()]]).score}
                if t == "ko":
                    e["violation_rate"] = sum(bool(r["violations"]) for r in v.values()) / len(v)
                for m in METRICS:
                    vals = score[c][m]
                    if any(x is None for x in vals.values()):
                        continue
                    e[m] = mean(vals.values())
                    if c != "S0" and "S0" in score:
                        b = score["S0"][m]
                        d = [vals[i] - b[i] for i in vals if b.get(i) is not None]
                        if d:
                            e[f"{m}_delta"] = mean(d)
                            e[f"{m}_ci95"] = ci(d)
                res[g][c] = e
        out[mdir.name] = res

        md += [f"## {mdir.name}", ""]
        for g, cs in res.items():
            md += [f"### {g}", "",
                   "| 조건 | n | COMET | ΔCOMET (95%) | MetricX↓ | ΔMetricX (95%) | Kiwi | ΔKiwi | chrF++ | 위반률 |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
            for c, e in cs.items():
                def d(m, p=3):
                    if f"{m}_delta" not in e:
                        return "-"
                    lo, hi = e[f"{m}_ci95"]
                    star = "*" if lo > 0 or hi < 0 else ""
                    return f"{e[m + '_delta']:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]{star}"
                f = lambda k, p=3: "-" if e.get(k) is None else f"{e[k]:.{p}f}"  # noqa: E731
                md.append(f"| {c} | {e['n']} | {f('comet')} | {d('comet')} | {f('metricx', 2)} | {d('metricx', 2)} | "
                          f"{f('kiwi')} | {d('kiwi')} | {f('chrfpp', 1)} | {f('violation_rate')} |")
            md.append("")
    (run_dir / "quality_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "quality_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
