"""2.4 집계 → results/<run_id>/format_summary.{json,md}

- 위반률: 모델 × 조건 × 문맥(0/16) × 세트(일반·스트레스). 한 행에 위반이 하나라도 있으면 위반 행.
- 위반 유형별 건수, 스트레스 범주별 위반률.
- 품질(일반 문장만, 정답 있음): COMET·MetricX·CometKiwi·chrF++ 와 F0 대비 쌍체 차이(부트스트랩 95%).
- 스트레스 세트는 CometKiwi 만 (정답 없음).
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

import sys

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tp.violations import _norm as norm, check  # noqa: E402


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
    cfg = yaml.safe_load((HERE / "configs" / "format.yml").read_text(encoding="utf-8"))
    run_dir = HERE / "results" / cfg["run_id"]
    cache = {c["key"]: c for c in read_jsonl(run_dir / "scores_cache.jsonl")}
    out, md = {}, []
    for mdir in sorted(p for p in run_dir.iterdir() if (p / "format.jsonl").exists()):
        rows = read_jsonl(mdir / "format.jsonl")
        # 판정기를 고친 뒤에도 같은 기준이 되도록 저장된 판정 대신 다시 판정한다. context_echo 는 문맥이
        # 저장돼 있지 않아 저장된 값을 쓰되, 같은 조건의 N=0 출력이 같으면 앞에 비슷한 원문이 있었던 것이라 뺀다.
        chain = {c["idx"]: c["hyp"] for c in read_jsonl(
            (HERE / cfg["context_source"]).resolve() / mdir.name / "translations.jsonl") if c["n"] == 16}
        n0 = {(r["cond"], r["id"]): r["hyp"] for r in rows if r["ctx_n"] == 0}
        for r in rows:
            prev = [chain[j] for j in range(max(0, r["pos"] - r["ctx_n"]), r["pos"])] if r["ctx_n"] else []
            v = [x for x in check(r["raw_output"], r["hyp"], r["src"], prev, r["truncated"], r["json_status"])
                 if x != "context_echo"]
            base = n0.get((r["cond"], r["id"]), "")
            lines = [ln.strip(" -") for ln in r["hyp"].splitlines() if ln.strip()]
            for ln in lines:
                sim = max((SequenceMatcher(None, norm(ln), norm(p)).ratio() for p in prev if len(norm(p)) >= 8),
                          default=0)
                if sim >= 0.8 and SequenceMatcher(None, norm(ln), norm(base)).ratio() < 0.8:
                    v.append("context_echo" if sim == 1 else "context_echo_fuzzy")
                    break
            r["violations"] = v
        groups = defaultdict(list)
        for r in rows:
            groups[(r["cond"], r["ctx_n"])].append(r)
        res = {}
        base = {}
        for (cond, n), rs in sorted(groups.items()):
            g = {}
            for st in ("talk", "stress"):
                sub = [r for r in rs if r["set"] == st]
                if not sub:
                    continue
                g[st] = {"rows": len(sub),
                         "violation_rate": sum(bool(r["violations"]) for r in sub) / len(sub),
                         "by_type": dict(Counter(v for r in sub for v in r["violations"]))}
                kiwi = [cache.get(key(r["src"], r["hyp"], r["ref"]), {}).get("kiwi") for r in sub]
                if all(k is not None for k in kiwi):
                    g[st]["kiwi"] = mean(kiwi)
            stress = [r for r in rs if r["set"] == "stress"]
            g["stress_by_category"] = {
                c: sum(bool(r["violations"]) for r in stress if r["category"] == c)
                / max(1, sum(1 for r in stress if r["category"] == c))
                for c in sorted({r["category"] for r in stress})}
            talk = {r["id"]: r for r in rs if r["set"] == "talk"}
            for m in ("comet", "metricx", "chrfpp"):
                vals = {i: cache.get(key(r["src"], r["hyp"], r["ref"]), {}).get(m) for i, r in talk.items()}
                if talk and all(v is not None for v in vals.values()):
                    g.setdefault("talk", {})[m] = mean(vals.values())
                    if cond == "F0":
                        base[(n, m)] = vals
                    elif (n, m) in base:
                        d = [vals[i] - base[(n, m)][i] for i in vals if i in base[(n, m)]]
                        g["talk"][f"{m}_delta_vs_F0"] = mean(d)
                        g["talk"][f"{m}_delta_ci95"] = ci(d)
            res[f"{cond}@N{n}"] = g
        out[mdir.name] = res

        md += [f"## {mdir.name}", "",
               "| 조건 | N | 위반률 일반 | 위반률 스트레스 | COMET | ΔCOMET vs F0 | MetricX↓ | ΔMetricX vs F0 | Kiwi 스트레스 |",
               "|---|---|---|---|---|---|---|---|---|"]
        f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"  # noqa: E731

        def dl(t, m):
            if f"{m}_delta_vs_F0" not in t:
                return "-"
            lo, hi = t[f"{m}_delta_ci95"]
            star = "*" if lo > 0 or hi < 0 else ""
            return f"{t[m + '_delta_vs_F0']:+.3f}{star}"
        for name, g in res.items():
            cond, n = name.split("@N")
            t, s = g.get("talk", {}), g.get("stress", {})
            md.append(f"| {cond} | {n} | {f(t.get('violation_rate'))} | {f(s.get('violation_rate'))} | "
                      f"{f(t.get('comet'))} | {dl(t, 'comet')} | {f(t.get('metricx'), 2)} | {dl(t, 'metricx')} | "
                      f"{f(s.get('kiwi'))} |")
        md += ["", "위반 유형별 건수 (일반+스트레스)", "", "| 조건 | N | 유형별 |", "|---|---|---|"]
        for name, g in res.items():
            cond, n = name.split("@N")
            c = Counter()
            for st in ("talk", "stress"):
                c.update(g.get(st, {}).get("by_type", {}))
            md.append(f"| {cond} | {n} | " + ", ".join(f"{k} {v}" for k, v in c.most_common()) + " |")
        md += ["", "스트레스 범주별 위반률", "",
               "| 조건 | N | " + " | ".join(next(iter(res.values()))["stress_by_category"]) + " |",
               "|---|---|" + "---|" * len(next(iter(res.values()))["stress_by_category"])]
        for name, g in res.items():
            cond, n = name.split("@N")
            md.append(f"| {cond} | {n} | " + " | ".join(f"{v:.2f}" for v in g["stress_by_category"].values()) + " |")
        md.append("")
    (run_dir / "format_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "format_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
