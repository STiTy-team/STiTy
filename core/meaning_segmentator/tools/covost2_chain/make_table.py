"""논문 표 — 지연 작동점별 COMET 을 LaTeX 로 낸다.

**작동점은 세 언어에 같은 값을 쓰는 것이 기본이다** (`--points 1200 1400 1600`).
언어마다 다른 값을 쓰면 각 칸이 다른 조건이 되어, 표 안에서 언어를 가로질러 읽는
독자가 없는 차이를 보게 된다. 결과를 보고 유리한 자리를 고르는 것은 더 나쁘다 —
그 표는 "이 방법이 낫다" 가 아니라 "유리한 자리를 골랐다" 를 말한다.

값은 **실측점 사이 선형 보간**이다. 비교군 실측점이 타깃당 48개(간격 중앙 28ms)라
보간 구간이 짧다. 작동점이 어느 실측점 사이인지 `--verbose` 로 확인할 수 있다.

`Prefix-match MU` 와 무분절은 지연대가 격자 밖이라 단일 점으로 적는다.

    python3 <이 파일> --points 1200 1400 1600 [--per-target zh:1100,1250,1360 ...]
"""
import argparse
import json
from pathlib import Path

A = Path(__file__).resolve().parents[2] / "experiment" / "artifacts" / "en2x" / "covost2"
GRID_ROWS = [("causal_align", r"Causal Align~\cite{koshkin-etal-2024-transllama}"),
             ("alignatt", r"AlignAtt~\cite{papi2023alignatt}"),
             ("syntax", r"SASST~\cite{yang2026sasst}")]
OURS = [("full_j44v0", "Multi-Agent (Ours; initial)"),
        ("full_j44best", "Multi-Agent (Ours; adopted)")]

p = argparse.ArgumentParser()
p.add_argument("--points", type=float, nargs=3, default=[1200, 1400, 1600],
               help="세 언어에 공통으로 쓸 지연 작동점 (ms)")
p.add_argument("--per-target", nargs="*", default=[],
               help="언어별로 다른 값을 쓸 때: zh:1100,1250,1360")
p.add_argument("--metric", default="comet", choices=("comet", "bleu"))
p.add_argument("--verbose", action="store_true", help="각 값이 어느 실측점 사이인지 찍는다")
a = p.parse_args()

per = {t: a.points for t in ("zh", "de", "ja")}
for spec in a.per_target:
    t, _, v = spec.partition(":")
    per[t] = [float(x) for x in v.split(",")]


def load(run: str, t: str) -> dict:
    """COMET 이 다 채워진 가장 최근 격자를 읽는다."""
    for d in ("bleu", "bleu_g11", "bleu_t6"):
        f = A / run / d / f"{t}.json"
        if f.exists():
            c = json.loads(f.read_text())["conditions"]
            if all(v.get(a.metric) is not None for v in c.values()):
                return c
    raise SystemExit(f"{run}/{t}: {a.metric} 가 채워진 격자가 없다")


def curve(c: dict, pick) -> list[tuple[float, float]]:
    return sorted((v["laal_ms"], v[a.metric]) for n, v in c.items() if pick(n))


def fam(f: str):
    return lambda n: n == f or n.startswith(f + "_T")


def ip(pts, x):
    for lo, hi in zip(pts, pts[1:]):
        if lo[0] <= x <= hi[0]:
            if hi[0] == lo[0]:
                return lo[1], lo, hi
            return lo[1] + (hi[1] - lo[1]) * (x - lo[0]) / (hi[0] - lo[0]), lo, hi
    return None, None, None


fmt = (lambda v: f"{v:.3f}".lstrip("0")) if a.metric == "comet" else (lambda v: f"{v:.1f}")
cols, body = {}, {}
for t in ("zh", "de", "ja"):
    v0 = load("full_j44v0", t)
    cols[t] = {"unseg": v0["unsegmented"], "mu": v0["mu_prefix"]}
    for key, _ in GRID_ROWS:
        body[(t, key)] = [ip(curve(v0, fam(key)), m)[0] for m in per[t]]
    for run, _ in OURS:
        c = v0 if run == "full_j44v0" else load(run, t)
        body[(t, run)] = [ip(curve(c, lambda n: n.startswith("auto_S")), m)[0] for m in per[t]]
    if a.verbose:
        for m in per[t]:
            for key, _ in GRID_ROWS:
                _, lo, hi = ip(curve(v0, fam(key)), m)
                print(f"# {t} {m:.0f}ms {key}: {lo[0]:.0f}~{hi[0]:.0f}ms 사이" if lo else
                      f"# {t} {m:.0f}ms {key}: 범위 밖")

keys = [k for k, _ in GRID_ROWS] + [r for r, _ in OURS]
best = {(t, i): max((body[(t, k)][i] for k in keys if body[(t, k)][i] is not None), default=None)
        for t in ("zh", "de", "ja") for i in range(3)}


def cell(t, k, i):
    v = body[(t, k)][i]
    if v is None:
        return "---"
    s = fmt(v)
    return rf"\textbf{{{s}}}" if abs(v - best[(t, i)]) < 1e-9 else s


L = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
     r"\renewcommand{\arraystretch}{1.05}", "",
     r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lccccccccc}", r"\toprule",
     r"& \multicolumn{3}{c}{EN--ZH}", r"& \multicolumn{3}{c}{EN--DE}",
     r"& \multicolumn{3}{c}{EN--JA} \\",
     r"\cmidrule(lr){2-4}", r"\cmidrule(lr){5-7}", r"\cmidrule(lr){8-10}", "",
     "Method",
     "& " + " & ".join(f"{m/1000:.2f}\\,s" for m in per["zh"]),
     "& " + " & ".join(f"{m/1000:.2f}\\,s" for m in per["de"]),
     "& " + " & ".join(f"{m/1000:.2f}\\,s" for m in per["ja"]) + r" \\",
     r"\midrule", ""]
for label, key in [("Full-sentence offline", "unseg"), (r"Prefix-match MU~\cite{zhang2020learning}", "mu")]:
    spans = " ".join(rf"& \multicolumn{{3}}{{c}}{{{fmt(cols[t][key][a.metric])} "
                     rf"({cols[t][key]['laal_ms']/1000:.1f}\,s)}}" for t in ("zh", "de", "ja"))
    L += [label, spans + r" \\", ""]
for key, label in GRID_ROWS + [(r, lab) for r, lab in OURS]:
    L += [label] + ["& " + " & ".join(cell(t, key, i) for i in range(3))
                    for t in ("zh", "de", "ja")]
    L[-1] += r" \\"
    L += [""]
L += [r"\bottomrule", r"\end{tabular}%", "}"]
print("\n".join(L))
