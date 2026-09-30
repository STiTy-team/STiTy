"""en→X 등지연 비교표 (LaTeX). 정책마다 T 격자가 달라 **지연축에서 선형 보간**해 맞댄다.

각 타깃에 지연 지점 3개를 두고, 정책 곡선(`<정책>_T*` 의 `(laal_ms, COMET)`)을 그 지점에서
보간한다. 측정 범위 밖은 외삽하지 않고 `--` 로 둔다 — 범위를 벗어난 정책은 그 지연대에
시스템으로 존재하지 않는다는 뜻이라, 외삽하면 없는 점을 지어내게 된다.

지연대가 아예 겹치지 않는 것들(prefix-match MU, 구두점, 무분절 상한)은 표 아래에 자기
지연과 함께 점 하나로 싣는다. MU 는 **가장 높은 지연 지점에 제일 가까운 점**을 쓴다.

    python core/meaning_segmentator/tools/covost2_chain/en2x_table.py \\
        --run-id en2x/covost2/full_judge13_cmp

입력은 `merge_variant.py` 로 합친 런이다 — 한 파일 안에 채택본(`auto_*`)과 v0(`auto_v0_*`)가
같이 있어야 두 행이 같은 비교군·같은 상한 위에서 나온다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.meaning_segmentator.autoseg.paths import RUNS_DIR  # noqa: E402

# (조건 줄기, 표에 적을 이름). 줄기는 **노브 글자까지** 포함한다 — 정책마다 노브가
# 달라서(`auto_T4`, `qwenseg_th0.05`) 접두사만으로는 어느 축을 쓸어 만든 곡선인지
# 갈리지 않고, 한 정책의 서로 다른 노브가 한 곡선으로 섞일 수 있다.
# 순서가 표의 행 순서다. `--row` 로 통째로 갈아끼울 수 있다.
ROWS = [
    ("alignatt_T", r"AlignAtt~\cite{papi-2023}"),
    ("causal_align_T", r"Causal Align~\cite{koshkin-etal-2024-transllama}"),
    ("syntax_T", r"SASST~\cite{yang2026sasst}"),
    ("auto_v0_T", r"Multi-Agent (Ours; initial)"),
    ("auto_T", r"Multi-Agent (Ours; adopted)"),
]
# 지연대가 안 겹치는 것들. (조건, 이름, 곡선인가)
OUTSIDE = [
    ("mu_prefix_T", r"Prefix-match MU~\cite{zhang2020learning}", True),
    ("punct", r"Punctuation", False),
    ("unsegmented", r"Full-sentence offline", False),
]
BANDS = {"zh": [1000, 1200, 1400], "de": [850, 1100, 1350], "ja": [1250, 1400, 1550]}


def interp(pts: list[tuple[float, float]], x: float) -> float | None:
    """측정점 사이 선형 보간. 범위 밖은 None (외삽하지 않는다)."""
    pts = sorted(pts)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def curve(C: dict, stem: str, metric: str) -> list[tuple[float, float]]:
    """`<줄기><수>` 꼴 조건만 모아 (지연, 품질) 로 정렬한다. 수는 정수(`_T4`)도
    소수(`_th0.05`)도 된다."""
    pat = re.compile(r"^" + re.escape(stem) + r"[0-9]+(?:\.[0-9]+)?$")
    return sorted((C[n]["laal_ms"], C[n][metric]) for n in C
                  if pat.match(n)
                  and C[n].get("laal_ms") is not None and C[n].get(metric) is not None)


def fmt(v: float | None) -> str:
    """선행 0 을 뗀 소수 세 자리 (`.684`). 표가 좁아 관례를 따른다."""
    return "--" if v is None else f"{v:.3f}".lstrip("0")


def main() -> int:
    ap = argparse.ArgumentParser(description="en→X 등지연 비교표 (LaTeX)")
    ap.add_argument("--run-id", required=True, help="merge_variant 로 합친 런 id")
    ap.add_argument("--targets", nargs="+", default=["zh", "de", "ja"])
    ap.add_argument("--metric", default="comet", choices=["comet", "bleu"])
    ap.add_argument("--out", default=None, help="기본: <런>/table_en2x_<metric>.tex")
    ap.add_argument("--row", action="append", default=[], metavar="STEM:LABEL",
                    help="본문 행을 통째로 갈아끼운다. 줄기는 노브 글자까지 적는다 "
                         "(`qwenseg_th:ASR-decoder threshold`). 여러 번 줄 수 있고, "
                         "준 순서가 행 순서다")
    ap.add_argument("--outside", nargs="+", default=None, metavar="COND",
                    help="표 아래 '지연대 밖' 항목을 고른다 (기본 전부). 이름은 조건 "
                         "이름이다 (mu_prefix_T / punct / unsegmented)")
    ap.add_argument("--outside-first", action="store_true",
                    help="지연대 밖 항목을 본문 **위**에 놓고 머리글 줄을 뺀다. 남는 것이 "
                         "무분절 상한 하나뿐이면 '지연대 밖' 이라는 머리글이 오히려 "
                         "상한을 경쟁 정책처럼 보이게 한다")
    a = ap.parse_args()

    rows = ROWS
    if a.row:
        rows = [(spec.partition(":")[0], spec.partition(":")[2] or spec.partition(":")[0])
                for spec in a.row]
    outside = OUTSIDE
    if a.outside is not None:
        outside = [x for x in OUTSIDE if x[0] in a.outside]
        unknown = set(a.outside) - {x[0] for x in OUTSIDE}
        if unknown:
            raise SystemExit(f"모르는 지연대 밖 항목: {sorted(unknown)}")

    d = RUNS_DIR / a.run_id / "bleu"
    blobs = {t: json.loads((d / f"{t}.json").read_text(encoding="utf-8")) for t in a.targets}
    n = {b["n"] for b in blobs.values()}
    if len(n) != 1:
        raise SystemExit(f"타깃마다 문장 수가 다르다: {n}")

    def outside_lines() -> list[str]:
        out: list[str] = []
        for cond, name, is_curve in outside:
            out.append("")
            out.append(name)
            for t in a.targets:
                C = blobs[t]["conditions"]
                if is_curve:
                    # 맞댈 수 없는 정책은 **가장 높은 지연 지점에 제일 가까운 점**을 싣는다.
                    pts = curve(C, cond, a.metric)
                    ms, v = min(pts, key=lambda p: abs(p[0] - BANDS[t][-1]))
                else:
                    ms, v = C[cond]["laal_ms"], C[cond][a.metric]
                out.append(r"& \multicolumn{3}{c}{%s (%.1f\,s)}" % (fmt(v), ms / 1000))
            out[-1] += r" \\"
        return out

    # 본문 값: [행][타깃][지연]
    vals = {pre: {t: [interp(curve(blobs[t]["conditions"], pre, a.metric), x)
                      for x in BANDS[t]] for t in a.targets} for pre, _ in rows}
    best = {t: [max((vals[pre][t][i] for pre, _ in rows
                     if vals[pre][t][i] is not None), default=None)
                for i in range(3)] for t in a.targets}

    L = [f"% table_en->x.tex — {a.run_id}, n={n.pop()}, {a.metric}",
         "% 생성: core/meaning_segmentator/tools/covost2_chain/en2x_table.py",
         r"% Requires: \usepackage{booktabs}",
         r"%           \usepackage{graphicx}",
         "",
         r"\scriptsize",
         r"\setlength{\tabcolsep}{2.5pt}",
         r"\renewcommand{\arraystretch}{1.05}",
         "",
         r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{l" + "ccc" * len(a.targets) + "}",
         r"\toprule",
         "& " + "\n& ".join(r"\multicolumn{3}{c}{EN--%s}" % t.upper() for t in a.targets)
         + r" \\"]
    L += [r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(a.targets))]
    L += ["", "Method"]
    # 0.85 은 두 자리, 1.0 은 한 자리 — 원표 표기를 따른다. 소수점까지 지우면 "1 s" 가 돼
    # 다른 칸과 자릿수가 어긋나므로 꼬리 0 하나만 뗀다.
    def _sec(x):
        s = "%.2f" % (x / 1000)
        return s[:-1] if s.endswith("0") else s
    L += ["& " + " & ".join(r"%s\,s" % _sec(x) for x in BANDS[t]) for t in a.targets]
    L[-1] += r" \\"
    L.append(r"\midrule")
    if outside and a.outside_first:
        L += outside_lines()

    for pre, name in rows:
        L.append("")
        L.append(name)
        for t in a.targets:
            cells = []
            for i, v in enumerate(vals[pre][t]):
                s = fmt(v)
                if v is not None and best[t][i] is not None and abs(v - best[t][i]) < 1e-9:
                    s = r"\textbf{%s}" % s
                cells.append(s)
            L.append("& " + " & ".join(cells))
        L[-1] += r" \\"

    if outside and not a.outside_first:
        L += ["", r"\midrule",
              r"\multicolumn{%d}{l}{" % (1 + 3 * len(a.targets)),
              r"\textit{Outside matched band (own latency in parentheses)}",
              r"} \\"]
        L += outside_lines()

    L += ["", r"\bottomrule", r"\end{tabular}%", "}"]

    out = Path(a.out) if a.out else RUNS_DIR / a.run_id / f"table_en2x_{a.metric}.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"→ {out}")
    for t in a.targets:
        rng = {pre: curve(blobs[t]["conditions"], pre, a.metric) for pre, _ in rows}
        print(f"[{t}] 지연 지점 {BANDS[t]}ms / 측정 범위 "
              + ", ".join(f"{p} {r[0][0]:.0f}-{r[-1][0]:.0f}" for p, r in rng.items() if r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
