"""품질–지연 트레이드오프 — 정책별 곡선. x축 ms LAAL, y축 단일(비축약) 축.

정책 6종을 모두 한 축에 그린다. `punct` 와 `mu_prefix` 는 다른 정책보다 훨씬 느린
지연대에 있어 맞대결(같은 지연에서의 품질 비교)은 성립하지 않지만, 곡선이 어디로
향하는지 보여주므로 그림에는 남긴다 — 지연대가 겹치지 않는다는 사실 자체가 결과다.

offline 상한(무분절 통번역)은 가로 파선으로만 표시한다. 상한의 지연은 정책들보다
2~5배 커서 x 범위에 넣으면 관심 구간이 짓눌리므로, 값과 지연은 주석으로 적는다.

`punct` 는 T 격자에 반응하지 않아 곡선이 아니라 점 하나로 그린다 (아래 `SINGLE` 주석).

`punct`/`mu_prefix` 가 멀리 떨어져 있어 x축 가운데에 점이 하나도 없는 빈 구간이
생긴다. 그 구간은 `FuncScale` 조각선형 변환으로 축약하고 `⋯` 로 표시한다 — 눈금은
축약 구간 안쪽을 빼고 다시 잡는다. 점의 x값 자체는 건드리지 않는다.

색 6종은 all-pairs CIEDE2000 × {정상, 적/녹/청색맹} 검증본이다. 최소 ΔE 11.1
(auto↔causal, 청색맹)로, 기존 4색본과 동일한 하한을 유지한다.
"""
import argparse
import json
import re
from pathlib import Path

# **import 하면 안 된다 — 모듈을 읽는 것만으로 그림을 덮어쓴다.** 인자 파싱부터 저장까지
# 전부 모듈 최상위에서 돌기 때문이다. 산출물이 git 에 추적되므로 실수로 import 하면
# 추적 파일이 조용히 바뀐다 (실제로 한 번 그랬다). 스크립트로만 실행할 것.
if __name__ != "__main__":
    raise RuntimeError(
        f"{__name__} 는 스크립트다 — import 하면 그림 산출물을 덮어쓴다. "
        "`python3 <파일경로>` 로 실행할 것")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator

SURFACE = "#ffffff"   # 순백 배경 — 논문 지면/슬라이드에서 회색 판이 보이지 않게
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
MAGENTA, BROWN = "#c9268f", "#8a4b1a"

SERIES = [
    ("auto",         BLUE,    "-",  "o", "Multi-agent loop (ours)"),
    ("causal_align", AQUA,    "-",  "s", "Causal align (TransLLaMa; Koshkin et al., 2024)"),
    ("alignatt",     ORANGE,  "-",  "^", "AlignAtt (Papi et al., 2023)"),
    ("syntax",       VIOLET,  "-",  "D", "SASST (Yang et al., 2026)"),
    ("mu_prefix",    MAGENTA, "-",  "v", "Prefix-match MU (Zhang et al., 2020)"),
]
# **`punct` 는 곡선이 아니라 점 하나다.** `coarsen` 은 경계를 *지우기만* 하므로 정책이
# 예산보다 적게 찍으면 T 를 바꿔도 산출이 그대로다. 구두점은 원래 성기게 찍어서
# (k=1.6~2.1) T 격자가 지연을 만들지 못하고, T 를 키우면 남은 경계마저 지워져
# 무분절 쪽으로 끌려갈 뿐이다 (FLEURS de: k 2.07→1.67, laal 4889→5093ms).
# 그 점들을 이어 그리면 없는 노브가 있는 것처럼 보인다.
SINGLE = [("punct", BROWN, "X", "Punctuation (no latency knob)")]
# **네이티브 노브 곡선.** 위 SERIES 는 정책 라벨 한 벌에 우리 노브 `T` 를 얹은
# `coarsen` 판이라 축이 우리 것이다. 이쪽은 그 정책 **원논문 노브**를 직접 쓸어 만든
# 라벨들이다 — AlignAtt 는 `f`(최근 f 어절). f 를 바꾸면 강제 디코딩 경로가 통째로
# 달라지므로 f 마다 라벨을 새로 만들어야 하고 사후 병합으로는 못 만든다.
# (조건 이름, 노브 값). 조건이 하나도 없으면 그냥 안 그린다.
NATIVE = [
    ("alignatt_native", ORANGE, "--", "^", "AlignAtt native f-sweep (Papi et al., 2023)",
     [("alignatt", 2), ("alignatt_f4", 4), ("alignatt_f6", 6), ("alignatt_f8", 8)]),
]
T_GRID = [4, 6, 8, 12]   # 기본값. 실제로는 아래에서 blob 의 조건 이름으로 덮어쓴다
GAP_MIN = 0.20   # 이보다 넓은 빈 구간만 축약 (전체 x 폭 대비).
                 # 0.12 로 내리면 경쟁 정책 사이의 정상적인 간격까지 잘린다.
GAP_KEEP = 0.045  # 축약 후 남길 폭
GAP_PAD = 0.015   # 축약 구간 양끝에 남길 여유 (마커가 잘리지 않게)


def find_gaps(xs, lo, hi):
    """점이 하나도 없는 넓은 x 구간을 찾는다."""
    span = hi - lo
    out, pts = [], sorted(set(xs))
    for a, b in zip(pts, pts[1:]):
        if b - a > GAP_MIN * span:
            out.append((a + GAP_PAD * span, b - GAP_PAD * span,
                        GAP_KEEP * span))
    return out


def gap_scale(gaps):
    """축약 구간을 좁히는 조각선형 정변환/역변환."""
    def fwd(x):
        x = np.asarray(x, dtype=float)
        y = np.array(x, dtype=float)
        for a, b, w in gaps:
            f = w / (b - a)
            y = y - np.where(x >= b, (b - a) - w,
                             np.where(x > a, (x - a) * (1 - f), 0.0))
        return y

    tb = [(float(fwd(a)), float(fwd(a)) + w) for a, b, w in gaps]

    def inv(y):
        y = np.asarray(y, dtype=float)
        x = np.array(y, dtype=float)
        for (a, b, w), (A, B) in zip(gaps, tb):
            k = (b - a) / w
            x = x + np.where(y >= B, (b - a) - w,
                             np.where(y > A, (y - A) * (k - 1), 0.0))
        return x

    return fwd, inv

_ap = argparse.ArgumentParser(description="품질–지연 곡선")
_ap.add_argument("--metric", default="bleu", choices=["bleu", "comet"],
                 help="y축 품질 지표. comet 은 `comet_score.py` 를 먼저 돌려야 한다")
_ap.add_argument("--out", default=None, help="출력 파일 stem (기본: tradeoff[_comet])")
_ap.add_argument("--run-id", default="covost2/full")
_ap.add_argument("--targets", nargs="+", default=["de", "ja"])
_ap.add_argument("--title", default="unseen FLEURS 500")
_ap.add_argument("--t-grid", nargs="+", type=int, default=None,
                 help="그릴 T 값. 기본은 데이터에 있는 격자의 **앞 5개** — 비교군은 T6 "
                      "부터 포화해 15ms 안에 겹치므로 뒤쪽을 다 그리면 ours 만 길어져 "
                      "축이 늘어나고 점 간격이 안 보인다")
_ap.add_argument("--point-labels", default="ours",
                 choices=["all", "ends", "ours", "native", "none"],
                 help="각 점에 노브 값을 적는다. all=모든 곡선의 모든 T (겹치는 패널에선 "
                      "글자가 엉킨다), ends=곡선마다 양 끝 T 만, ours=제안 곡선만 전부, "
                      "native=네이티브 f 만. 어느 모드든 네이티브 f 는 항상 적는다")
_ap.add_argument("--no-native", action="store_true",
                 help="네이티브 노브 곡선을 안 그린다")
_ap.add_argument("--no-header", action="store_true",
                 help="상단 제목·설명 문단을 안 그린다. 논문/슬라이드에 캡션이 따로 붙는 "
                      "경우 그림 안의 제목은 중복이고 패널 높이만 먹는다")
_ap.add_argument("--legend-ncol", type=int, default=None,
                 help="범례 열 수. 기본은 패널 하나면 2, 둘이면 3, 셋 이상이면 4. 항목이 7개라 "
                      "3 을 주면 세 줄(3+3+1)로 접힌다 — 패널 하나짜리는 글자를 조금 줄여야 "
                      "폭 안에 든다")
_ap.add_argument("--drop", nargs="+", default=[],
                 help="그리지 않을 정책 (조건 접두사: punct / mu_prefix / syntax …). "
                      "빼면 x 범위·축약 구간·범례도 그 정책 없이 다시 잡힌다")
_ap.add_argument("--short-ylabel", action="store_true",
                 help="y축 라벨을 지표 이름만(BLEU / COMET) 적는다. 언어쌍은 패널 제목에, "
                      "채점 모델은 캡션에 있으므로 논문 그림에서는 중복이다")
_ap.add_argument("--no-cite", action="store_true",
                 help="범례에서 인용 표기(`Papi et al., 2023`)를 뺀다. 본문 캡션에 인용이 "
                      "붙는 논문 그림용")
_ap.add_argument("--serif", action="store_true",
                 help="Times 계열 세리프(Nimbus Roman → Liberation Serif → STIX)와 STIX "
                      "수식 폰트로 그린다. 본문이 Times 인 논문용. 굵기는 그대로 볼드다")
_ap.add_argument("--no-legend", action="store_true",
                 help="범례를 그리지 않는다. 캡션이 계열을 설명하는 논문 그림용")
_ap.add_argument("--ceiling-in-ylim", action="store_true",
                 help="offline 상한을 y 범위에 포함시킨다. 기본은 제외 — 상한이 높아서 "
                      "포함하면 곡선이 아래로 눌려 점 간격이 안 보인다")
ARGS = _ap.parse_args()
M = ARGS.metric
if ARGS.no_cite:
    def _strip(lbl):
        lbl = re.sub(r";?\s*[A-Z][A-Za-z]+ et al\., \d{4}", "", lbl)
        return re.sub(r"\s*\(\)", "", lbl)
    SERIES = [(*x[:4], _strip(x[4])) for x in SERIES]
    SINGLE = [(*x[:3], _strip(x[3])) for x in SINGLE]
    NATIVE = [(*x[:4], _strip(x[4]), x[5]) for x in NATIVE]
SERIES = [x for x in SERIES if x[0] not in ARGS.drop]
SINGLE = [x for x in SINGLE if x[0] not in ARGS.drop]
NATIVE = [x for x in NATIVE if x[0] not in ARGS.drop]
STEM = ARGS.out or ("tradeoff" if M == "bleu" else f"tradeoff_{M}")

d = Path("core/meaning_segmentator/experiment/artifacts") / ARGS.run_id / "bleu"
TARGETS = ARGS.targets
blobs = {t: json.loads((d / f"{t}.json").read_text(encoding="utf-8"))
         for t in TARGETS}
missing = [t for t, b in blobs.items()
           if any(M not in c for c in b["conditions"].values())]
if missing:
    raise SystemExit(f"{missing} 에 `{M}` 값이 없다 — comet_score.py 를 먼저 돌릴 것")

# 글씨는 전부 잉크색 볼드다 — 회색(INK2) 라벨은 축소 인쇄와 빔프로젝터에서 먼저
# 사라진다. INK2 는 이제 보조 선(상한 파선·절단 표시)에만 남는다.
_W = "bold"
plt.rcParams.update({
    # 세리프 목록 끝의 DejaVu Sans 는 글리프 폴백용이다 — Nimbus Roman 에 '→' 가 없다.
    "font.family": (["Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Sans"]
                    if ARGS.serif else "DejaVu Sans"),
    "mathtext.fontset": "stix" if ARGS.serif else "dejavusans",
    # 논문 PDF 는 Type 3 폰트를 거부하는 곳이 많다 (IEEE/ACM 검사기). 세리프 모드는 TrueType 으로 심는다.
    "pdf.fonttype": 42 if ARGS.serif else 3,
    "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 1.0,
    "axes.labelweight": _W, "axes.titleweight": _W,
    "xtick.labelsize": 11 if len(TARGETS) > 1 else 13.5,
    "ytick.labelsize": 11 if len(TARGETS) > 1 else 13.5,
    "font.weight": _W,
})
# 패널 하나짜리는 지면에서 한 칸을 통째로 차지하므로 글자를 한 단계 키운다.
_FS = 1.0 if len(TARGETS) > 1 else 1.25
fig, axes = plt.subplots(
    1, len(TARGETS), squeeze=False,
    figsize=(5.5 * len(TARGETS) if len(TARGETS) > 1 else 9.2,
             5.9 if len(TARGETS) > 1 else 6.6))
axes = axes[0]
_SINGLE = len(TARGETS) == 1
_L = (0.095 if M == "comet" else 0.075) if not _SINGLE else \
     (0.11 if M == "comet" else 0.095)
_TOP = 0.90 if ARGS.no_header else (0.815 if M == "comet" else 0.845)
# 패널 하나짜리는 폭이 좁아 범례를 2열로 접어야 한다 — 3열이면 긴 라벨(TransLLaMa 인용)이
# 그림 밖으로 나간다. 그만큼 아래 여백을 더 준다.
fig.subplots_adjust(left=_L, right=0.985, top=_TOP,
                    bottom=0.255 if _SINGLE else 0.235, wspace=0.22)


# **격자를 데이터에서 읽는다.** 종전에는 위 상수가 그대로 쓰여 T=2,3,5,7,10 점이
# 그려지지 않았고, x 축 범위도 그려진 점에서만 뽑히므로 저지연 구간이 통째로 사라졌다.
_seen = set()
for _b in blobs.values():
    for _n in _b["conditions"]:
        _m = re.match(r"^.*_T(\d+)$", _n)
        if _m:
            _seen.add(int(_m.group(1)))
if _seen:
    T_GRID = sorted(_seen)
if ARGS.t_grid:
    T_GRID = sorted(ARGS.t_grid)
elif len(T_GRID) > 5:
    T_GRID = T_GRID[:5]
print(f"[plot] T 격자 = {T_GRID}")
_TSTR = "/".join(str(x) for x in T_GRID)


def native_curve(C, entries):
    """(조건 이름, 노브 값) 목록에서 있는 것만 (x, y, 노브) 로 뽑는다."""
    pts = [(C[n]["laal_ms"], C[n][M], k) for n, k in entries
           if n in C and C[n].get("laal_ms") is not None and C[n].get(M) is not None]
    return sorted(pts)


def label_points(ax, pts, color, prefix, dy):
    """노브 값 라벨. **색은 안 쓴다** — 계열 색으로 적으면 축소 시 글자가 뭉개진다.
    계열 구분은 마커가 하고 글자는 읽히는 것이 우선이다."""
    for x, y, v in pts:
        ax.annotate(f"{prefix}{v}", (x, y), textcoords="offset points",
                    xytext=(0, dy), fontsize=9.5 * _FS, color=INK, fontweight=_W,
                    ha="center", va="bottom" if dy > 0 else "top", zorder=7)


def _knob_label(c: dict, T) -> str:
    """점에 적을 노브 값. **실현 평균 조각 크기**(`piece_units`, bleu_eval 산출)가 있으면
    그걸 소수 한 자리로 적고, 없으면(옛 산출) 목표 T 를 적는다. T 는 `round(길이/T)` 로
    조각 수를 정하는 노브라 실현값과 0.1~0.9 어긋난다 (de-en T=4 → 3.8, zh-en T=21 → 16.4).
    논문 표의 T 열도 실현값이므로 그림과 같은 수가 보여야 한다."""
    pu = c.get("piece_units")
    return f"{pu:.1f}" if pu is not None else str(T)


def curve(C, prefix):
    """`(x, y, 라벨)` 목록. **라벨을 같이 돌려준다** — 조건이 빠질 수 있고 큰 T 는 포화해
    지연이 역전되기도 해서, 점 순서로 T 를 되짚으면 라벨이 어긋난다."""
    pts = [(C[f"{prefix}_T{T}"]["laal_ms"], C[f"{prefix}_T{T}"][M],
            _knob_label(C[f"{prefix}_T{T}"], T))
           for T in T_GRID
           if f"{prefix}_T{T}" in C and C[f"{prefix}_T{T}"].get("laal_ms") is not None]
    return sorted(pts)


for ax, tgt in zip(axes, TARGETS):
    C = blobs[tgt]["conditions"]
    unseg = C["unsegmented"]
    fmt = (lambda v: f"{v:.1f}") if M == "bleu" else (lambda v: f"{v:.3f}")
    pad = 1.3 if M == "bleu" else 0.012

    ax.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)

    for si, (prefix, color, ls, mk, label) in enumerate(SERIES):
        pts = curve(C, prefix)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                color=color, lw=2.3, ms=6.5, mew=0, zorder=5,
                label=label)
        # 곡선이 겹치는 패널(en→de)에서 라벨끼리 붙지 않게 위/아래를 번갈아 둔다.
        _dy = 9 if si % 2 == 0 else -10
        if ARGS.point_labels == "all" or (ARGS.point_labels == "ours"
                                          and prefix == "auto"):
            label_points(ax, pts, color, "T", _dy)
        elif ARGS.point_labels == "ends" and len(pts) >= 2:
            label_points(ax, [pts[0], pts[-1]], color, "T", _dy)

    _native = [] if ARGS.no_native else NATIVE
    for prefix, color, ls, mk, label, entries in _native:
        pts = native_curve(C, entries)
        if len(pts) < 2:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                color=color, lw=2.3, ms=7.0, mfc="none", mew=1.6, zorder=6,
                label=label)
        if ARGS.point_labels != "none":
            label_points(ax, pts, color, "f=", -11)

    for prefix, color, mk, label in SINGLE:
        c = C.get(prefix)
        if not c or c.get("laal_ms") is None:
            continue
        ax.plot([c["laal_ms"]], [c[M]], marker=mk, ls="none", color=color,
                ms=9, mew=0, zorder=5, label=label)

    single = [C[p] for p, *_ in SINGLE
              if p in C and C[p].get("laal_ms") is not None]
    _nat_pts = [q for *_h, e in _native for q in native_curve(C, e)]
    ys = ([y for p, *_ in SERIES for _, y, _ in curve(C, p)]
          + [c[M] for c in single] + [y for _, y, _ in _nat_pts]
          + ([unseg[M]] if ARGS.ceiling_in_ylim else []))
    xs = ([x for p, *_ in SERIES for x, _, _ in curve(C, p)]
          + [c["laal_ms"] for c in single] + [x for x, _, _ in _nat_pts])
    ylo, yhi = min(ys) - pad, max(ys) + pad * 1.6
    # offline 상한 — gtx 통번역을 데이터셋 정답 번역으로 채점한 값.
    # 상한의 지연(x)은 축 밖이라 선으로만 긋고 값·지연은 주석으로 적는다.
    # 상한이 y 범위 위로 벗어나면 선이 안 보이고 주석만 축 밖으로 나가 패널 제목과
    # 겹친다 (en→de 에서 실제로 그랬다: 상한 39.9, 데이터 최대 35.4). 데이터 폭의
    # 25% 이내로 벗어난 경우에만 축을 넓혀 안에 넣고, 그보다 멀면 넓히는 순간 곡선이
    # 눌리므로 상한을 아예 안 그린다 (--ceiling-in-ylim 으로 강제 포함 가능).
    _ceil = unseg[M]
    _show_ceil = True
    if _ceil > yhi:
        if ARGS.ceiling_in_ylim or _ceil - yhi <= (yhi - ylo) * 0.25:
            yhi = _ceil + pad * 0.9
        else:
            _show_ceil = False
    ax.set_ylim(ylo, yhi)

    if _show_ceil:
        ax.axhline(_ceil, color=INK, lw=1.8, ls=(0, (5, 3)), zorder=3,
                   label="Full-sentence offline (ceiling)")
        ax.annotate(f"offline ceiling {fmt(_ceil)} @ {unseg['laal_ms'] / 1000:.1f}s "
                    f"(no segmentation)",
                    (0.015, _ceil), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, -15),
                    color=INK, fontsize=9 * _FS, fontweight=_W, zorder=6)
    span = max(xs) - min(xs)
    # 패널 하나짜리는 오른쪽 여백을 줄인다 — 다중 패널의 0.11 은 punct 마커와 축약
    # 표시가 들어갈 자리였고, 없으면 파선 상한만 빈 데로 한 마디 더 뻗는다.
    xlo, xhi = ((min(xs) - span * 0.04, max(xs) + span * 0.02) if _SINGLE
                else (min(xs) - span * 0.06, max(xs) + span * 0.11))

    # 점이 없는 넓은 구간을 축약한다 — 관심 구간(경쟁 정책들)이 짓눌리지 않도록.
    gaps = find_gaps(xs, xlo, xhi)
    if gaps:
        fwd, inv = gap_scale(gaps)
        ax.set_xscale("function", functions=(fwd, inv))
        # 눈금은 축약 구간 안쪽을 빼고 다시 잡는다.
        ticks = [t for t in MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10])
                 .tick_values(xlo, xhi)
                 if xlo - span * 0.02 <= t <= xhi + span * 0.02
                 and not any(a < t < b for a, b, _ in gaps)]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xlim(xlo, xhi)

    if gaps:
        f0, f1 = (fwd(xlo), fwd(xhi))
        for a, b, _w in gaps:
            fr = (fwd((a + b) / 2) - f0) / (f1 - f0)
            for e in (a, b):   # 축약 구간의 양 끝 — 점선으로 표시
                ax.axvline(e, color=INK2, lw=0.8, ls=(0, (2, 2)),
                           alpha=0.55, zorder=2)
            for dx in (-0.006, 0.006):   # 축 위의 절단 표시
                ax.plot([fr + dx - 0.007, fr + dx + 0.007], [-0.013, 0.013],
                        transform=ax.transAxes, color=INK2, lw=1.1,
                        clip_on=False, zorder=10)
    ax.set_xlabel("LAAL (ms of source audio)", fontsize=12 * _FS, labelpad=8)
    if ARGS.short_ylabel:
        ax.set_ylabel(M.upper(), fontsize=12 * _FS, labelpad=8)
    else:
        ylab = (f"BLEU  (EN→{tgt.upper()}, {blobs[tgt]['tokenize']})" if M == "bleu"
                else f"COMET  (EN→{tgt.upper()}, wmt22-comet-da)")
        ax.set_ylabel(ylab, fontsize=12)
        ax.yaxis.set_label_coords((-0.135 if M == "comet" else -0.085)
                                  * (1.0 if len(TARGETS) > 1 else 0.68), 0.5)
    ax.set_title(f"EN→{tgt.upper()}", loc="left", fontsize=14 * _FS,
                 fontweight=_W, pad=8)
    ax.tick_params(length=4, width=1.0)

# 상한을 못 그린 패널이 첫 칸일 수 있으므로 범례는 전 패널에서 모아 중복만 뺀다.
h, l = [], []
for _ax in axes:
    for _h, _l in zip(*_ax.get_legend_handles_labels()):
        if _l not in l:
            h.append(_h)
            l.append(_l)
_CEIL_LBL = "Full-sentence offline (ceiling)"
if _CEIL_LBL in l:   # 상한은 종전대로 맨 앞에 둔다
    _i = l.index(_CEIL_LBL)
    h.insert(0, h.pop(_i))
    l.insert(0, l.pop(_i))
_NCOL = ARGS.legend_ncol or (2 if _SINGLE else (3 if len(TARGETS) < 3 else 4))
if not ARGS.no_legend:
    _leg = fig.legend(h, l, loc="lower center", ncol=_NCOL,
                      frameon=False,
                      fontsize=(10.5 if _NCOL >= 3 else 12.5) if _SINGLE else 11.5,
                      handlelength=2.6, handletextpad=0.7,
                      columnspacing=(1.1 if _NCOL >= 3 else 1.6) if _SINGLE else 2.0,
                      labelspacing=0.7, bbox_to_anchor=(0.5, 0.005))
    for _t in _leg.get_texts():
        _t.set_fontweight(_W)
        _t.set_color(INK)
# **번역기 이름을 결과에서 읽는다.** 종전에는 "gtx" 가 제목에 박혀 있어서, madlad 로 잰
# 그림이 스스로를 gtx 라고 말했다 (`bleu_eval` 리포트에 있던 것과 같은 종류의 사고다).
_trs = sorted({b.get("translator", "?").split(":")[1] if b.get("translator", "").startswith("local:")
               else b.get("translator", "?").split(":")[0] for b in blobs.values()})
_MT = "/".join(t.split("/")[-1] for t in _trs)
if not ARGS.no_header:
    fig.text(0.008, 0.985, f"{'BLEU' if M == 'bleu' else 'COMET'}–latency trade-off on "
             f"{ARGS.title}"
             + ((f" (same translator, {_MT}; T = {_TSTR} per curve)")
                if len(TARGETS) > 1 else f"  ·  {_MT}, T = {_TSTR}"),
             ha="left", va="top", fontweight=_W,
             fontsize=12.5 if len(TARGETS) > 1 else 12.0)
    fig.text(0.008, 0.945,
             "Upper-left is better. LAAL is forced-aligned (Qwen3-ForcedAligner; wav2vec2 CTC "
             "agrees within 22 ms). Empty x ranges are compressed (break marks on the axis)."
             "\nPunctuation has no latency knob (it segments below the T budget), so it is one "
             "point; it and prefix-match MU sit in a slower band — shown, not matched.\n"
             + ("BLEU is NOT comparable across panels (de 13a, ja ja-mecab)."
                if M == "bleu" else
                "COMET uses one multilingual encoder, so panels are far more comparable "
                "than under BLEU — but it stays reference-based.")
             if len(TARGETS) > 1 else
             ("Punctuation has no latency knob (it segments below the T budget), so it is one "
              "point, not a curve.\nIt and prefix-match MU sit in a slower latency band — shown, "
              "but not matched head-to-head.\nEmpty x ranges are compressed (break marks on the "
              "axis). "
              + ("BLEU tokenisation: " + blobs[TARGETS[0]]["tokenize"] if M == "bleu"
                 else "COMET: wmt22-comet-da (reference-based).")),
             ha="left", va="top", fontsize=7.5, color=INK2, linespacing=1.6)
# 패널 하나짜리는 범례가 그림 폭을 정하므로 여백을 그려진 것에 맞춰 잘라낸다.
# 다중 패널은 종전 여백을 유지한다 (기존 산출물과 크기가 달라지지 않게).
_SAVE = dict(bbox_inches="tight", pad_inches=0.04) if _SINGLE else {}
fig.savefig(d / f"{STEM}.png", dpi=200, facecolor=SURFACE, **_SAVE)
fig.savefig(d / f"{STEM}.pdf", facecolor=SURFACE, **_SAVE)
print("saved", d / f"{STEM}.png")
