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
MAGENTA, BROWN, GREEN = "#c9268f", "#8a4b1a", "#008300"

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
# 마지막 칸은 점에 적을 노브 이름이다 — AlignAtt 는 `f`, MU 는 후보 수 `n` 이다.
NATIVE = [
    ("alignatt_native", ORANGE, "--", "^", "AlignAtt native f-sweep (Papi et al., 2023)",
     [("alignatt", 2), ("alignatt_f4", 4), ("alignatt_f6", 6), ("alignatt_f8", 8)], "f="),
    # 판정용 내부 NMT 를 평가 번역기와 같은 madlad 로 맞춘 라벨들 (`*_mad_*`).
    # 옛 NLLB 산출과는 한 곡선에 못 섞는다 — 판정 모델이 다르면 다른 정책이다.
    ("alignatt_mad", ORANGE, "--", "^", "AlignAtt, f knob (Papi et al., 2023)",
     [("alignatt_mad_f2", 2), ("alignatt_mad_f4", 4),
      ("alignatt_mad_f6", 6), ("alignatt_mad_f8", 8)], "f="),
    ("mu_prefix_mad", MAGENTA, "--", "v", "Prefix-match MU native n-sweep (Zhang et al., 2020)",
     [("mu_prefix_mad_n2", 2), ("mu_prefix_mad_n10", 10),
      ("mu_prefix_mad_n50", 50)], "n="),
    # Qwen3-ASR 파인튜닝 디코더에 텍스트만 흘려 P(<SEG>) ≥ θ 에서 자른다 (`gates/qwen_seg_baselines.py`).
    # 문장 끝을 안 보는 스트리밍 정책이고 노브는 임계값 θ 다.
    ("qwenseg", GREEN, "-", "o", "Qwen3-ASR <SEG> stream (ours, no training)",
     [(f"qwenseg_th{t}", t) for t in (0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)], "θ="),
    # judge13 채택 프롬프트(auto_judge13_iter3)의 auto_T — full_judge13_cmp 에서 `judge13_T*` 로 옮겨 온다
    # (`tools/autoseg_en2x/qwenseg/covost2_plot_all.sh`). 문장 전체를 보고 점수를 매긴다.
    ("judge13", BLUE, "-", "o", "judge13 prompt (LLM score), T",
     [(f"judge13_T{t}", t) for t in (2, 3, 4, 6)], "T="),
    # 같은 확률을 오프라인 점수로 — auto_T 와 같은 절단기(T, min_gap 1). 문장 끝을 봐야 한다
    # (`gates/qwen_seg_offline_baselines.py`). p0 는 앞만 본 점수, lrall 은 뒤 어절까지 본 점수.
    ("qwenp0", GREEN, "--", "s", "Qwen3-ASR P(<SEG>) offline, T",
     [(f"qwenp0_T{t}", t) for t in (2, 3, 4, 6)], "T="),
    ("qwenlrall", GREEN, ":", "D", "Qwen3-ASR <SEG> LR (full sentence), T",
     [(f"qwenlrall_T{t}", t) for t in (2, 3, 4, 6)], "T="),
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
_ap.add_argument("--variant", action="append", default=[], metavar="PREFIX:LABEL",
                 help="제안 곡선의 **변형**을 같은 패널에 겹쳐 그린다 (예: 같은 루프의 v0). "
                      "여러 번 줄 수 있다. 조건은 `<PREFIX>_T*` 로 한 파일 안에 있어야 "
                      "하며, 두 런의 산출을 합치는 것은 `merge_variant.py` 가 한다. "
                      "제안 곡선과 같은 색에 파선·빈 마커로 그린다 — 같은 정책의 다른 "
                      "프롬프트라는 뜻이고, 비교군 5색은 그대로 둔다")
_ap.add_argument("--legend-ncol", type=int, default=None,
                 help="범례 열 수. 기본은 패널 하나면 2, 둘이면 3, 셋 이상이면 4. 항목이 7개라 "
                      "3 을 주면 세 줄(3+3+1)로 접힌다 — 패널 하나짜리는 글자를 조금 줄여야 "
                      "폭 안에 든다")
_ap.add_argument("--drop", nargs="+", default=[],
                 help="그리지 않을 정책 (조건 접두사: punct / mu_prefix / syntax …). "
                      "빼면 x 범위·축약 구간·범례도 그 정책 없이 다시 잡힌다")
_ap.add_argument("--solid", action="store_true",
                 help="모든 곡선을 실선으로 그린다. 계열 구분은 색과 마커가 맡는다")
_ap.add_argument("--xlim", nargs="+", default=None, metavar="LO:HI",
                 help="패널마다 그릴 지연 구간(ms). `--targets` 와 같은 순서로 하나씩 준다 "
                      "(`1000:1400 850:1350 1250:1550`). 등지연 비교가 성립하는 구간만 "
                      "남기고 싶을 때 쓴다 — 곡선의 나머지 구간은 어느 정책도 상대가 "
                      "없어서 맞댈 수 없다. 주면 y 범위도 그 구간 안의 점으로만 잡고, "
                      "빈 구간 축약(`⋯`)은 하지 않는다")
_ap.add_argument("--xlim-cover", default=None, metavar="PREFIX",
                 help="이 정책의 **측정점 전부**가 들어가도록 지연 구간을 잡는다 "
                      "(`--xlim` 대신). 격자가 성긴 비교군에 맞춰 그릴 때 쓴다 — 그 정책의 "
                      "점을 하나도 빼지 않고, 다른 곡선은 `--xlim-snap` 으로 그 구간을 "
                      "덮는 점까지만 그린다. 패널마다 따로 계산한다")
_ap.add_argument("--xlim-snap", action="store_true",
                 help="`--xlim` 구간을 **바깥쪽 실측점까지** 넓힌다. 구간 끝에서 곡선을 "
                      "그냥 자르면 양 끝이 허공에서 시작·끝나 마커가 없는 선분이 되고, "
                      "보는 사람이 그 자리를 측정점으로 읽는다. 구간을 바로 넘어서는 "
                      "실측점을 한 개씩 넣어 곡선에 첫 점과 끝 점이 있게 한다")
_ap.add_argument("--font-scale", type=float, default=1.0,
                 help="글씨 전체 배율 — 눈금·축 제목·패널 제목·주석·범례에 같이 걸린다. "
                      "여백(왼쪽·아래)도 같은 비율로 따라 넓어져 라벨이 잘리지 않는다. "
                      "키울 때는 옆 패널 눈금 숫자와 부딪히지 않는지 보고 `--wspace` 를 "
                      "함께 올린다")
_ap.add_argument("--wspace", type=float, default=0.22,
                 help="패널 사이 가로 간격 (패널 폭 대비 비율). 키우면 옆 패널의 y축 "
                      "눈금 숫자와 이쪽 패널 오른쪽 끝이 덜 붙는다")
_ap.add_argument("--panel-width", type=float, default=5.5,
                 help="패널 하나의 폭(인치). 키우면 지연 축이 가로로 펴진다")
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
if ARGS.xlim and ARGS.xlim_cover:
    raise SystemExit("--xlim 과 --xlim-cover 는 함께 쓸 수 없다")
XLIM = None
if ARGS.xlim:
    if len(ARGS.xlim) != len(ARGS.targets):
        raise SystemExit(f"--xlim 은 타깃 수({len(ARGS.targets)})만큼 줘야 한다")
    XLIM = [tuple(float(v) for v in spec.split(":")) for spec in ARGS.xlim]
if ARGS.no_cite:
    def _strip(lbl):
        lbl = re.sub(r";?\s*[A-Z][A-Za-z]+ et al\., \d{4}", "", lbl)
        return re.sub(r"\s*\(\)", "", lbl)
    SERIES = [(*x[:4], _strip(x[4])) for x in SERIES]
    SINGLE = [(*x[:3], _strip(x[3])) for x in SINGLE]
    NATIVE = [(*x[:4], _strip(x[4]), x[5], x[6]) for x in NATIVE]
# 변형 곡선의 파선 패턴. 색은 제안 곡선과 같은 BLUE 로 두고 선 모양으로만 가른다.
_VDASH = [(0, (6, 3)), (0, (2, 2)), (0, (7, 2, 1, 2))]
VARIANTS = []
for _i, _spec in enumerate(ARGS.variant):
    _pre, _, _lbl = _spec.partition(":")
    VARIANTS.append((_pre, BLUE, _VDASH[_i % len(_VDASH)], "o", _lbl or _pre))
SERIES = [x for x in SERIES if x[0] not in ARGS.drop]
SINGLE = [x for x in SINGLE if x[0] not in ARGS.drop]
NATIVE = [x for x in NATIVE if x[0] not in ARGS.drop]
if ARGS.solid:
    SERIES = [(x[0], x[1], "-", *x[3:]) for x in SERIES]
    NATIVE = [(x[0], x[1], "-", *x[3:]) for x in NATIVE]
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
    "font.size": 11 * ARGS.font_scale,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 1.0,
    "axes.labelweight": _W, "axes.titleweight": _W,
    "xtick.labelsize": (11 if len(TARGETS) > 1 else 13.5) * ARGS.font_scale,
    "ytick.labelsize": (11 if len(TARGETS) > 1 else 13.5) * ARGS.font_scale,
    "font.weight": _W,
})
# 패널 하나짜리는 지면에서 한 칸을 통째로 차지하므로 글자를 한 단계 키운다.
_FS = (1.0 if len(TARGETS) > 1 else 1.25) * ARGS.font_scale
fig, axes = plt.subplots(
    1, len(TARGETS), squeeze=False,
    figsize=(ARGS.panel_width * len(TARGETS) if len(TARGETS) > 1 else 9.2,
             5.9 if len(TARGETS) > 1 else 6.6))
axes = axes[0]
_SINGLE = len(TARGETS) == 1
_L = (0.095 if M == "comet" else 0.075) if not _SINGLE else \
     (0.11 if M == "comet" else 0.095)
# 글씨를 키우면 y축 제목·눈금이 그만큼 왼쪽으로 번진다. 여백이 그대로면 잘린다.
_L *= 1 + 0.5 * (ARGS.font_scale - 1)
_TOP = 0.90
# 패널 하나짜리는 폭이 좁아 범례를 2열로 접어야 한다 — 3열이면 긴 라벨(TransLLaMa 인용)이
# 그림 밖으로 나간다. 그만큼 아래 여백을 더 준다.
_BOTTOM = (0.255 if _SINGLE else 0.235) * (1 + 0.6 * (ARGS.font_scale - 1))
fig.subplots_adjust(left=_L, right=0.985, top=_TOP,
                    bottom=_BOTTOM, wspace=ARGS.wspace)


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


def snap(curve_xs, lo, hi):
    """구간 [lo, hi] 를 **곡선마다** 바로 바깥의 실측점까지 넓힌다. 그래야 모든 곡선이
    구간 안에 첫 점과 끝 점을 갖는다 — 넓히지 않으면 곡선이 허공에서 시작·끝나고,
    마커가 없는 그 자리를 측정점으로 읽게 된다."""
    for xs in curve_xs:
        lo = min(lo, max((x for x in xs if x <= lo), default=lo))
        hi = max(hi, min((x for x in xs if x >= hi), default=hi))
    return lo, hi


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


for _pi, (ax, tgt) in enumerate(zip(axes, TARGETS)):
    C = blobs[tgt]["conditions"]
    unseg = C["unsegmented"]
    fmt = (lambda v: f"{v:.1f}") if M == "bleu" else (lambda v: f"{v:.3f}")
    pad = 1.3 if M == "bleu" else 0.012

    ax.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)

    _native = [] if ARGS.no_native else NATIVE
    # **창을 그리기 전에 정하고 데이터를 잘라서 그린다.** 축 범위로만 자르면 창 밖의
    # 점으로 가는 선분이 반쯤 남아, 마커 없는 선이 축 끝에서 흘러나간다.
    # `--xlim-snap` 은 곡선마다 창 바로 바깥의 점을 한 개씩 끌어와 첫 점·끝 점을 만든다.
    _WIN = None
    if XLIM or ARGS.xlim_cover:
        _all_xs = [[x for x, _, _ in pts] for pts in
                   ([curve(C, p) for p, *_ in SERIES]
                    + [native_curve(C, e) for *_h, e, _k in _native]
                    + [curve(C, p) for p, *_ in VARIANTS]) if pts]
        if ARGS.xlim_cover:
            _cov = curve(C, ARGS.xlim_cover) or next(
                (native_curve(C, e) for pre, *_h, e, _k in _native
                 if pre == ARGS.xlim_cover), [])
            if not _cov:
                raise SystemExit(f"--xlim-cover {ARGS.xlim_cover}: [{tgt}] 에 점이 없다")
            _base = (_cov[0][0], _cov[-1][0])
        else:
            _base = XLIM[_pi]
        _WIN = snap(_all_xs, *_base) if ARGS.xlim_snap else _base

    def clip(pts):
        return pts if _WIN is None else [q for q in pts if _WIN[0] <= q[0] <= _WIN[1]]

    for si, (prefix, color, ls, mk, label) in enumerate(SERIES):
        pts = clip(curve(C, prefix))
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

    for vi, (prefix, color, ls, mk, label) in enumerate(VARIANTS):
        pts = clip(curve(C, prefix))
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker=mk,
                color=color, lw=2.1, ms=6.5, mfc="none", mew=1.6, zorder=5,
                label=label)
        # **`ours` 모드에서는 변형에 노브 라벨을 안 적는다.** 변형은 같은 정책의 다른
        # 프롬프트라 점이 기준 곡선과 거의 겹치는데, 양쪽에 같은 수를 적으면 글자만
        # 두 겹으로 쌓인다. 계열 구분은 빈 마커와 파선이 한다.
        if ARGS.point_labels == "all":
            label_points(ax, pts, color, "T", -10 - vi * 11)

    for prefix, color, ls, mk, label, entries, knob in _native:
        pts = clip(native_curve(C, entries))
        if len(pts) < 2:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                color=color, lw=2.3, ms=7.0, mfc="none", mew=1.6, zorder=6,
                label=label)
        if ARGS.point_labels != "none":
            label_points(ax, pts, color, knob, -11)

    for prefix, color, mk, label in SINGLE:
        c = C.get(prefix)
        if not c or c.get("laal_ms") is None:
            continue
        ax.plot([c["laal_ms"]], [c[M]], marker=mk, ls="none", color=color,
                ms=9, mew=0, zorder=5, label=label)

    single = [C[p] for p, *_ in SINGLE
              if p in C and C[p].get("laal_ms") is not None]
    _nat_pts = [q for *_h, e, _k in _native for q in clip(native_curve(C, e))]
    _var_pts = [q for p, *_ in VARIANTS for q in clip(curve(C, p))]
    # **xs 와 ys 는 같은 점을 같은 순서로 담는다** — `--xlim` 이 둘을 zip 해서 구간
    # 안의 점만 고르므로, 한쪽에만 원소를 더하면 짝이 어긋난다. 상한은 점이 아니라
    # 가로선이라 ys 에만 들어가고, 그래서 맨 뒤에 붙인다.
    _pts = ([(x, y) for p, *_ in SERIES for x, y, _ in clip(curve(C, p))]
            + [(c["laal_ms"], c[M]) for c in single]
            + [(x, y) for x, y, _ in _nat_pts]
            + [(x, y) for x, y, _ in _var_pts])
    xs = [x for x, _ in _pts]
    ys = [y for _, y in _pts]
    _ceil_in_ylim = [unseg[M]] if ARGS.ceiling_in_ylim else []
    ys = ys + _ceil_in_ylim
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
    if _WIN:
        span = _WIN[1] - _WIN[0]
        # 끝점 마커가 축선에 걸리지 않게 양쪽에 여유를 준다.
        xlo, xhi = _WIN[0] - span * 0.03, _WIN[1] + span * 0.03

    # 점이 없는 넓은 구간을 축약한다 — 관심 구간(경쟁 정책들)이 짓눌리지 않도록.
    # 구간을 직접 지정한 경우는 이미 관심 구간만 남아 축약할 것이 없다.
    gaps = [] if _WIN else find_gaps(xs, xlo, xhi)
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
    ax.set_ylabel(M.upper(), fontsize=12 * _FS, labelpad=8)
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
                      fontsize=(((10.5 if _NCOL >= 3 else 12.5) if _SINGLE else 11.5)
                                * ARGS.font_scale),
                      handlelength=2.6, handletextpad=0.7,
                      columnspacing=(1.1 if _NCOL >= 3 else 1.6) if _SINGLE else 2.0,
                      labelspacing=0.7, bbox_to_anchor=(0.5, 0.005))
    for _t in _leg.get_texts():
        _t.set_fontweight(_W)
        _t.set_color(INK)
    # 범례가 3줄을 넘으면 아래 여백이 모자라 축 제목을 덮는다 — 넘는 줄만큼만 늘린다 (3줄 이하는 종전 그대로)
    _rows = -(-len(l) // _NCOL)
    if _rows > 3 and not _SINGLE:
        fig.subplots_adjust(bottom=_BOTTOM + 0.055 * (_rows - 3) * ARGS.font_scale)
# 패널 하나짜리는 범례가 그림 폭을 정하므로 여백을 그려진 것에 맞춰 잘라낸다.
# 다중 패널은 종전 여백을 유지한다 (기존 산출물과 크기가 달라지지 않게).
_SAVE = dict(bbox_inches="tight", pad_inches=0.04) if _SINGLE else {}
fig.savefig(d / f"{STEM}.png", dpi=200, facecolor=SURFACE, **_SAVE)
fig.savefig(d / f"{STEM}.pdf", facecolor=SURFACE, **_SAVE)
print("saved", d / f"{STEM}.png")
