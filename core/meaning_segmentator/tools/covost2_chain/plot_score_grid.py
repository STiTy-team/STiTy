"""점수 임계값 곡선 하나와 비교군 셋을 같은 축에 놓는다 — 타깃별 한 칸씩.

x 축은 **실측 LAAL** 이다. 우리 노브(점수 임계값)와 비교군 노브(`T`)는 단위가 아예
다르므로 노브 값으로는 나란히 놓을 수 없고, 둘 다 같은 자로 잰 지연으로 환산해야
한 그림이 된다.

**기본은 ms 다** (`--x ms`). 문헌의 LAAL 이 ms 이고, 청자가 겪는 지연도 시간이지
어절 수가 아니다. 두 축은 조건의 순서를 실제로 다르게 매긴다 — zh 실측으로
`auto_T4` 는 `syntax_T6` 보다 어절을 더 읽지만(4.18 대 3.92) 시간은 덜 쓴다
(1,389 대 1,434ms). 점수 기반 절단이 **짧게 발음되는 어절 뒤에** 떨어지기 때문이다.
어절 축으로 읽으면 이 이득이 통째로 사라져 판정이 뒤집힌다. `--x words` 는 강제정렬
타임스탬프가 없는 데이터셋용으로만 남긴다.

비교군은 `alignatt` · `mu_prefix` · `causal_align` · `syntax` 다. `punct` 는 뺐다 —
T 격자에 반응하지 않아 점 여럿이 LAAL 7.5 어절대에 겹쳐 쌓이기만 한다.

y 축은 `--metric` 이 고른다. **COMET 쪽이 주 근거다** — BLEU 는 토크나이저가 타깃마다
달라 달성률(분절/무분절)이 언어별로 73~88% 로 흩어지는데, 다국어 인코더 하나로 재는
COMET 에서는 92~95% 로 모인다.

    python3 <이 파일> --run full_j44v0 [--run full_j44best] [--metric comet]

`--run` 을 두 개 주면 실선/파선으로 겹쳐 그려 v0 와 채택본을 비교한다.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path(__file__).resolve().parents[2] / "experiment" / "artifacts" / "en2x" / "covost2"

# 비교군 셋 + 우리 것. 색은 명도까지 벌려 흑백 출력에서도 갈린다.
BASE = [("syntax", "#c2453a", "s", "syntax (SASST)"),
        ("causal_align", "#2f7d4f", "^", "causal_align"),
        ("alignatt", "#c98a1e", "v", "alignatt"),
        ("mu_prefix", "#7b5aa6", "D", "mu_prefix")]
OURS = ("#1a4f9c", "o", "auto (score threshold)")

p = argparse.ArgumentParser()
p.add_argument("--run", action="append", required=True,
               help="비교할 런 디렉토리 이름. 두 번까지 준다 (첫째 실선, 둘째 파선)")
p.add_argument("--targets", nargs="+", default=["zh", "de", "ja"])
p.add_argument("--metric", default="bleu", choices=("bleu", "comet"))
p.add_argument("--x", default="ms", choices=("ms", "words"),
               help="지연 축. ms 가 기본이자 문헌 표준 — words 는 타임스탬프가 없을 때만")
p.add_argument("--out", default=None)
a = p.parse_args()


XKEY = {"ms": "laal_ms", "words": "laal_words"}[a.x]
XLABEL = {"ms": "LAAL (ms of source audio)",
          "words": "LAAL (source words)"}[a.x]


def curve(cond: dict, pick) -> list[tuple[float, float]]:
    """(LAAL, 지표) 를 지연 오름차순으로. 값이 없는 조건은 빠진다."""
    return sorted((v[XKEY], v[a.metric]) for n, v in cond.items()
                  if pick(n) and v.get(a.metric) is not None)


def fam(name: str):
    return lambda n: n == name or n.startswith(name + "_T")


runs = [(r, {t: json.loads((A / r / "bleu" / f"{t}.json").read_text())["conditions"]
             for t in a.targets}) for r in a.run]

plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 10,
                     "axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff"})
fig, axes = plt.subplots(1, len(a.targets), figsize=(4.4 * len(a.targets), 4.3))
axes = [axes] if len(a.targets) == 1 else list(axes)

for ax, t in zip(axes, a.targets):
    cond0 = runs[0][1][t]
    # 비교군은 첫 런에서만 — 분절이 달라도 비교군은 같은 정책이라 값이 같다.
    for name, color, mk, label in BASE:
        pts = curve(cond0, fam(name))
        if pts:
            ax.plot(*zip(*pts), color=color, marker=mk, ms=4.5, lw=1.6,
                    label=label, zorder=3)
    for i, (rid, per) in enumerate(runs):
        pts = curve(per[t], lambda n: n.startswith("auto_S"))
        ax.plot(*zip(*pts), color=OURS[0], marker=OURS[1], ms=5, lw=2.2,
                ls="-" if i == 0 else "--", zorder=4,
                label=f"{OURS[2]} — {rid.replace('full_j44', '')}"
                      if len(runs) > 1 else OURS[2])
    unseg = cond0["unsegmented"]
    fmt = "{:.4f}" if a.metric == "comet" else "{:.1f}"
    ax.axhline(unseg[a.metric], color="#8a8a8a", ls=":", lw=1.2, zorder=1)
    ux = (f"{unseg['laal_ms']:.0f}ms" if a.x == "ms"
          else f"{unseg['laal_words']:.1f}w")
    ax.annotate(f"unsegmented {fmt.format(unseg[a.metric])}  (LAAL {ux})",
                xy=(0.985, unseg[a.metric]), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="#5a5a5a")
    ax.set_title(f"en→{t}", fontsize=12)
    ax.set_xlabel(XLABEL)
    ax.grid(True, color="#e8e7e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_ylabel(a.metric.upper())
axes[0].legend(fontsize=8.5, loc="upper left", framealpha=0.95)
fig.suptitle("CoVoST2 test 15,530 — quality vs latency"
             + ("  (BLEU tokenizers differ by target; read each panel on its own)"
                if a.metric == "bleu" else "  (COMET: one multilingual encoder)"),
             fontsize=10, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955))
out = (Path(a.out) if a.out else
       A / runs[0][0] / "bleu" / f"tradeoff_score_grid_{a.metric}_{a.x}.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170)
print(out)
