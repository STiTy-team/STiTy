"""점수 임계값 곡선 하나와 비교군 셋을 같은 축에 놓는다 — 타깃별 한 칸씩.

x 축은 **실측 LAAL** 이다. 우리 노브(점수 임계값)와 비교군 노브(`T`)는 단위가 아예
다르므로 노브 값으로는 나란히 놓을 수 없고, 둘 다 같은 자로 잰 지연으로 환산해야
한 그림이 된다.

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
p.add_argument("--out", default=None)
a = p.parse_args()


def curve(cond: dict, pick) -> list[tuple[float, float]]:
    """(LAAL 어절, 지표) 를 지연 오름차순으로. 값이 없는 조건은 빠진다."""
    return sorted((v["laal_words"], v[a.metric]) for n, v in cond.items()
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
    ax.annotate(f"unsegmented {fmt.format(unseg[a.metric])}  "
                f"(LAAL {unseg['laal_words']:.1f}w)",
                xy=(0.985, unseg[a.metric]), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="#5a5a5a")
    ax.set_title(f"en→{t}", fontsize=12)
    ax.set_xlabel("LAAL (source words)")
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
       A / runs[0][0] / "bleu" / f"tradeoff_score_grid_{a.metric}.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170)
print(out)
