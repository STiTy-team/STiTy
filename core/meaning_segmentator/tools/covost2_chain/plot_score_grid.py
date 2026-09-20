"""점수 임계값 곡선 하나와 비교군 셋을 같은 축에 놓는다 — 타깃별 한 칸씩.

x 축은 **실측 LAAL** 이다. 우리 노브(점수 임계값)와 비교군 노브(`T`)는 단위가 아예
다르므로 노브 값으로는 나란히 놓을 수 없고, 둘 다 같은 자로 잰 지연으로 환산해야
한 그림이 된다.

비교군은 `mu_prefix` · `causal_align` · `syntax` 셋만 그린다. `punct` 는 지연대가
2배 떨어져 있어(LAAL 7어절대) 같은 x 범위에 넣으면 관심 구간이 짓눌린다 — 무분절과
함께 가로 파선으로만 적는다.

    python3 <이 파일> --run full_j44v0 [--run full_j44best] [--out <경로>]

두 개를 주면 실선/파선으로 겹쳐 그려 v0 와 채택본을 비교한다.
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
        ("mu_prefix", "#7b5aa6", "D", "mu_prefix")]
OURS = ("#1a4f9c", "o", "auto (score threshold)")

p = argparse.ArgumentParser()
p.add_argument("--run", action="append", required=True,
               help="비교할 런 디렉토리 이름. 두 번까지 준다 (첫째 실선, 둘째 파선)")
p.add_argument("--targets", nargs="+", default=["zh", "de", "ja"])
p.add_argument("--out", default=None)
a = p.parse_args()


def curve(cond: dict, pick) -> list[tuple[float, float]]:
    """(LAAL 어절, BLEU) 를 지연 오름차순으로."""
    return sorted((v["laal_words"], v["bleu"]) for n, v in cond.items() if pick(n))


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
    ax.axhline(unseg["bleu"], color="#8a8a8a", ls=":", lw=1.2, zorder=1)
    ax.annotate(f"unsegmented {unseg['bleu']:.1f}  (LAAL {unseg['laal_words']:.1f}w)",
                xy=(0.985, unseg["bleu"]), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="#5a5a5a")
    punct = cond0.get("punct")
    if punct:
        ax.axhline(punct["bleu"], color="#9a6b3f", ls=":", lw=1.2, zorder=1)
        ax.annotate(f"punct {punct['bleu']:.1f}  (LAAL {punct['laal_words']:.1f}w)",
                    xy=(0.985, punct["bleu"]), xycoords=("axes fraction", "data"),
                    ha="right", va="top", fontsize=8, color="#7d5632")
    ax.set_title(f"en→{t}", fontsize=12)
    ax.set_xlabel("LAAL (source words)")
    ax.grid(True, color="#e8e7e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_ylabel("BLEU")
axes[0].legend(fontsize=8.5, loc="upper left", framealpha=0.95)
fig.suptitle("CoVoST2 test 15,530 — quality vs latency  "
             "(BLEU tokenizers differ by target; read each panel on its own)",
             fontsize=10, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955))
out = Path(a.out) if a.out else A / runs[0][0] / "bleu" / "tradeoff_score_grid.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170)
print(out)
