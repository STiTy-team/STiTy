"""`qwen_seg_sweep.py` 산출물 → 지연 대비 H_set 그림(PNG)과 표(md).

    python -m core.meaning_segmentator.autoseg.gates.plot_qwen_sweep <sweep_*.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# (키, 라벨, 색, 선, 스트리밍인가) — 색은 기본 팔레트 순서 그대로, 오라클은 기준선이라 중립색
SERIES = [
    ("qwen_stream_feed", "Qwen <SEG> stream, θ (feed <SEG>)", "#2a78d6", "-", True),
    ("qwen_stream", "Qwen <SEG> stream, θ (no feed)", "#eb6834", "-", True),
    ("fixed_N", "every N words", "#1baf7a", "-", True),
    ("punct_stream", "after punctuation", "#eda100", "-", True),
    ("judge13", "judge13 prompt, top-k", "#e87ba4", "--", False),
    ("punct_k", "punctuation, top-k", "#008300", "--", False),
    ("oracle", "oracle, top-k", INK2, ":", False),
]


def main() -> int:
    path = Path(sys.argv[1])
    curves = json.loads(path.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 11, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=160)
    ax.grid(True, color=GRID, linewidth=1.0, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for key, label, color, ls, streaming in SERIES:
        pts = curves.get(key)
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: p["chunk"])
        xs, ys = [p["chunk"] for p in pts], [p["h"] for p in pts]
        ax.plot(xs, ys, ls=ls, marker="o" if streaming else "s", color=color, lw=2, ms=8,
                mec=SURFACE, mew=2, label=label + ("" if streaming else "  [needs full sentence]"),
                zorder=4 if streaming else 3)
        ax.annotate(label.split(",")[0], (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    for p in curves.get("qwen_stream_feed", []):
        ax.annotate(f"θ={p['knob']}", (p["chunk"], p["h"]), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("average chunk length (words)  →  more latency")
    ax.set_ylabel("H_set  (CometKiwi × (1 − max contra)), mean of 200 sentences")
    ax.set_title("Latency vs quality — run27 test (FLEURS en→zh/ja/de/es), 200 sentences",
                 loc="left", fontsize=12, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_xlim(left=1.5)
    fig.tight_layout()
    png = path.with_suffix(".png")
    fig.savefig(png)

    lines = ["| 정책 | 노브 | 평균 조각(어절) | H_set | 절단 수 |", "|---|---|---|---|---|"]
    for key, *_ in SERIES:
        for p in sorted(curves.get(key, []), key=lambda p: p["chunk"]):
            lines.append(f"| {key} | {p['knob']} | {p['chunk']:.2f} | {p['h']:.4f} | {p['cuts']} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
