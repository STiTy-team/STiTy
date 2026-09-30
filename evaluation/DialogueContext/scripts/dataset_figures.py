"""평가 데이터셋 구성 그림: 태그별 인스턴스 수, 단서 거리 분포, 인스턴스 × 태그 지도."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures as F  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
TAGS = ["gender_reference", "omitted_argument", "pronoun_coreference", "context_trap",
        "discourse_connective", "lexical_consistency", "word_sense", "fragment_incremental",
        "register_politeness", "entity_consistency"]
CUE_LABEL = {1: "1 turn back", 3: "2-3 turns back", 5: "4-5 turns back"}
CUE_KEY = {k: f"cue {v}" for k, v in CUE_LABEL.items()}


def cue_group(instance_id: str) -> int:
    for n, key in CUE_KEY.items():
        if instance_id in F.CUE_DISTANCE_GROUPS[key]:
            return n
    raise KeyError(instance_id)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=str(HERE / "data/instances.jsonl"))
    ap.add_argument("--out-dir", default=str(HERE / "results/dctx-20260925/figures"))
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(args.instances, encoding="utf-8")]
    out = Path(args.out_dir)
    F._style()

    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 1.15],
                            hspace=0.55, wspace=0.35)
    ax_tag = fig.add_subplot(grid[0, 0])
    ax_cue = fig.add_subplot(grid[0, 1])
    ax_map = fig.add_subplot(grid[1, :])

    tag_counts = Counter(tag for row in rows for tag in row["challenge_tags"])
    check_counts = Counter(check["tag"] for row in rows for check in row["checks"])
    order = sorted(TAGS, key=lambda t: tag_counts[t])
    ax_tag.barh(order, [tag_counts[t] for t in order], color=F.MODEL_COLORS[0], height=0.62)
    for y, tag in enumerate(order):
        ax_tag.text(tag_counts[tag] + 0.1, y, f"{tag_counts[tag]} inst / {check_counts[tag]} checks",
                    va="center", fontsize=8.5, color=F.INK)
    ax_tag.set_xlim(0, 7.5)
    ax_tag.set_xlabel("instances")
    ax_tag.set_title("Challenge tags (an instance can carry several)", loc="left")
    ax_tag.grid(axis="y", visible=False)

    cue_counts = Counter(cue_group(row["instance_id"]) for row in rows)
    keys = [1, 3, 5]
    bars = ax_cue.bar([CUE_LABEL[k] for k in keys], [cue_counts[k] for k in keys],
                      color=[F.CUE_STYLE[CUE_KEY[k]]["color"] for k in keys], width=0.6)
    for bar, k in zip(bars, keys):
        ax_cue.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, str(cue_counts[k]),
                    ha="center", fontsize=10, color=F.INK)
    ax_cue.set_ylim(0, max(cue_counts.values()) + 2)
    ax_cue.set_ylabel("instances")
    ax_cue.set_title("Where the needed cue sits (smallest n that reaches it)", loc="left")
    ax_cue.grid(axis="x", visible=False)

    ids = [row["instance_id"] for row in rows]
    for x, row in enumerate(rows):
        for tag in row["challenge_tags"]:
            y = TAGS.index(tag)
            ax_map.scatter(x, y, s=70, marker="s",
                           color=F.CUE_STYLE[CUE_KEY[cue_group(row["instance_id"])]]["color"], zorder=3)
    ax_map.set_xticks(range(len(ids)), ids, rotation=60, ha="right", fontsize=8.5)
    ax_map.set_yticks(range(len(TAGS)), TAGS, fontsize=8.5)
    ax_map.set_xlim(-0.6, len(ids) - 0.4)
    ax_map.set_ylim(len(TAGS) - 0.5, -0.5)
    for boundary in range(4, len(ids), 4):
        ax_map.axvline(boundary - 0.5, color=F.GRID, lw=1.2)
    handles = [plt.Line2D([], [], marker="s", ls="", color=F.CUE_STYLE[CUE_KEY[k]]["color"], markersize=8,
                          label=f"cue {CUE_LABEL[k]}") for k in keys]
    ax_map.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
                  frameon=False)
    ax_map.set_title("Instance × tag map (colour = cue distance; vertical lines separate dialogues c01-c05)",
                     loc="left")

    F._header(fig, "DialogueContext evaluation set: 5 dialogues, 20 target turns, 36 checks",
              "Cues were planted deliberately 1, 2-3 or 4-5 turns before each target turn.")
    path = F._save(fig, out / "dataset_overview.png")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
