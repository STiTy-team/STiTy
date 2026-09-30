"""Figures for aggregate.py (matplotlib, PNG).

Colour follows the entity and never changes between figures: models take the validated
categorical slots blue / orange / aqua / violet (all-pairs CVD check passes for these
four). Strategies only ever appear as colour inside one model's panel, so they take a
second set - green / yellow / magenta / violet - that passes the all-pairs checks as a set;
they always carry a distinct marker and line style too, so identity never rests on colour
alone. Cue-distance groups are ordered, so they take three steps of one blue ramp.
Heatmaps use the same blue ramp, light = low.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BASELINE_GRAY = "#8a8985"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
EXTRA_MODEL_COLORS = ["#e87ba4", "#008300", "#e34948", "#eda100"]
STRATEGY_STYLE = {
    "NONE": {"color": BASELINE_GRAY, "marker": "*", "ls": ":"},
    "SRC": {"color": "#008300", "marker": "o", "ls": "-"},
    "TGT": {"color": "#eda100", "marker": "s", "ls": "--"},
    "SRC_TGT": {"color": "#e87ba4", "marker": "^", "ls": "-."},
    "SPK_SRC_TGT": {"color": "#4a3aa7", "marker": "D", "ls": (0, (1, 1))},
}
STRATEGIES = ["SRC", "TGT", "SRC_TGT", "SPK_SRC_TGT"]
NS = [0, 1, 3, 5]
N_SIZE = {0: 70, 1: 30, 3: 65, 5: 120}
SEQ_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
            "#0d366b"]
LATENCY_TICKS = [200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 5000]
GPU_NOTE = "로컬 모델 지연은 다른 작업과 GPU 공유 중 측정 — 절대값 부정확"
MIN_CHECKS = 10

CUE_DISTANCE_GROUPS = {
    "cue 1 turn back": ["c01-t12", "c02-t10", "c02-t12", "c02-t15", "c03-t14", "c03-t17",
                        "c04-t08", "c04-t12", "c05-t12"],
    "cue 2-3 turns back": ["c01-t06", "c01-t08", "c01-t14", "c03-t05", "c04-t05", "c05-t14"],
    "cue 4-5 turns back": ["c02-t07", "c03-t12", "c04-t13", "c05-t09", "c05-t13"],
}
CUE_STYLE = {
    "cue 1 turn back": {"color": "#6da7ec", "marker": "o", "ls": "-"},
    "cue 2-3 turns back": {"color": "#256abf", "marker": "s", "ls": "--"},
    "cue 4-5 turns back": {"color": "#0d366b", "marker": "D", "ls": "-."},
}

METRIC_LABEL = {
    "judge_mean": "Context judge, mean of 6 dimensions (1-5)",
    "comet": "COMET (wmt22-comet-da)",
    "check_pass_rate": "Check pass rate",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 150,
        "font.family": ["DejaVu Sans", "Noto Sans CJK KR"],
        "axes.unicode_minus": False,
    })


def _header(fig, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title and muted subtitle above the plotting area."""
    fig.text(0.01, 1.0 + (0.075 if subtitle else 0.03), title, ha="left", va="bottom",
             fontsize=12, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1.01, subtitle, ha="left", va="bottom", fontsize=9, color=INK_2)


def _save(fig, path: Path) -> Path:
    extra = list(fig.legends) + list(fig.texts)
    for ax in fig.axes:
        extra += [c for c in ax.get_children() if isinstance(c, Legend)]
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15, bbox_extra_artists=extra)
    plt.close(fig)
    return path


def _plain_log_axis(axis, lo: float, hi: float) -> None:
    ticks = [t for t in LATENCY_TICKS if lo * 0.97 <= t <= hi * 1.03]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.set_minor_locator(NullLocator())


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Push label positions apart so no two sit closer than min_gap (keeps order)."""
    order = np.argsort(values)
    placed = list(np.asarray(values, dtype=float)[order])
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    shift = (np.mean(np.asarray(values)[order]) - np.mean(placed))
    placed = [p + shift for p in placed]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    out = [0.0] * len(values)
    for rank, idx in enumerate(order):
        out[idx] = placed[rank]
    return out


def model_colors(models: list[str]) -> dict[str, str]:
    palette = MODEL_COLORS + EXTRA_MODEL_COLORS
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def _cells(frame: pd.DataFrame) -> pd.DataFrame:
    """model x strategy x n summary over successful main/baseline rows."""
    main = frame[frame["phase"].isin(["main", "baseline"]) & ~frame["failed"]]
    records = []
    for (model, strategy, n), g in main.groupby(["model", "context_strategy", "context_n"]):
        passed, total = g["checks_passed"].sum(), g["checks_total"].sum()
        records.append({
            "model": model, "context_strategy": strategy, "context_n": int(n),
            "judge_mean": g["judge_mean"].mean(), "comet": g["comet"].mean(),
            "check_pass_rate": passed / total if total else np.nan,
            "latency_median_ms": g["latency_ms"].median(),
            "latency_p95_ms": float(np.percentile(g["latency_ms"].dropna(), 95))
            if g["latency_ms"].notna().any() else np.nan,
            "n_instances": g["instance_id"].nunique() if "instance_id" in g else len(g),
        })
    return pd.DataFrame(records)


def _quality_metric(cells: pd.DataFrame) -> str:
    return "judge_mean" if cells["judge_mean"].notna().any() else "comet"


def _n_instances(cells: pd.DataFrame) -> str:
    counts = sorted(set(int(v) for v in cells["n_instances"].dropna()))
    return "/".join(str(c) for c in counts) if counts else "?"


def scatter_quality_latency(cells: pd.DataFrame, colors: dict, out: Path,
                            metric: str | None = None, suffix: str = "") -> Path | None:
    metric = metric or _quality_metric(cells)
    data = cells.dropna(subset=[metric, "latency_median_ms"])
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for model in [m for m in colors if m in set(data["model"])]:
        part = data[data["model"] == model]
        color = colors[model]
        base = part[part["context_strategy"] == "NONE"]
        for strategy in STRATEGIES:
            s = part[part["context_strategy"] == strategy].sort_values("context_n")
            if s.empty:
                continue
            path_rows = pd.concat([base, s]) if len(base) else s
            ax.plot(path_rows["latency_median_ms"], path_rows[metric], color=color,
                    lw=0.9, alpha=0.55, ls=STRATEGY_STYLE[strategy]["ls"], zorder=2)
            style = STRATEGY_STYLE[strategy]
            ax.scatter(s["latency_median_ms"], s[metric],
                       s=[N_SIZE.get(int(n), 60) for n in s["context_n"]],
                       marker=style["marker"], color=color, edgecolors=SURFACE,
                       linewidths=1.0, zorder=3)
        if len(base):
            ax.scatter(base["latency_median_ms"], base[metric], s=260, marker="*",
                       color=color, edgecolors=INK, linewidths=0.8, zorder=4)
    lo, hi = data["latency_median_ms"].min(), data["latency_median_ms"].max()
    ax.set_xscale("log")
    ax.set_xlim(lo * 0.9, hi * 1.1)
    _plain_log_axis(ax.xaxis, lo * 0.9, hi * 1.1)
    ax.set_xlabel("Median latency per request (ms, log scale)")
    ax.set_ylabel(METRIC_LABEL[metric])
    present = set(data["model"])
    model_handles = [Line2D([], [], marker="o", ls="", color=c, markersize=8, label=m)
                     for m, c in colors.items() if m in present]
    strat_handles = [Line2D([], [], marker="*", ls="", color=INK_2, markersize=13,
                            markeredgecolor=INK, markeredgewidth=0.6,
                            label="NONE (n=0, path start)")]
    strat_handles += [Line2D([], [], marker=STRATEGY_STYLE[s]["marker"], ls=STRATEGY_STYLE[s]["ls"],
                             lw=0.9, color=INK_2, markersize=7, label=s)
                      for s in STRATEGIES if s in set(data["context_strategy"])]
    size_handles = [Line2D([], [], marker="o", ls="", color=INK_2,
                           markersize=np.sqrt(N_SIZE[n]), label=f"n={n}") for n in (1, 3, 5)]
    first = ax.legend(handles=model_handles, title="Model (colour)", loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), alignment="left")
    ax.add_artist(first)
    second = ax.legend(handles=strat_handles, title="Strategy (marker, line)",
                       loc="upper left", bbox_to_anchor=(1.01, 0.70), alignment="left")
    ax.add_artist(second)
    ax.legend(handles=size_handles, title="Context turns (size)", loc="upper left",
              bbox_to_anchor=(1.01, 0.30), alignment="left", labelspacing=1.0)
    fig.tight_layout()
    _header(fig, f"{METRIC_LABEL[metric]} vs latency, per model x strategy as n grows",
            "Each path starts at the NONE star and passes n = 1, 3, 5. " + GPU_NOTE)
    return _save(fig, out / f"quality_latency_scatter{suffix}.png")


def quality_vs_n(cells: pd.DataFrame, models: list[str], metric: str, out: Path) -> Path | None:
    if cells.empty or not cells[metric].notna().any():
        return None
    present = [m for m in models if m in set(cells["model"])]
    ncols = 2 if len(present) > 1 else 1
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols + 1.8, 3.4 * nrows),
                             sharey=True, sharex=True, squeeze=False)
    flat = axes.ravel()
    for ax, model in zip(flat, present):
        part = cells[cells["model"] == model]
        base = part[part["context_strategy"] == "NONE"][metric]
        base_value = base.iloc[0] if len(base) else np.nan
        if not np.isnan(base_value):
            ax.axhline(base_value, color=BASELINE_GRAY, lw=1, ls=":", zorder=1)
        for strategy in STRATEGIES:
            s = part[part["context_strategy"] == strategy].sort_values("context_n")
            if s.empty:
                continue
            xs = ([0] if not np.isnan(base_value) else []) + s["context_n"].tolist()
            ys = ([base_value] if not np.isnan(base_value) else []) + s[metric].tolist()
            style = STRATEGY_STYLE[strategy]
            ax.plot(xs, ys, color=style["color"], ls=style["ls"], marker=style["marker"],
                    markersize=6, markeredgecolor=SURFACE, label=strategy, zorder=3)
        ax.set_title(model, loc="left")
        ax.set_xticks(NS)
    for ax in flat[len(present):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Context turns n (0 = NONE)")
    for row in axes:
        row[0].set_ylabel(METRIC_LABEL[metric])
    handles = [Line2D([], [], color=STRATEGY_STYLE[s]["color"], ls=STRATEGY_STYLE[s]["ls"],
                      marker=STRATEGY_STYLE[s]["marker"], label=s) for s in STRATEGIES]
    handles.append(Line2D([], [], color=BASELINE_GRAY, ls=":", lw=1, label="NONE baseline"))
    fig.tight_layout()
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               title="Strategy")
    _header(fig, f"{METRIC_LABEL[metric]} vs context length, by model",
            f"Each point is the mean over {_n_instances(cells)} instances "
            "(check pass rate pools every check); n = 0 is the NONE baseline.")
    return _save(fig, out / f"quality_vs_n_{metric}.png")


def latency_vs_n(frame: pd.DataFrame, colors: dict, out: Path) -> Path | None:
    main = frame[frame["phase"].isin(["main", "baseline"]) & ~frame["failed"]]
    if main["latency_ms"].dropna().empty:
        return None
    stats_by_model = {}
    for model in colors:
        part = main[main["model"] == model]
        if part.empty or part["latency_ms"].dropna().empty:
            continue
        stats_by_model[model] = part.groupby("context_n")["latency_ms"].agg(
            median="median",
            p95=lambda v: np.percentile(v.dropna(), 95) if v.notna().any() else np.nan,
        ).reindex(NS).dropna()
    all_vals = pd.concat([s[["median", "p95"]] for s in stats_by_model.values()]).stack()
    lo, hi = all_vals.min() * 0.85, all_vals.max() * 1.15
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, column, title in ((axes[0], "median", "Median"), (axes[1], "p95", "95th percentile")):
        ends, names = [], []
        for model, stats in stats_by_model.items():
            ax.plot(stats.index, stats[column], color=colors[model], marker="o", markersize=6,
                    markeredgecolor=SURFACE)
            ends.append(stats[column].iloc[-1])
            names.append(model)
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        _plain_log_axis(ax.yaxis, lo, hi)
        positions = np.exp(_spread(list(np.log(ends)), 0.075)) if ends else []
        for model, y_end, y_label in zip(names, ends, positions):
            ax.annotate(model, (NS[-1], y_end), xytext=(NS[-1] + 0.35, y_label),
                        textcoords="data", fontsize=8.5, color=INK_2, va="center",
                        arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8})
        ax.set_xlim(-0.3, 7.6)
        ax.set_xticks(NS)
        ax.set_xlabel("Context turns n (0 = NONE; strategies pooled)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Latency per request (ms, log scale)")
    fig.tight_layout()
    _header(fig, "Latency vs context length", GPU_NOTE)
    return _save(fig, out / "latency_vs_n.png")


def tag_heatmaps(frame: pd.DataFrame, models: list[str], out: Path) -> list[Path]:
    from aggregate import VIEWS, tag_pass

    long = tag_pass(frame[frame["phase"].isin(["main", "baseline"])])
    if long.empty:
        return []
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    paths = []
    for model in [m for m in models if m in set(long["model"])] + ["ALL"]:
        part = long[long["model"] == model]
        rates = part.pivot(index="tag", columns="view", values="pass_rate")
        counts = part.pivot(index="tag", columns="view", values="n_checks")
        views = [v for v in VIEWS if v in rates.columns]
        rates, counts = rates.reindex(columns=views), counts.reindex(columns=views)
        fig, ax = plt.subplots(figsize=(1.0 * len(views) + 3.2, 0.5 * len(rates) + 2.2))
        image = ax.imshow(rates.values.astype(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(views)), views, rotation=30, ha="right")
        ax.set_yticks(range(len(rates.index)), rates.index)
        n_views = [v for v in views if v.startswith("n=")]
        s_views = [v for v in views if v in STRATEGIES]
        has_none = "NONE" in views
        first_n = 1 if has_none else 0
        if has_none:
            ax.axvline(0.5, color=SURFACE, lw=3)
        ax.axvline(first_n - 0.5 + len(n_views), color=SURFACE, lw=3)
        groups = []
        if has_none:
            groups.append((0, 0, "baseline"))
        if n_views:
            groups.append((first_n, first_n + len(n_views) - 1, "by n\n(strategies pooled)"))
        if s_views:
            start = first_n + len(n_views)
            groups.append((start, start + len(s_views) - 1, "by strategy\n(n pooled)"))
        for start, end, label in groups:
            ax.annotate("", xy=(start - 0.42, 1.02), xytext=(end + 0.42, 1.02),
                        xycoords=("data", "axes fraction"),
                        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 1})
            ax.text((start + end) / 2, 1.04, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color=INK_2)
        n_low = 0
        for i in range(rates.shape[0]):
            for j in range(rates.shape[1]):
                value = rates.values[i, j]
                if np.isnan(value):
                    continue
                count = int(counts.values[i, j])
                low = count < MIN_CHECKS
                if low:
                    n_low += 1
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=SURFACE,
                                           alpha=0.6, edgecolor=BASELINE_GRAY, hatch="////",
                                           lw=0, zorder=2))
                ax.text(j, i, f"{value:.0%}\n({count})", ha="center", va="center",
                        fontsize=7, zorder=3,
                        color=INK_2 if low else (INK if value < 0.55 else "#ffffff"))
        colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
        colorbar.set_label("pass rate (checks in brackets)", color=INK_2)
        note = (f"Hatched, faded cells rest on fewer than {MIN_CHECKS} checks; read them "
                "as indicative only." if n_low else None)
        fig.tight_layout()
        _header(fig, f"Check pass rate by tag - {'all models' if model == 'ALL' else model}",
                note)
        safe = model.replace("/", "_")
        paths.append(_save(fig, out / f"tag_pass_heatmap_{safe}.png"))
    return paths


def _cue_rates(rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled check pass rate per cue group x n (same pooling as aggregate.py)."""
    records = []
    for group, ids in CUE_DISTANCE_GROUPS.items():
        part = rows[rows["instance_id"].isin(ids)]
        for n, g in part.groupby("context_n"):
            total = g["checks_total"].sum()
            records.append({"group": group, "context_n": int(n), "n_checks": int(total),
                            "rate": g["checks_passed"].sum() / total if total else np.nan})
    return pd.DataFrame(records)


def _draw_cue(ax, rates: pd.DataFrame, end_labels: bool) -> None:
    for group, style in CUE_STYLE.items():
        s = rates[rates["group"] == group].set_index("context_n").reindex(NS).dropna()
        if s.empty:
            continue
        ax.plot(s.index, s["rate"], color=style["color"], ls=style["ls"],
                marker=style["marker"], markersize=6, markeredgecolor=SURFACE, zorder=3)
    if end_labels:
        ends = []
        for group in CUE_STYLE:
            s = rates[rates["group"] == group].set_index("context_n").reindex(NS).dropna()
            if len(s):
                ends.append((group, s.index[-1], s["rate"].iloc[-1]))
        positions = _spread([e[2] for e in ends], 0.05)
        for (group, x, y), y_label in zip(ends, positions):
            ax.annotate(group.replace("cue ", ""), (x, y), xytext=(x + 0.3, y_label),
                        textcoords="data", fontsize=8.5, color=INK_2, va="center",
                        arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8})
    ax.set_ylim(0, 1)
    ax.set_xticks(NS)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))


def cue_distance_vs_n(frame: pd.DataFrame, models: list[str], out: Path) -> list[Path]:
    rows = frame[frame["phase"].isin(["main", "baseline"]) & ~frame["failed"]]
    if rows.empty or "instance_id" not in rows:
        return []
    handles = [Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"],
                      label=f"{g} ({len(CUE_DISTANCE_GROUPS[g])} instances)")
               for g, s in CUE_STYLE.items()]
    pooled = _cue_rates(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _draw_cue(ax, pooled, end_labels=True)
    ax.set_xlim(-0.3, 6.6)
    ax.set_xlabel("Context turns n (0 = NONE)")
    ax.set_ylabel("Check pass rate")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              fontsize=8.5)
    counts = pooled.groupby("context_n")["n_checks"].sum()
    fig.tight_layout()
    _header(fig, "Check pass rate by how far back the needed cue is",
            f"All models and strategies pooled. Checks per point: "
            f"{int(pooled[pooled.context_n == 0].n_checks.min())}-"
            f"{int(pooled[pooled.context_n == 0].n_checks.max())} at n = 0, "
            f"{int(pooled[pooled.context_n > 0].n_checks.min())}-"
            f"{int(pooled[pooled.context_n > 0].n_checks.max())} at n > 0 "
            f"({int(counts.sum())} in total).")
    paths = [_save(fig, out / "cue_distance_vs_n.png")]

    present = [m for m in models if m in set(rows["model"])]
    ncols = 2 if len(present) > 1 else 1
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols + 1.0, 3.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    per_model = {m: _cue_rates(rows[rows["model"] == m]) for m in present}
    base_counts = pd.concat([r[r.context_n == 0].n_checks for r in per_model.values()])
    for ax, model in zip(flat, present):
        _draw_cue(ax, per_model[model], end_labels=False)
        ax.set_title(model, loc="left")
        ax.set_xlim(-0.3, 5.3)
    for ax in flat[len(present):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Context turns n (0 = NONE)")
    for row in axes:
        row[0].set_ylabel("Check pass rate")
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=8.5)
    _header(fig, "Check pass rate by cue distance, per model",
            "Strategies pooled; each model sees the same instances. "
            f"n = 0 points rest on only {int(base_counts.min())}-{int(base_counts.max())} "
            "checks per group.")
    paths.append(_save(fig, out / "cue_distance_vs_n_by_model.png"))
    return paths


def draw_all(frame: pd.DataFrame, out: Path, models: list[str]) -> list[Path]:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    colors = model_colors(models)
    cells = _cells(frame)
    paths = [scatter_quality_latency(cells, colors, out),
             scatter_quality_latency(cells, colors, out, metric="check_pass_rate",
                                     suffix="_check")]
    for metric in ("judge_mean", "comet", "check_pass_rate"):
        paths.append(quality_vs_n(cells, models, metric, out))
    paths.append(latency_vs_n(frame, colors, out))
    paths += tag_heatmaps(frame, models, out)
    paths += cue_distance_vs_n(frame, models, out)
    return [p for p in paths if p is not None]
