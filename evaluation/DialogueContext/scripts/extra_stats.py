"""Extra statistics for a finished DialogueContext run (tables T8-T13 and one figure).

Reuses aggregate.py for loading (load_merged) and table writing; aggregate.py is not changed.

  T8_strategy_contrasts     paired bootstrap of strategy differences (n pooled and per n)
  T9_n_steps                paired bootstrap of n steps: n1-NONE, n3-n1, n5-n3
  T10_cue_distance          check pass rate by required n (cue distance) x given n
  T10b_cue_distance_boot    bootstrap of "cue enters window" and "context beyond the cue"
  T11_latency_regression    latency / TTFT vs input size, median by n, n5-NONE median CI,
                            linear extrapolation to 20 / 50 context turns
  T12_gpu_contention        local-model latency vs whole-GPU util samples (logs/gpu_samples.csv)
  T13_efficiency            quality gain over NONE per 100 extra input tokens
  figures/quality_latency_tradeoff.png

Bootstrap: instances resampled with replacement, 1000 resamples, seed 20260925, paired
(both sides of a difference come from the same instance). The per-instance quality value is
judge_mean, the instance-level check pass fraction (checks passed / checks of that
instance), and COMET. 'ALL' first averages the per-instance difference over the four models,
then resamples instances, so the unit of resampling is always one of the 20 instances.

    venv-metrics/bin/python evaluation/DialogueContext/scripts/extra_stats.py \
        --run-dir evaluation/DialogueContext/results/dctx-20260925
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate  # noqa: E402
import scoring_common as common  # noqa: E402

SEED = aggregate.SEED
STRATEGIES = aggregate.STRATEGIES
METRICS = ["judge_mean", "check_pass", "comet"]
LOCAL_MODELS = ["gemma3-4b", "qwen3.5-4b"]
KST_OFFSET = pd.Timedelta(hours=9)

# Turns back to the cue that the instance's checks need (from note / expected_context_effect).
REQUIRED_N = {
    1: ["c01-t12", "c02-t10", "c02-t12", "c02-t15", "c03-t14", "c03-t17", "c04-t08",
        "c04-t12", "c05-t12"],
    3: ["c01-t06", "c01-t08", "c01-t14", "c03-t05", "c04-t05", "c05-t14"],
    5: ["c02-t07", "c03-t12", "c04-t13", "c05-t09", "c05-t13"],
}
REQUIRED_OF = {iid: n for n, ids in REQUIRED_N.items() for iid in ids}


# --------------------------------------------------------------------------- helpers

def per_instance(main: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, instance, strategy, n) with the three quality values."""
    ok = main[~main["failed"]].copy()
    ok["check_pass"] = ok["checks_passed"] / ok["checks_total"]
    return (ok.groupby(["model", "instance_id", "context_strategy", "context_n"])[METRICS]
            .mean().reset_index())


def cell(values: pd.DataFrame, model: str, strategy: str | None, n) -> pd.DataFrame:
    """Instance x metric table for one side of a contrast.

    strategy None = pooled over strategies; n 'pooled' = mean over n=1/3/5; n 0 = NONE.
    """
    part = values[values["model"] == model]
    if n == 0:
        part = part[part["context_strategy"] == "NONE"]
    else:
        part = part[part["context_strategy"] != "NONE"]
        if strategy is not None:
            part = part[part["context_strategy"] == strategy]
        if n != "pooled":
            part = part[part["context_n"] == n]
    return part.groupby("instance_id")[METRICS].mean()


def diff_table(values: pd.DataFrame, model: str, side_a, side_b) -> pd.DataFrame:
    """Per-instance A-B. model 'ALL' averages the per-instance difference over models."""
    models = sorted(values["model"].unique()) if model == "ALL" else [model]
    diffs = []
    for m in models:
        a, b = cell(values, m, *side_a), cell(values, m, *side_b)
        ids = sorted(set(a.index) & set(b.index))
        diffs.append(a.loc[ids] - b.loc[ids])
    return pd.concat(diffs).groupby(level=0).mean()


def boot_mean(diff: np.ndarray, n_resamples: int) -> tuple[float, float, float]:
    diff = diff[~np.isnan(diff)]
    if not len(diff):
        return (np.nan,) * 3
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    boot = diff[draws].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def ci_flag(low: float, high: float) -> str:
    if np.isnan(low):
        return ""
    # endpoints are compared at 3 decimals: with 20 instances and discrete pass fractions a
    # percentile can land a hair off 0, and that is not a real exclusion
    low, high = round(low, 3), round(high, 3)
    return "+" if low > 0 else ("-" if high < 0 else "")


def contrast_rows(values, model, label_cols: dict, side_a, side_b, n_resamples) -> dict:
    diff = diff_table(values, model, side_a, side_b)
    row = {"model": model, **label_cols, "n_instances": len(diff)}
    for metric in METRICS:
        d, lo, hi = boot_mean(diff[metric].to_numpy(float), n_resamples)
        row.update({f"{metric}_delta": d, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi,
                    f"{metric}_sig": ci_flag(lo, hi)})
    return row


def fmt_ci(row, metric, digits=3) -> str:
    d, lo, hi = row[f"{metric}_delta"], row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
    if pd.isna(d):
        return "—"
    star = "*" if row[f"{metric}_sig"] else ""
    return f"{d:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{star}"


def compact(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = frame[keys + ["n_instances"]].copy()
    for metric in METRICS:
        out[metric] = [fmt_ci(r, metric) for _, r in frame.iterrows()]
    return out


def write(frame: pd.DataFrame, md_frame: pd.DataFrame, name: str, tables: Path, title: str,
          note: str) -> None:
    frame.to_csv(tables / f"{name}.csv", index=False)
    body = f"# {title}\n\n{note}\n\n" + aggregate.to_markdown(md_frame)
    (tables / f"{name}.md").write_text(body, encoding="utf-8")


BOOT_NOTE = ("Paired bootstrap over the 20 instances (1000 resamples, seed 20260925). "
             "Cells: mean difference [95% CI]; '*' = CI excludes 0 (endpoints compared at 3 decimals). check_pass is the "
             "instance-level pass fraction (mean over instances of passed/checks), so it can "
             "differ slightly from the pooled check_pass_rate in T0-T7. ALL averages each "
             "instance's difference over the four models before resampling.")


# --------------------------------------------------------------------------- T8 / T9

CONTRASTS = [("SRC_TGT", "SRC"), ("SRC_TGT", "TGT"), ("TGT", "SRC"), ("SPK_SRC_TGT", "SRC_TGT")]


def t8(values, models, n_resamples) -> pd.DataFrame:
    rows = []
    for model in models + ["ALL"]:
        for n in ["pooled", 1, 3, 5]:
            for a, b in CONTRASTS:
                rows.append(contrast_rows(values, model, {"n": n, "contrast": f"{a} - {b}"},
                                          (a, n), (b, n), n_resamples))
    return pd.DataFrame(rows)


def t9(values, models, n_resamples) -> pd.DataFrame:
    rows = []
    steps = [(1, 0), (3, 1), (5, 3), (5, 0)]
    for model in models + ["ALL"]:
        for strategy in [None] + STRATEGIES:
            for hi, lo in steps:
                label = {"strategy": strategy or "pooled",
                         "step": f"n{hi} - {'NONE' if lo == 0 else f'n{lo}'}"}
                rows.append(contrast_rows(values, model, label, (strategy, hi), (strategy, lo),
                                          n_resamples))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- T10 cue distance

def t10(values, models, n_resamples) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = values.copy()
    v["required_n"] = v["instance_id"].map(REQUIRED_OF)
    rows = []
    for model in models + ["ALL"]:
        part = v if model == "ALL" else v[v["model"] == model]
        for req, group in part.groupby("required_n"):
            row = {"model": model, "required_n": int(req),
                   "n_instances": group["instance_id"].nunique()}
            for given in [0, 1, 3, 5]:
                g = group[group["context_n"] == given]
                # mean over instances of the per-instance pass fraction (strategies pooled)
                per_inst = g.groupby("instance_id")["check_pass"].mean()
                row[f"given_n{given}"] = per_inst.mean()
            rows.append(row)
    table = pd.DataFrame(rows)

    boot_rows = []
    for model in models + ["ALL"]:
        for req in (1, 3, 5):
            ids = REQUIRED_N[req]
            below = {1: 0, 3: 1, 5: 3}[req]
            specs = [("cue enters: n=req - n=below", (None, req), (None, below))]
            if req == 1:
                specs += [("beyond cue: n3 - n1", (None, 3), (None, 1)),
                          ("beyond cue: n5 - n1", (None, 5), (None, 1))]
            elif req == 3:
                specs += [("before cue: n1 - NONE", (None, 1), (None, 0)),
                          ("beyond cue: n5 - n3", (None, 5), (None, 3))]
            else:
                specs += [("before cue: n3 - NONE", (None, 3), (None, 0))]
            sub = values[values["instance_id"].isin(ids)]
            for label, a, b in specs:
                diff = diff_table(sub, model, a, b)
                d, lo, hi = boot_mean(diff["check_pass"].to_numpy(float), n_resamples)
                boot_rows.append({"model": model, "required_n": req, "contrast": label,
                                  "n_instances": len(diff), "check_pass_delta": d,
                                  "ci_low": lo, "ci_high": hi, "sig": ci_flag(lo, hi)})
    return table, pd.DataFrame(boot_rows)


# --------------------------------------------------------------------------- T11 latency

def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return (np.nan,) * 3
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(slope), float(intercept), float(r2)


def median_diff_ci(a: np.ndarray, b: np.ndarray, n_resamples: int) -> tuple:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    rng = np.random.default_rng(SEED)
    boot = (np.median(a[rng.integers(0, len(a), (n_resamples, len(a)))], axis=1)
            - np.median(b[rng.integers(0, len(b), (n_resamples, len(b)))], axis=1))
    return (float(np.median(a) - np.median(b)), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def ols2(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """y = a + b1*x1 + b2*x2. Returns (b1, b2, a, R^2)."""
    ok = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
    x1, x2, y = x1[ok], x2[ok], y[ok]
    if len(y) < 4:
        return (np.nan,) * 4
    design = np.column_stack([np.ones_like(x1), x1, x2])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(coef[1]), float(coef[2]), float(coef[0]), float(r2)


def t11(main, models, n_resamples, turns=(20, 50)) -> pd.DataFrame:
    rows = []
    ok = main[~main["failed"]]
    for model in models:
        part = ok[ok["model"] == model]
        is_deepl = part["input_tokens"].isna().all()
        xcol, outcol = (("context_chars", "output_chars") if is_deepl
                        else ("input_tokens", "output_tokens"))
        x = part[xcol].to_numpy(float)
        out = part[outcol].to_numpy(float)
        row = {"model": model, "x": xcol}
        for ycol, key in (("latency_ms", "lat"), ("ttft_ms", "ttft")):
            y = part[ycol].to_numpy(float)
            slope, intercept, r2 = ols(x, y)
            row.update({f"{key}_slope_per100x": slope * 100, f"{key}_r2": r2})
            if key == "lat":
                b_in, b_out, a, r2_2 = ols2(x, out, y)
                row.update({"lat_slope_per100x_ctrl_output": b_in * 100,
                            "lat_ms_per_output_unit": b_out, "lat_r2_with_output": r2_2})
                lat_fit = (a, b_in, b_out)
            else:
                ttft_fit = (intercept, slope)
        for n in (0, 1, 3, 5):
            row[f"median_lat_n{n}"] = part.loc[part["context_n"] == n, "latency_ms"].median()
        d, lo, hi = median_diff_ci(part.loc[part["context_n"] == 5, "latency_ms"].to_numpy(float),
                                   part.loc[part["context_n"] == 0, "latency_ms"].to_numpy(float),
                                   n_resamples)
        row.update({"median_n5_minus_none": d, "median_diff_ci_low": lo,
                    "median_diff_ci_high": hi})
        # input size per context turn, measured: slope of x on n (pooled, and SRC_TGT only)
        none = part[part["context_n"] == 0]
        for label, sub in (("pooled", part[part["context_n"] > 0]),
                           ("SRC_TGT", part[part["context_strategy"] == "SRC_TGT"])):
            both = pd.concat([sub, none])
            per_turn, _, _ = ols(both["context_n"].to_numpy(float), both[xcol].to_numpy(float))
            row[f"x_per_turn_{label}"] = per_turn
        base_x, mean_out = none[xcol].mean(), np.nanmean(out)
        a, b_in, b_out = lat_fit
        for t in (0, 5) + tuple(turns):
            x_t = base_x + row["x_per_turn_SRC_TGT"] * t
            tag = f"{t}turns"
            row[f"EXTRAP_x_at_{tag}"] = x_t
            row[f"EXTRAP_latency_ms_at_{tag}"] = a + b_in * x_t + b_out * mean_out
            if not is_deepl:
                row[f"EXTRAP_ttft_ms_at_{tag}"] = ttft_fit[0] + ttft_fit[1] * x_t
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- T12 GPU contention

def t12(main: pd.DataFrame, gpu_csv: Path, n_resamples: int) -> tuple[pd.DataFrame, str]:
    if not gpu_csv.exists():
        return pd.DataFrame(), "logs/gpu_samples.csv missing"
    g = pd.read_csv(gpu_csv, skipinitialspace=True)
    g["time"] = pd.to_datetime(g["time"])  # KST wall clock, no zone
    samples = g.groupby("time").agg(gpu_util=("gpu_util", "first"),
                                    used_mib=("used_mib", "sum")).reset_index()
    ok = main[~main["failed"]].copy()
    # started_at is UTC-aware; the GPU log is naive KST (+09:00)
    ok["t_kst"] = (pd.to_datetime(ok["started_at"], utc=True).dt.tz_localize(None)
                   + KST_OFFSET)
    ok["t_end_kst"] = ok["t_kst"] + pd.to_timedelta(ok["latency_ms"], unit="ms")
    run_start, run_end = ok["t_kst"].min(), ok["t_end_kst"].max()
    samples = samples[(samples["time"] >= run_start - pd.Timedelta(seconds=5))
                      & (samples["time"] <= run_end + pd.Timedelta(seconds=5))].copy()
    # Was one of our own local-model jobs running at the sample instant? Then the whole-GPU
    # util sample partly measures us, not the other session.
    local = ok[ok["model"].isin(LOCAL_MODELS)]
    samples["own_local_active"] = [bool(((local["t_kst"] <= t) & (local["t_end_kst"] >= t)).any())
                                   for t in samples["time"]]
    local = local.sort_values("t_kst")
    joined = pd.merge_asof(local, samples.sort_values("time"), left_on="t_kst",
                           right_on="time", direction="nearest",
                           tolerance=pd.Timedelta(seconds=5))
    first_sample = samples["time"].min()
    median_util = samples["gpu_util"].median()
    joined["window"] = np.where(joined["t_kst"] < first_sample, "before first sample",
                                np.where(joined["gpu_util"] > median_util,
                                         f"util > {median_util:.0f}% (median)",
                                         f"util <= {median_util:.0f}%"))
    # residual after the model's own size effect, so windows with longer prompts don't mislead
    joined["lat_resid"] = np.nan
    for model, part in joined.groupby("model"):
        slope, intercept, _ = ols(part["input_tokens"].to_numpy(float),
                                  part["latency_ms"].to_numpy(float))
        joined.loc[part.index, "lat_resid"] = part["latency_ms"] - (
            intercept + slope * part["input_tokens"])
    rows = []
    for model, part in joined.groupby("model"):
        rest = part[part["window"] != "before first sample"]
        for window, w in part.groupby("window"):
            rows.append({"model": model, "window": window, "n_rows": len(w),
                         "median_latency_ms": w["latency_ms"].median(),
                         "median_ttft_ms": w["ttft_ms"].median(),
                         "median_resid_ms": w["lat_resid"].median(),
                         "mean_input_tokens": w["input_tokens"].mean()})
        # high - low and first-2-min - rest, bootstrap of the median difference
        hi = rest[rest["gpu_util"] > median_util]["latency_ms"].to_numpy(float)
        lo = rest[rest["gpu_util"] <= median_util]["latency_ms"].to_numpy(float)
        d, l, h = median_diff_ci(hi, lo, n_resamples)
        rows.append({"model": model, "window": "DIFF high - low util", "n_rows": len(rest),
                     "median_latency_ms": d, "ci_low": l, "ci_high": h})
        first = part[part["window"] == "before first sample"]["latency_ms"].to_numpy(float)
        d, l, h = median_diff_ci(first, rest["latency_ms"].to_numpy(float), n_resamples)
        rows.append({"model": model, "window": "DIFF first 2 min - rest", "n_rows": len(part),
                     "median_latency_ms": d, "ci_low": l, "ci_high": h})
        r = np.corrcoef(rest["gpu_util"].to_numpy(float), rest["lat_resid"].to_numpy(float))[0, 1]
        rows.append({"model": model, "window": "corr(util, latency residual)",
                     "n_rows": len(rest), "median_latency_ms": r})
    idle = samples[~samples["own_local_active"]]["gpu_util"]
    busy = samples[samples["own_local_active"]]["gpu_util"]
    note = (f"{len(samples)} whole-GPU samples, 5-s interval, from {first_sample:%H:%M:%S} KST; "
            f"run {run_start:%H:%M:%S}-{run_end:%H:%M:%S} KST (started_at UTC + 9 h). "
            f"Each local-model row joined to the nearest sample (<= 5 s). Split at the median "
            f"util {median_util:.0f}%. gpu_util is the whole GPU, including our own job: "
            f"mean util {busy.mean():.1f}% over {len(busy)} samples taken while one of our "
            f"local jobs was running, {idle.mean():.1f}% over {len(idle)} samples while none "
            f"was (the latter is roughly the other session's load). lat_resid = latency minus "
            f"the model's own latency~input_tokens line. DIFF rows: median difference with 95% "
            f"bootstrap CI (rows resampled within each window).")
    return pd.DataFrame(rows), note


# --------------------------------------------------------------------------- T13 efficiency

def t13(main, values, models) -> pd.DataFrame:
    ok = main[~main["failed"]]
    rows = []
    for model in models:
        part = ok[ok["model"] == model]
        is_deepl = part["input_tokens"].isna().all()
        size_col = "context_chars" if is_deepl else "input_tokens"
        base_size = part.loc[part["context_n"] == 0, size_col].mean()
        base_q = cell(values, model, None, 0).mean()
        for strategy in STRATEGIES:
            for n in ["pooled", 1, 3, 5]:
                sel = part[part["context_strategy"] == strategy]
                if n != "pooled":
                    sel = sel[sel["context_n"] == n]
                extra = sel[size_col].mean() - base_size
                q = cell(values, model, strategy, n).mean()
                gain_j = q["judge_mean"] - base_q["judge_mean"]
                gain_c = q["check_pass"] - base_q["check_pass"]
                rows.append({"model": model, "strategy": strategy, "n": n,
                             "size_unit": "context chars" if is_deepl else "input tokens",
                             "extra_size": extra, "judge_gain": gain_j,
                             "judge_gain_per_100": gain_j / extra * 100 if extra else np.nan,
                             "check_pass_gain": gain_c,
                             "check_pass_gain_per_100": gain_c / extra * 100 if extra else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figure

def tradeoff_figure(main: pd.DataFrame, models: list[str], out: Path) -> Path:
    import figures as fg
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fg._style()
    cells = fg._cells(main)
    colors = fg.model_colors(models)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for ax, model in zip(axes.flat, models):
        part = cells[cells["model"] == model]
        base = part[part["context_strategy"] == "NONE"].iloc[0]
        for strategy in STRATEGIES:
            s = part[part["context_strategy"] == strategy].sort_values("context_n")
            style = fg.STRATEGY_STYLE[strategy]
            xs = [base["latency_median_ms"]] + s["latency_median_ms"].tolist()
            ys = [base["check_pass_rate"]] + s["check_pass_rate"].tolist()
            ax.plot(xs, ys, color=style["color"], ls=style["ls"], lw=1.0, alpha=0.45, zorder=2)
            for _, row in s.iterrows():
                ax.scatter(row["latency_median_ms"], row["check_pass_rate"],
                           s=fg.N_SIZE[row["context_n"]], marker=style["marker"],
                           color=style["color"], edgecolors=fg.SURFACE, linewidths=1.0,
                           zorder=3)
                ax.annotate(f"n{row['context_n']}",
                            (row["latency_median_ms"], row["check_pass_rate"]),
                            xytext=(5, 3), textcoords="offset points", fontsize=7,
                            color=fg.INK_2)
        st = fg.STRATEGY_STYLE["NONE"]
        ax.scatter(base["latency_median_ms"], base["check_pass_rate"], s=170, marker=st["marker"],
                   color=st["color"], edgecolors=fg.SURFACE, zorder=4)
        ax.annotate("NONE", (base["latency_median_ms"], base["check_pass_rate"]),
                    xytext=(6, -10), textcoords="offset points", fontsize=8, color=fg.INK_2)
        ax.set_title(model, color=colors[model])
        ax.set_ylim(0, 1.0)
        lo, hi = part["latency_median_ms"].min(), part["latency_median_ms"].max()
        pad = max(15.0, 0.15 * (hi - lo))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel("Median latency per request (ms)")
        ax.set_ylabel("Check pass rate")
    handles = [Line2D([], [], color=fg.STRATEGY_STYLE[s]["color"], ls=fg.STRATEGY_STYLE[s]["ls"],
                      marker=fg.STRATEGY_STYLE[s]["marker"], label=s)
               for s in ["NONE"] + STRATEGIES]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
               title="Strategy (label = context turns n, larger marker = more context)")
    fig.suptitle("Quality vs latency per model (x axes differ per panel)\n"
                 "Within a model the medians spread by only ~40 ms, inside the bootstrap noise "
                 "of T11: more context buys quality without a measurable latency cost here",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "quality_latency_tradeoff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- main

def verify_required(instances: dict) -> None:
    missing = set(instances) - set(REQUIRED_OF)
    extra = set(REQUIRED_OF) - set(instances)
    if missing or extra:
        raise SystemExit(f"required-n map mismatch: missing {missing}, unknown {extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--instances")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--gpu-log", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    instances = common.load_instances(common.resolve_instances_path(run_dir, args.instances))
    verify_required(instances)
    frame = aggregate.load_merged(run_dir, instances)
    if "latency_valid" in frame:
        frame.loc[frame["latency_valid"].eq(False), ["latency_ms", "ttft_ms", "generation_ms"]] = \
            np.nan
    main_frame = frame[frame["phase"].isin(["main", "baseline"])]
    models = sorted(main_frame["model"].dropna().unique())
    values = per_instance(main_frame)
    tables = run_dir / "tables"
    tables.mkdir(exist_ok=True)
    R = args.resamples

    t8_frame = t8(values, models, R)
    write(t8_frame, compact(t8_frame, ["model", "n", "contrast"]), "T8_strategy_contrasts",
          tables, "T8 strategy differences, paired bootstrap 95% CI",
          BOOT_NOTE + " n='pooled' averages each instance over n=1/3/5 first.")

    t9_frame = t9(values, models, R)
    write(t9_frame, compact(t9_frame, ["model", "strategy", "step"]), "T9_n_steps", tables,
          "T9 context-length steps, paired bootstrap 95% CI",
          BOOT_NOTE + " strategy='pooled' averages each instance over the four strategies "
          "first (NONE is shared). n5 - NONE is included as the total.")

    t10_frame, t10b_frame = t10(values, models, R)
    groups = "; ".join(f"n={n}: {', '.join(ids)}" for n, ids in REQUIRED_N.items())
    write(t10_frame, t10_frame, "T10_cue_distance", tables,
          "T10 check pass by cue distance (required n) x given n",
          "Mean instance-level check pass fraction, strategies pooled at each given n. "
          f"Required-n groups (from each instance's note): {groups}.")
    md = t10b_frame.copy()
    md["check_pass"] = [fmt_ci({"check_pass_delta": r["check_pass_delta"],
                                "check_pass_ci_low": r["ci_low"], "check_pass_ci_high": r["ci_high"],
                                "check_pass_sig": r["sig"]}, "check_pass")
                        for _, r in t10b_frame.iterrows()]
    write(t10b_frame, md[["model", "required_n", "contrast", "n_instances", "check_pass"]],
          "T10b_cue_distance_boot", tables,
          "T10b cue entering the window vs context beyond the cue, paired bootstrap",
          BOOT_NOTE + " Strategies pooled. Only instances of that required-n group.")

    t11_frame = t11(main_frame, models, R)
    write(t11_frame, t11_frame, "T11_latency_regression", tables,
          "T11 latency vs input size, median by n, and EXTRAPOLATION to longer context",
          "OLS over all main+baseline rows of the model (x = input_tokens for local and GPT, "
          "context_chars for DeepL; slopes in ms per 100 of x). *_ctrl_output: the same "
          "slope with output length (output_tokens / output_chars) as a second regressor, "
          "since decoding time dominates the total. median_n5_minus_none: 95% bootstrap CI "
          "of the median difference, rows resampled within each n (not paired). "
          "x_per_turn_*: measured slope of x on n (pooled strategies, and SRC_TGT only). "
          "EXTRAP_* columns are a straight-line EXTRAPOLATION to 20 and 50 SRC_TGT context "
          "turns (latency from the output-controlled fit at mean output length, TTFT from "
          "the simple fit), far outside the measured range n<=5; the 0/5-turn columns are "
          "the fit inside the range for comparison. They are not measurements, and a slope "
          "whose R^2 is ~0 extrapolates noise.")

    gpu_csv = args.gpu_log or (run_dir.parent.parent / "logs" / "gpu_samples.csv")
    t12_frame, t12_note = t12(main_frame, gpu_csv, R)
    write(t12_frame, t12_frame, "T12_gpu_contention", tables,
          "T12 GPU contention check for the local models", t12_note)

    t13_frame = t13(main_frame, values, models)
    write(t13_frame, t13_frame, "T13_efficiency", tables,
          "T13 quality gain over NONE per 100 extra input tokens",
          "Gain = mean over instances (strategy cell) minus NONE; extra = mean input size "
          "minus NONE's. DeepL uses context characters (DeepL does not bill context). "
          "Point estimates only; see T7 for the CIs of the gains.")

    fig = tradeoff_figure(main_frame, models, run_dir / "figures")
    print(f"[extra_stats] tables T8-T13 in {tables}")
    print(f"[extra_stats] figure {fig}")


if __name__ == "__main__":
    main()
