"""Meta-evaluation: how well COMET, Doc-COMET, XCOMET, MetricX-24 and chrF++ agree with the judge.

Main and baseline rows only. The judge (gpt-6-sol) is the yardstick; it is an LLM, not
a human, so these numbers say "agrees with the judge", not "is correct". MetricX is an error
score, so it enters every table as `metricx_neg` (sign flipped, higher = better). Writes into tables/:

  T15a_pairwise        segment pairwise accuracy within an instance, per judge target
  T15b_pairwise_diff   bootstrap CIs (resampling instances) of accuracy differences
  T15c_tag             pairwise accuracy on pairs where one planted check flips, per tag
  T15d_error_labels    within-instance score gap (SD units) for rows carrying a judge error label
  T15e_system          system level: 52 model x condition cells, and per-model condition ranking
  T15f_best_condition  the condition each metric would pick per model, vs the judge
  T15g_probes          score drop for defects planted in the gold reference (probe_scores.jsonl)
  T15h_gain            gain of each context condition over NONE, in within-instance SD units

    venv-tf5/bin/python evaluation/DialogueContext/scripts/metric_meta_eval.py \
        --run-dir evaluation/DialogueContext/results/<run_id>
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import scoring_common as common
from aggregate import load_merged, write_table

METRICS = ["comet", "doc_comet_w2", "doc_comet_w5", "xcomet", "metricx_neg", "chrf_pp"]
TARGETS = ["judge_check_pass_rate", "judge_mean", "judge_contextual_correctness",
           "judge_meaning_preservation", "judge_naturalness"]
CONTRASTS = [("doc_comet_w2", "comet"), ("xcomet", "comet"), ("doc_comet_w2", "xcomet"),
             ("doc_comet_w5", "doc_comet_w2"), ("metricx_neg", "comet"),
             ("metricx_neg", "doc_comet_w2"), ("metricx_neg", "xcomet")]
PROBE_METRICS = ["comet", "doc_comet_w2", "xcomet", "metricx_neg"]


def load(run_dir: Path) -> pd.DataFrame:
    frame = load_merged(run_dir, common.load_instances(common.resolve_instances_path(run_dir, None)))
    doc = pd.DataFrame(common.read_jsonl(run_dir / "scores_doc_comet.jsonl")).drop(columns="comet_model")
    metricx = pd.DataFrame(common.read_jsonl(run_dir / "scores_metricx.jsonl"))
    frame = frame.merge(doc, on="job_id", how="left").merge(metricx[["job_id", "metricx"]], on="job_id", how="left")
    frame["metricx_neg"] = -frame["metricx"]
    return frame[frame["phase"].isin(["baseline", "main"]) & ~frame["failed"]].reset_index(drop=True)


def pair_outcomes(group: pd.DataFrame, target: str) -> dict[str, list[float]]:
    """Per metric, 1/0/0.5 for each pair the judge separates (identical hypotheses skipped)."""
    out = {m: [] for m in METRICS}
    rows = group[group[target].notna()].to_dict("records")
    for a, b in combinations(rows, 2):
        if a["hypothesis"] == b["hypothesis"] or a[target] == b[target]:
            continue
        sign = np.sign(a[target] - b[target])
        for m in METRICS:
            d = np.sign(a[m] - b[m])
            out[m].append(0.5 if d == 0 else float(d == sign))
    return out


def per_instance(frame: pd.DataFrame, target: str) -> dict[str, dict[str, list[float]]]:
    return {iid: pair_outcomes(g, target) for iid, g in frame.groupby("instance_id")}


def accuracy(per_inst: dict, ids, metric: str) -> float:
    vals = [v for i in ids for v in per_inst[i][metric]]
    return float(np.mean(vals)) if vals else np.nan


def bootstrap(per_inst: dict, fn, resamples: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ids = np.array(list(per_inst))
    draws = [fn(rng.choice(ids, len(ids))) for _ in range(resamples)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def pairwise_tables(frame: pd.DataFrame, resamples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc_rows, diff_rows = [], []
    for target in TARGETS:
        per_inst = per_instance(frame, target)
        ids = list(per_inst)
        n_pairs = sum(len(per_inst[i]["comet"]) for i in ids)
        row = {"judge_target": target, "n_pairs": n_pairs}
        for m in METRICS:
            row[m] = accuracy(per_inst, ids, m)
            lo, hi = bootstrap(per_inst, lambda s, m=m: accuracy(per_inst, s, m), resamples)
            row[f"{m}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        acc_rows.append(row)
        for a, b in CONTRASTS:
            diff = accuracy(per_inst, ids, a) - accuracy(per_inst, ids, b)
            lo, hi = bootstrap(per_inst, lambda s: accuracy(per_inst, s, a) - accuracy(per_inst, s, b),
                               resamples)
            diff_rows.append({"judge_target": target, "contrast": f"{a} - {b}", "diff": diff,
                              "ci_low": lo, "ci_high": hi, "excludes_0": lo > 0 or hi < 0})
    return pd.DataFrame(acc_rows), pd.DataFrame(diff_rows)


def tag_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairs of the same instance where one check (by index) passes in one row and fails in the other."""
    outcomes: dict[str, dict[str, list[float]]] = {}
    for _, g in frame.groupby("instance_id"):
        rows = [r for r in g.to_dict("records") if r.get("judge_checks")]
        for a, b in combinations(rows, 2):
            if a["hypothesis"] == b["hypothesis"]:
                continue
            for ca, cb in zip(a["judge_checks"], b["judge_checks"]):
                if bool(ca["pass"]) == bool(cb["pass"]):
                    continue
                sign = 1.0 if ca["pass"] else -1.0
                bucket = outcomes.setdefault(ca["tag"], {m: [] for m in METRICS})
                for m in METRICS:
                    d = np.sign(a[m] - b[m])
                    bucket[m].append(0.5 if d == 0 else float(d == sign))
    rows = [{"tag": tag, "n_pairs": len(v["comet"]), **{m: np.mean(v[m]) for m in METRICS}}
            for tag, v in outcomes.items()]
    return pd.DataFrame(rows).sort_values("n_pairs", ascending=False)


def error_label_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean within-instance z-score of rows with the label minus rows without it."""
    z = frame.copy()
    for m in METRICS:
        z[m] = z.groupby("instance_id")[m].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    labels = sorted({l for ls in frame["judge_error_labels"].dropna() for l in ls})
    rows = []
    for label in labels:
        has = z["judge_error_labels"].apply(lambda ls: isinstance(ls, list) and label in ls)
        gaps = {m: [] for m in METRICS}
        for _, g in z.assign(has=has).groupby("instance_id"):
            if g["has"].any() and (~g["has"]).any():
                for m in METRICS:
                    gaps[m].append(g.loc[g["has"], m].mean() - g.loc[~g["has"], m].mean())
        if gaps["comet"]:
            rows.append({"error_label": label, "n_rows": int(has.sum()),
                         "n_instances": len(gaps["comet"]),
                         **{m: float(np.mean(gaps[m])) for m in METRICS}})
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False)


def system_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = frame.groupby(["model", "condition"]).agg(
        **{m: (m, "mean") for m in METRICS}, judge_mean=("judge_mean", "mean"),
        passed=("checks_passed", "sum"), total=("checks_total", "sum")).reset_index()
    cells["judge_check_pass_rate"] = cells["passed"] / cells["total"]
    rows = []
    for target in ("judge_check_pass_rate", "judge_mean"):
        row = {"level": "52 cells, Spearman", "judge_target": target}
        row.update({m: cells[m].corr(cells[target], method="spearman") for m in METRICS})
        rows.append(row)
        row = {"level": "52 cells, pairwise accuracy", "judge_target": target}
        for m in METRICS:
            hits = [float(np.sign(a[m] - b[m]) == np.sign(a[target] - b[target]))
                    for a, b in combinations(cells.to_dict("records"), 2) if a[target] != b[target]]
            row[m] = np.mean(hits)
        rows.append(row)
        row = {"level": "condition ranking within model, mean Spearman", "judge_target": target}
        row.update({m: np.mean([g[m].corr(g[target], method="spearman")
                                for _, g in cells.groupby("model")]) for m in METRICS})
        rows.append(row)
    best = []
    for model, g in cells.groupby("model"):
        row = {"model": model}
        for col in ["judge_check_pass_rate", "judge_mean"] + METRICS:
            top = g.sort_values(col, ascending=False).iloc[0]
            row[col] = top["condition"]
        best.append(row)
    return pd.DataFrame(rows), pd.DataFrame(best)


def within_sd(frame: pd.DataFrame, metric: str) -> float:
    return float(np.sqrt(frame.groupby("instance_id")[metric].var().mean()))


def probe_table(frame: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    probes = pd.DataFrame(common.read_jsonl(run_dir / "probe_scores.jsonl"))
    probes["metricx_neg"] = -probes["metricx"]
    sd = {m: within_sd(frame, m) for m in PROBE_METRICS}
    base = probes[probes["probe"] == "ref"].set_index("instance_id")
    rows = [{"probe": "ref (absolute score)", "n": len(base),
             **{f"{m}_raw": float(base[m].mean()) for m in PROBE_METRICS}}]
    for kind, g in probes[probes["probe"] != "ref"].groupby("probe"):
        row = {"probe": kind, "n": len(g)}
        for m in PROBE_METRICS:
            drop = g.set_index("instance_id")[m] - base[m].reindex(g["instance_id"]).values
            row[f"{m}_drop_sd"] = float(drop.mean() / sd[m])
            row[f"{m}_raw"] = float(drop.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def gain_table(frame: pd.DataFrame) -> pd.DataFrame:
    means = frame.groupby("condition")[METRICS].mean()
    gains = means - means.loc["NONE"]
    for m in METRICS:
        gains[m] = gains[m] / within_sd(frame, m)
    gains = gains.drop("NONE")
    gains.loc["mean over conditions"] = gains.mean()
    return gains.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()

    frame = load(args.run_dir)
    tables = args.run_dir / "tables"
    note = (f"{len(frame)} rows, {frame['instance_id'].nunique()} instances. Judge = gpt-6-sol. "
            "Pairs with identical hypotheses and judge ties are skipped; a metric tie counts 0.5. "
            "CIs resample instances.")
    acc, diff = pairwise_tables(frame, args.resamples)
    write_table(acc.round(3), "T15a_pairwise", tables, "T15a Segment pairwise accuracy vs the judge", note)
    write_table(diff.round(3), "T15b_pairwise_diff", tables, "T15b Pairwise accuracy differences", note)
    write_table(tag_table(frame).round(3), "T15c_tag", tables,
                "T15c Pairwise accuracy on pairs where one planted check flips", note)
    write_table(error_label_table(frame).round(3), "T15d_error_labels", tables,
                "T15d Within-instance score gap (SD units) of rows with each judge error label",
                "Negative = the metric scores labelled rows lower, as it should.")
    system, best = system_tables(frame)
    write_table(system.round(3), "T15e_system", tables, "T15e System-level agreement with the judge")
    write_table(best, "T15f_best_condition", tables, "T15f Best condition per model by each metric")
    write_table(probe_table(frame, args.run_dir).round(3), "T15g_probes", tables,
                "T15g Score drop for defects planted in the gold reference",
                "*_drop_sd = mean drop / within-instance SD of that metric on the real run. "
                "More negative = the metric notices the defect more. MetricX sign flipped.")
    write_table(gain_table(frame).round(3), "T15h_gain", tables,
                "T15h Gain over NONE in within-instance SD units",
                "Models pooled. Larger = the metric shows the context effect more clearly.")
    print("[meta-eval] wrote T15a-T15h", flush=True)


if __name__ == "__main__":
    main()
