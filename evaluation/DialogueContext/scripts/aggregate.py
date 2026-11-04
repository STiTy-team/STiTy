"""Join translations with their scores and write tables and figures for one run.

Inputs (all in --run-dir): translations.jsonl, scores_reference.jsonl, scores_judge.jsonl,
and the instances (run-dir copy, else data/instances.jsonl). Missing score files are fine:
the columns stay empty and the tables say so.

Outputs:
  merged.jsonl / merged.csv       one row per non-warm-up job, translation fields + metrics
  tables/T0_conditions            every model x strategy x n cell (incl. corpus chrF++/BLEU)
  tables/T1_model_strategy        model x strategy (n pooled)
  tables/T2_context_length        n = 0/1/3/5 per model and overall (quality + latency)
  tables/T3_strategy              strategy (n pooled) per model and overall
  tables/T4_tag_pass              check pass rate per challenge tag x condition
  tables/T5_s1_generated_vs_gold  S1 self-generated history vs gold history (n=3)
  tables/T6_violations            format violations, failed jobs, judge error labels
  tables/T7_bootstrap_vs_none     paired bootstrap 95% CI of the gain over NONE per model
  figures/*.png

Runs in venv-metrics (pandas, numpy, sacrebleu, matplotlib):

    venv-metrics/bin/python evaluation/DialogueContext/scripts/aggregate.py \
        --run-dir evaluation/DialogueContext/results/<run_id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scoring_common as common

SEED = 20260925
STRATEGIES = ["SRC", "TGT", "SRC_TGT", "SPK_SRC_TGT"]
ALL_STRATEGIES = ["NONE"] + STRATEGIES
NS = [1, 3, 5]
DIMENSIONS = ["meaning_preservation", "no_hallucination", "contextual_correctness",
              "speaker_consistency", "register_consistency", "naturalness"]
ERROR_LABELS = ["omission", "addition", "wrong_coreference", "wrong_gender", "wrong_register",
                "lexical_inconsistency", "unnatural", "mistranslation", "context_copied",
                "format_violation"]
DROP_FROM_CSV = ["request", "prompt_system", "prompt_user", "raw_output"]


# --------------------------------------------------------------------------- loading

def condition_of(strategy, n) -> str:
    if strategy in (None, "NONE") or not n:
        return "NONE"
    return f"{strategy}@{int(n)}"


def load_merged(run_dir: Path, instances: dict) -> pd.DataFrame:
    rows = [r for r in common.latest_by_job(common.read_jsonl(run_dir / "translations.jsonl")).values()
            if r.get("phase") != "warmup"]
    reference = {r["job_id"]: r for r in common.read_jsonl(run_dir / "scores_reference.jsonl")}
    judge = {r["job_id"]: r for r in common.read_jsonl(run_dir / "scores_judge.jsonl")}
    merged = []
    for row in rows:
        out = dict(row)
        ref = reference.get(row["job_id"], {})
        for key in ("comet", "xcomet", "xcomet_error_spans", "chrf_pp", "bleu"):
            out[key] = ref.get(key)
        jud = judge.get(row["job_id"], {})
        for dim in DIMENSIONS:
            out[f"judge_{dim}"] = jud.get(f"judge_{dim}")
        out["judge_mean"] = jud.get("judge_mean")
        out["judge_error_labels"] = jud.get("judge_error_labels")
        out["judge_checks"] = jud.get("judge_checks")
        out["judge_check_pass_rate"] = jud.get("judge_check_pass_rate")
        out["judge_rationale"] = jud.get("judge_rationale")
        checks = jud.get("judge_checks") or []
        out["checks_passed"] = sum(bool(c["pass"]) for c in checks) if checks else None
        out["checks_total"] = len(checks) if checks else None
        instance = instances.get(row.get("instance_id")) or {}
        out["challenge_tags"] = instance.get("challenge_tags")
        out["condition"] = condition_of(row.get("context_strategy"), row.get("context_n"))
        out["failed"] = bool(row.get("error")) or row.get("hypothesis") is None
        merged.append(out)
    frame = pd.DataFrame(merged)
    if frame.empty:
        raise SystemExit("no non-warm-up rows in translations.jsonl")
    frame["context_n"] = frame["context_n"].fillna(0).astype(int)
    frame.loc[frame["context_strategy"].isna(), "context_strategy"] = "NONE"
    # pandas turns a missing value into NaN, and bool(NaN) is True: test for a real reason.
    frame["format_violation_flag"] = frame["format_violation"].apply(
        lambda v: isinstance(v, str) and bool(v.strip()) or (isinstance(v, bool) and v))
    numeric = ["comet", "xcomet", "chrf_pp", "bleu", "judge_mean", "judge_check_pass_rate",
               "checks_passed", "checks_total", "latency_ms", "ttft_ms", "generation_ms",
               "input_tokens", "output_tokens", "input_chars", "context_chars", "output_chars"]
    for col in numeric + [f"judge_{d}" for d in DIMENSIONS]:
        if col not in frame:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


# --------------------------------------------------------------------------- summaries

def _corpus(frame: pd.DataFrame) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    ok = frame[~frame["failed"] & frame["reference"].notna()]
    if ok.empty:
        return {"corpus_chrf_pp": np.nan, "corpus_bleu": np.nan}
    hyps = ok["hypothesis"].tolist()
    refs = [ok["reference"].tolist()]
    return {
        "corpus_chrf_pp": CHRF(word_order=2).corpus_score(hyps, refs).score,
        "corpus_bleu": BLEU(tokenize="13a").corpus_score(hyps, refs).score,
    }


def _pct(values: pd.Series, q: float) -> float:
    values = values.dropna()
    return float(np.percentile(values, q)) if len(values) else np.nan


def summarize(frame: pd.DataFrame, *, corpus: bool = True) -> dict:
    ok = frame[~frame["failed"]]
    passed = ok["checks_passed"].dropna().sum()
    total = ok["checks_total"].dropna().sum()
    out = {
        "n_rows": len(frame),
        "n_failed": int(frame["failed"].sum()),
        "n_judged": int(ok["judge_mean"].notna().sum()),
        "comet": ok["comet"].mean(),
        "xcomet": ok["xcomet"].mean() if ok["xcomet"].notna().any() else np.nan,
        "chrf_pp_sent": ok["chrf_pp"].mean(),
        "judge_mean": ok["judge_mean"].mean(),
        **{dim: ok[f"judge_{dim}"].mean() for dim in DIMENSIONS},
        "check_pass_rate": passed / total if total else np.nan,
        "checks_total": int(total),
        "latency_mean_ms": ok["latency_ms"].mean(),
        "latency_median_ms": ok["latency_ms"].median(),
        "latency_p95_ms": _pct(ok["latency_ms"], 95),
        "ttft_median_ms": ok["ttft_ms"].median() if ok["ttft_ms"].notna().any() else np.nan,
        "input_tokens_mean": ok["input_tokens"].mean(),
        "input_chars_mean": ok["input_chars"].mean(),
        "context_chars_mean": ok["context_chars"].mean(),
        "output_tokens_mean": ok["output_tokens"].mean(),
        "format_violation_rate": ok["format_violation_flag"].mean() if len(ok) else np.nan,
    }
    if corpus:
        out.update(_corpus(frame))
    return out


def grouped(frame: pd.DataFrame, keys: list[str], **kwargs) -> pd.DataFrame:
    records = []
    for values, group in frame.groupby(keys, sort=False, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        records.append({**dict(zip(keys, values)), **summarize(group, **kwargs)})
    return pd.DataFrame(records)


def with_overall(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-model rows plus 'ALL' rows that pool every model."""
    per_model = grouped(frame, ["model"] + keys)
    overall = grouped(frame, keys)
    overall.insert(0, "model", "ALL")
    return pd.concat([per_model, overall], ignore_index=True)


def order_frame(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    sort_cols = []
    if "model" in frame:
        order = {m: i for i, m in enumerate(models + ["ALL"])}
        frame["_m"] = frame["model"].map(order)
        sort_cols.append("_m")
    if "context_strategy" in frame:
        frame["_s"] = frame["context_strategy"].map({s: i for i, s in enumerate(ALL_STRATEGIES)})
        sort_cols.append("_s")
    if "context_n" in frame:
        sort_cols.append("context_n")
    if sort_cols:
        frame = frame.sort_values(sort_cols)
    return frame.drop(columns=[c for c in ("_m", "_s") if c in frame]).reset_index(drop=True)


# --------------------------------------------------------------------------- tag / S1 / violations

def tag_pass(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, tag, condition-view) with the pooled check pass rate."""
    records = []
    for _, row in frame[~frame["failed"]].iterrows():
        for check in row["judge_checks"] or []:
            base = {"model": row["model"], "tag": check.get("tag"), "pass": bool(check["pass"])}
            strategy, n = row["context_strategy"], row["context_n"]
            if strategy == "NONE":
                records.append({**base, "view": "NONE"})
            else:
                records.append({**base, "view": f"n={n}"})
                records.append({**base, "view": strategy})
    if not records:
        return pd.DataFrame(columns=["model", "tag", "view", "n_checks", "pass_rate"])
    long = pd.DataFrame(records)
    both = pd.concat([long, long.assign(model="ALL")])
    out = (both.groupby(["model", "tag", "view"])["pass"]
           .agg(n_checks="count", pass_rate="mean").reset_index())
    return out


VIEWS = ["NONE"] + [f"n={n}" for n in NS] + STRATEGIES


def s1_table(frame: pd.DataFrame) -> pd.DataFrame:
    s1 = frame[frame["phase"] == "s1"]
    records = []
    for (model, strategy, n), gen in s1.groupby(["model", "context_strategy", "context_n"]):
        gold = frame[(frame["phase"] == "main") & (frame["model"] == model)
                     & (frame["context_strategy"] == strategy) & (frame["context_n"] == n)]
        common_ids = set(gen["instance_id"]) & set(gold["instance_id"])
        for label, part in (("gold_history", gold), ("generated_history", gen)):
            part = part[part["instance_id"].isin(common_ids)]
            records.append({"model": model, "context_strategy": strategy, "context_n": n,
                            "history": label, "n_instances": len(common_ids),
                            **{k: v for k, v in summarize(part).items()
                               if k in ("n_rows", "n_failed", "comet", "xcomet", "judge_mean",
                                        "contextual_correctness", "speaker_consistency",
                                        "check_pass_rate", "corpus_chrf_pp", "corpus_bleu",
                                        "latency_median_ms")}})
    out = pd.DataFrame(records)
    if out.empty:
        return out
    deltas = []
    for (model, strategy, n), part in out.groupby(["model", "context_strategy", "context_n"]):
        gold = part[part["history"] == "gold_history"].iloc[0]
        gen = part[part["history"] == "generated_history"].iloc[0]
        row = {"model": model, "context_strategy": strategy, "context_n": n,
               "history": "delta(generated-gold)", "n_instances": gold["n_instances"]}
        for col in ("comet", "xcomet", "judge_mean", "contextual_correctness",
                    "speaker_consistency", "check_pass_rate", "corpus_chrf_pp", "corpus_bleu"):
            row[col] = gen[col] - gold[col]
        deltas.append(row)
    out = pd.concat([out, pd.DataFrame(deltas)], ignore_index=True)
    out["_h"] = out["history"].map({"gold_history": 0, "generated_history": 1,
                                    "delta(generated-gold)": 2})
    return out.sort_values(["model", "context_strategy", "_h"]).drop(columns="_h").reset_index(
        drop=True)


def condition_rank(condition: str) -> tuple:
    if condition == "NONE":
        return (0, 0)
    strategy, _, n = condition.partition("@")
    return (1 + ALL_STRATEGIES.index(strategy) if strategy in ALL_STRATEGIES else 9, int(n))


def violations(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = sorted(frame.groupby(["model", "condition"]).groups,
                  key=lambda k: (k[0], condition_rank(k[1])))
    for model, condition in keys:
        group = frame[(frame["model"] == model) & (frame["condition"] == condition)]
        ok = group[~group["failed"]]
        labels = [label for value in ok["judge_error_labels"].dropna() for label in value]
        records.append({
            "model": model, "condition": condition, "n_rows": len(group),
            "n_failed": int(group["failed"].sum()),
            "format_violations": int(ok["format_violation_flag"].sum()),
            "format_violation_rate": ok["format_violation_flag"].mean() if len(ok) else np.nan,
            "n_judged": int(ok["judge_mean"].notna().sum()),
            **{f"err_{label}": labels.count(label) for label in ERROR_LABELS},
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- bootstrap

def _instance_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Per instance: mean judge score, summed check passes and totals."""
    ok = frame[~frame["failed"]]
    return ok.groupby("instance_id").agg(
        judge=("judge_mean", "mean"),
        passed=("checks_passed", "sum"),
        total=("checks_total", "sum"),
    )


def bootstrap(frame: pd.DataFrame, n_resamples: int = 1000, seed: int = SEED) -> pd.DataFrame:
    main = frame[frame["phase"].isin(["main", "baseline"])]
    records = []
    for model, part in main.groupby("model"):
        base = _instance_values(part[part["condition"] == "NONE"])
        if base.empty:
            continue
        cells = [(c, part[part["condition"] == c]) for c in sorted(part["condition"].unique())
                 if c != "NONE"]
        cells += [(f"{s}@pooled", part[part["context_strategy"] == s]) for s in STRATEGIES
                  if (part["context_strategy"] == s).any()]
        for condition, cell in cells:
            values = _instance_values(cell)
            ids = sorted(set(values.index) & set(base.index))
            if not ids:
                continue
            a, b = values.loc[ids], base.loc[ids]
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, len(ids), size=(n_resamples, len(ids)))
            row = {"model": model, "condition": condition, "n_instances": len(ids),
                   "n_resamples": n_resamples, "seed": seed}
            judge_ok = a["judge"].notna().values & b["judge"].notna().values
            if judge_ok.any():
                diff = (a["judge"].values - b["judge"].values)
                point = np.nanmean(np.where(judge_ok, diff, np.nan))
                boot = np.nanmean(np.where(judge_ok[draws], diff[draws], np.nan), axis=1)
                row.update(judge_mean_delta=point,
                           judge_mean_ci_low=np.nanpercentile(boot, 2.5),
                           judge_mean_ci_high=np.nanpercentile(boot, 97.5))
            ta, tb = a["total"].values.astype(float), b["total"].values.astype(float)
            pa, pb = a["passed"].values.astype(float), b["passed"].values.astype(float)
            if ta.sum() and tb.sum():
                point = pa.sum() / ta.sum() - pb.sum() / tb.sum()
                with np.errstate(invalid="ignore", divide="ignore"):
                    boot = (pa[draws].sum(1) / ta[draws].sum(1)
                            - pb[draws].sum(1) / tb[draws].sum(1))
                row.update(check_pass_delta=point,
                           check_pass_ci_low=np.nanpercentile(boot, 2.5),
                           check_pass_ci_high=np.nanpercentile(boot, 97.5))
            records.append(row)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- writing

def to_markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    if frame.empty:
        return "_(no rows)_\n"
    cols = list(frame.columns)

    def fmt(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if abs(value) < 1000 else f"{value:.0f}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, name: str, tables: Path, title: str, note: str = "") -> None:
    frame.to_csv(tables / f"{name}.csv", index=False)
    body = f"# {title}\n\n" + (f"{note}\n\n" if note else "") + to_markdown(frame)
    (tables / f"{name}.md").write_text(body, encoding="utf-8")


def write_merged(frame: pd.DataFrame, run_dir: Path) -> None:
    records = json.loads(frame.to_json(orient="records", force_ascii=False))
    common.write_jsonl(run_dir / "merged.jsonl", records)
    flat = frame.drop(columns=[c for c in DROP_FROM_CSV if c in frame]).copy()
    for col in flat.columns:
        if flat[col].apply(lambda v: isinstance(v, (list, dict))).any():
            flat[col] = flat[col].apply(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    flat.to_csv(run_dir / "merged.csv", index=False)


QUALITY_COLS = ["comet", "xcomet", "judge_mean", "meaning_preservation",
                "contextual_correctness", "naturalness", "check_pass_rate",
                "corpus_chrf_pp", "corpus_bleu"]
LATENCY_COLS = ["latency_mean_ms", "latency_median_ms", "latency_p95_ms", "ttft_median_ms",
                "input_tokens_mean", "input_chars_mean", "context_chars_mean"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--instances")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    instances = common.load_instances(common.resolve_instances_path(run_dir, args.instances))
    frame = load_merged(run_dir, instances)
    write_merged(frame, run_dir)
    # The runner marks rows whose timing is not comparable (e.g. S1 run in parallel) with
    # latency_valid=false. merged.* keep the raw numbers; the statistics leave them out.
    if "latency_valid" in frame:
        invalid = frame["latency_valid"].eq(False)
        frame.loc[invalid, ["latency_ms", "ttft_ms", "generation_ms"]] = np.nan
        if invalid.any():
            print(f"[aggregate] {int(invalid.sum())} rows with latency_valid=false "
                  f"excluded from latency statistics")
    models = sorted(frame["model"].dropna().unique())
    tables = run_dir / "tables"
    tables.mkdir(exist_ok=True)
    main_frame = frame[frame["phase"].isin(["main", "baseline"])]
    n_ref = frame["comet"].notna().sum()
    n_judge = frame["judge_mean"].notna().sum()
    note = (f"Rows: {len(frame)} non-warm-up ({len(main_frame)} main+baseline). "
            f"COMET scored {n_ref}, judged {n_judge}. check_pass_rate pools every check in "
            f"the cell. Latency columns are over successful jobs.")

    t0 = order_frame(grouped(main_frame, ["model", "context_strategy", "context_n"]), models)
    write_table(t0, "T0_conditions", tables, "T0 every condition (model x strategy x n)", note)

    t1 = order_frame(grouped(main_frame, ["model", "context_strategy"]), models)
    write_table(t1[["model", "context_strategy", "n_rows", "n_failed"] + QUALITY_COLS
                   + ["latency_median_ms"]],
                "T1_model_strategy", tables, "T1 model x strategy (n pooled)", note)

    t2 = order_frame(with_overall(main_frame, ["context_n"]), models)
    write_table(t2[["model", "context_n", "n_rows", "comet", "judge_mean", "check_pass_rate",
                    "corpus_chrf_pp"] + LATENCY_COLS],
                "T2_context_length", tables,
                "T2 context length (n=0 is NONE; strategies pooled)", note)

    t3 = order_frame(with_overall(main_frame, ["context_strategy"]), models)
    write_table(t3[["model", "context_strategy", "n_rows"] + QUALITY_COLS
                   + ["latency_mean_ms", "latency_median_ms", "latency_p95_ms",
                      "ttft_median_ms", "input_tokens_mean"]],
                "T3_strategy", tables, "T3 strategy (n pooled)", note)

    t4_long = tag_pass(main_frame)
    t4_long.to_csv(tables / "T4_tag_pass_long.csv", index=False)
    if not t4_long.empty:
        t4 = t4_long.pivot_table(index=["model", "tag"], columns="view", values="pass_rate")
        t4 = t4.reindex(columns=[v for v in VIEWS if v in t4.columns]).reset_index()
        counts = t4_long.pivot_table(index=["model", "tag"], columns="view", values="n_checks")
        t4["n_checks_NONE"] = [counts.loc[(m, t)].get("NONE", np.nan) for m, t in
                               zip(t4["model"], t4["tag"])]
        t4["n_checks_NONE"] = t4["n_checks_NONE"].astype("Int64")
        t4 = order_frame(t4, models)
    else:
        t4 = pd.DataFrame()
    write_table(t4, "T4_tag_pass", tables, "T4 check pass rate by challenge tag and condition",
                "Columns: NONE baseline, context length pooled over strategies (n=k), "
                "strategy pooled over n. Long form with counts: T4_tag_pass_long.csv.")

    t5 = s1_table(frame)
    write_table(t5, "T5_s1_generated_vs_gold", tables,
                "T5 S1: model's own earlier translations vs gold history",
                "Same model, strategy and n; restricted to instances present in both.")

    t6 = violations(main_frame)
    s1_viol = violations(frame[frame["phase"] == "s1"])
    if not s1_viol.empty:
        s1_viol["condition"] = "S1:" + s1_viol["condition"]
        t6 = pd.concat([t6, s1_viol], ignore_index=True)
    write_table(t6, "T6_violations", tables,
                "T6 format violations, failed jobs and judge error labels")

    t7 = bootstrap(frame, n_resamples=args.resamples)
    write_table(t7, "T7_bootstrap_vs_none", tables,
                "T7 gain over NONE, paired bootstrap 95% CI",
                f"Resampling instances with replacement, {args.resamples} resamples, "
                f"seed {SEED}. '@pooled' averages each instance over n=1/3/5 first.")

    print(f"[aggregate] merged {len(frame)} rows -> {run_dir / 'merged.jsonl'}")
    print(f"[aggregate] tables in {tables}")
    if not args.no_figures:
        import figures
        paths = figures.draw_all(frame, run_dir / "figures", models)
        for path in paths:
            print(f"[aggregate] figure {path}")


if __name__ == "__main__":
    main()
