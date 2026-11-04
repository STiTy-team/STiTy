"""Tables comparing Doc-COMET (scores_doc_comet.jsonl) with sentence COMET and XCOMET.

Main and baseline rows only (S1 is left out, as in T1-T3). Writes into tables/:

  T14_doc_comet_condition   models pooled, one row per condition, with the gain over NONE
  T14b_doc_comet_model      model x condition
  T14c_doc_comet_agreement  how each metric tracks the judge (row-level Spearman)

    venv-tf5/bin/python evaluation/DialogueContext/scripts/doc_comet_tables.py \
        --run-dir evaluation/DialogueContext/results/<run_id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import scoring_common as common
from aggregate import load_merged, write_table

CONDITION_ORDER = ["NONE"] + [f"{s}@{n}" for s in ("SRC", "TGT", "SRC_TGT", "SPK_SRC_TGT")
                              for n in (1, 3, 5)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    instances = common.load_instances(common.resolve_instances_path(run_dir, None))
    frame = load_merged(run_dir, instances)
    doc = pd.DataFrame(common.read_jsonl(run_dir / "scores_doc_comet.jsonl"))
    windows = sorted(int(c.rsplit("w", 1)[1]) for c in doc.columns if c.startswith("doc_comet_w"))
    windows = [w for w in windows if w > 0]
    frame = frame.merge(doc, on="job_id", how="left")
    frame = frame[frame["phase"].isin(["baseline", "main"]) & ~frame["failed"]]
    metrics = ["comet"] + [f"doc_comet_w{w}" for w in windows] + ["xcomet"]
    tables = run_dir / "tables"

    by_cond = frame.groupby("condition")[metrics].mean()
    by_cond.insert(0, "n_rows", frame.groupby("condition").size())
    by_cond = by_cond.reindex(CONDITION_ORDER)
    for m in metrics:
        by_cond[f"{m}_gain"] = by_cond[m] - by_cond.loc["NONE", m]
    write_table(by_cond.reset_index().round(4), "T14_doc_comet_condition", tables,
                "T14 Doc-COMET vs COMET / XCOMET by condition (models pooled)",
                f"Doc-COMET = Unbabel/wmt22-comet-da with the previous w gold turns (source and "
                f"reference) prepended, pooled over the current turn only; w in {windows}. "
                "The scoring context is identical for every condition. *_gain = minus NONE.")

    by_model = frame.groupby(["model", "condition"])[metrics].mean().reset_index()
    by_model["condition"] = pd.Categorical(by_model["condition"], CONDITION_ORDER, ordered=True)
    by_model = by_model.sort_values(["model", "condition"])
    for m in metrics:
        none = by_model[by_model["condition"] == "NONE"].set_index("model")[m]
        by_model[f"{m}_gain"] = by_model[m] - by_model["model"].map(none)
    write_table(by_model.round(4), "T14b_doc_comet_model", tables,
                "T14b Doc-COMET vs COMET / XCOMET by model and condition")

    rows = []
    for target in ("judge_mean", "judge_contextual_correctness", "judge_check_pass_rate"):
        sub = frame[frame[target].notna()]
        rows.append({"judge_target": target, "n": len(sub),
                     **{m: sub[m].corr(sub[target], method="spearman") for m in metrics}})
    write_table(pd.DataFrame(rows).round(4), "T14c_doc_comet_agreement", tables,
                "T14c Row-level Spearman between each metric and the judge")


if __name__ == "__main__":
    main()
