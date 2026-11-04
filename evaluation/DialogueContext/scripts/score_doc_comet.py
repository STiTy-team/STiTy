"""Doc-COMET scores for every row that score_reference.py scored.

Doc-COMET (Vernikos et al., WMT 2022) runs the same Unbabel/wmt22-comet-da checkpoint,
but prepends the previous turns of the dialogue to the source and the reference,
joined with the tokenizer's separator, and pools only the tokens of the current turn.
The hypothesis gets the *reference* previous turns as its context, so an earlier
mistake of the system is not charged again. The scoring context is therefore the
same for every experimental condition; only the current hypothesis differs.

`--windows 0` reproduces plain COMET and is always run as a check against
scores_reference.jsonl.

    venv-metrics/bin/python evaluation/DialogueContext/scripts/score_doc_comet.py \
        --run-dir evaluation/DialogueContext/results/<run_id>
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import scoring_common as common

DEFAULT_COMET = "Unbabel/wmt22-comet-da"


def context_of(turns: dict, dialogue_id: str, turn_id: int, window: int) -> list[dict]:
    return [turns[(dialogue_id, t)] for t in range(max(0, turn_id - window), turn_id)
            if (dialogue_id, t) in turns]


def with_context(current: str, previous: list[str], sep: str) -> str:
    return f" {sep} ".join(previous + [current])


def load_model(model_name: str):
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(model_name))
    model.enable_context()
    if not model.use_context:
        raise SystemExit(f"{model_name} does not support context (needs average pooling)")
    return model


def run_doc_comet(model, samples_by_window: dict[int, list[dict]],
                  batch_size: int) -> dict[int, list[float]]:
    import torch

    gpus = 1 if torch.cuda.is_available() else 0
    out = {}
    for window, samples in samples_by_window.items():
        started = time.time()
        output = model.predict(samples, batch_size=batch_size, gpus=gpus, progress_bar=False)
        out[window] = [float(s) for s in output.scores]
        print(f"[doc-comet] window={window}: {len(samples)} triples in "
              f"{time.time() - started:.1f}s", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--comet-model", default=DEFAULT_COMET)
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 5],
                        help="previous turns used as scoring context (paper default: 2)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_dir = args.run_dir
    reference = {r["job_id"]: r for r in common.read_jsonl(run_dir / "scores_reference.jsonl")}
    rows = [r for r in common.latest_by_job(common.read_jsonl(run_dir / "translations.jsonl")).values()
            if r["job_id"] in reference]
    turns = {(t["dialogue_id"], t["turn_id"]): t
             for t in common.read_jsonl(run_dir / "dialogues.jsonl")}
    print(f"[doc-comet] {len(rows)} rows scored by score_reference.py", flush=True)

    model = load_model(args.comet_model)
    sep = model.encoder.tokenizer.sep_token

    windows = sorted(set([0] + args.windows))
    samples_by_window: dict[int, list[dict]] = {}
    keys_by_window: dict[int, list[str]] = {}
    row_keys: dict[str, dict[int, str]] = {}
    for window in windows:
        seen: dict[str, dict] = {}
        for row in rows:
            prev = context_of(turns, row["dialogue_id"], row["target_turn_id"], window)
            src = with_context(row["current_source"], [t["ko"] for t in prev], sep)
            ref = with_context(row["reference"], [t["en"] for t in prev], sep)
            mt = with_context(row["hypothesis"], [t["en"] for t in prev], sep)
            key = common.digest(src, mt, ref)
            row_keys.setdefault(row["job_id"], {})[window] = key
            seen.setdefault(key, {"src": src, "mt": mt, "ref": ref})
        keys_by_window[window] = list(seen)
        samples_by_window[window] = list(seen.values())
        print(f"[doc-comet] window={window}: {len(seen)} unique triples", flush=True)

    scores = run_doc_comet(model, samples_by_window, args.batch_size)
    by_key = {w: dict(zip(keys_by_window[w], scores[w])) for w in windows}

    out_rows, drift = [], []
    for row in rows:
        entry = {"job_id": row["job_id"], "comet_model": args.comet_model}
        for window in windows:
            entry[f"doc_comet_w{window}"] = by_key[window][row_keys[row["job_id"]][window]]
        drift.append(abs(entry["doc_comet_w0"] - reference[row["job_id"]]["comet"]))
        out_rows.append(entry)
    common.write_jsonl(run_dir / "scores_doc_comet.jsonl", out_rows)

    status = {"model": args.comet_model, "at": datetime.now(timezone.utc).isoformat(),
              "windows": windows, "n_rows": len(out_rows), "separator": sep,
              "w0_vs_comet_max_abs_diff": max(drift), "w0_vs_comet_mean_abs_diff": sum(drift) / len(drift)}
    (run_dir / "doc_comet_status.json").write_text(json.dumps(status, indent=2))
    print(f"[doc-comet] window 0 vs stored COMET: max |diff| {max(drift):.2e}", flush=True)


if __name__ == "__main__":
    main()
