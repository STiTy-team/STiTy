"""Controlled probes: perturb each gold reference and see how far each metric drops.

For every instance the hypothesis is the reference itself, or the reference with one
planted defect:

  copy_context   the previous turn's reference prepended (the context leaked into the output)
  truncate       only the first 60% of the words
  pronoun_flip   she/he, her/him/his swapped (instances with such a pronoun only)
  drop_pronoun   the first subject pronoun removed (a Korean-style omitted argument)

Scores COMET, Doc-COMET (w2) and XCOMET-XL into probe_scores.jsonl. score_metricx.py
adds MetricX to the same file, and metric_meta_eval.py turns it into tables/T15g_probes.

    venv-metrics/bin/python evaluation/DialogueContext/scripts/metric_probes.py \
        --run-dir evaluation/DialogueContext/results/<run_id>
"""
from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import scoring_common as common
from score_doc_comet import context_of, load_model, with_context

SWAP = {"she": "he", "he": "she", "her": "him", "him": "her", "his": "her", "hers": "his",
        "herself": "himself", "himself": "herself"}
# Only a pronoun standing alone: "She's" -> "'s" would be a broken sentence, not an omission.
SUBJECTS = r"\b(I|you|he|she|we|they|it)\b(?!['’]) ?"


def pronoun_flip(text: str) -> str:
    def repl(m):
        word = m.group(0)
        new = SWAP[word.lower()]
        return new.capitalize() if word[0].isupper() else new
    return re.sub(r"\b(" + "|".join(SWAP) + r")\b", repl, text, flags=re.IGNORECASE)


def probes(row: dict, prev_ref: str | None) -> dict[str, str]:
    ref = row["reference_translation"]
    words = ref.split()
    out = {"ref": ref, "truncate": " ".join(words[:max(1, int(len(words) * 0.6))])}
    if prev_ref:
        out["copy_context"] = f"{prev_ref} {ref}"
    flipped = pronoun_flip(ref)
    if flipped != ref:
        out["pronoun_flip"] = flipped
    match = re.search(SUBJECTS, ref, flags=re.IGNORECASE)
    if match:
        dropped = ref[:match.start()] + ref[match.end():]
        out["drop_pronoun"] = dropped[:1].upper() + dropped[1:] if match.start() == 0 else dropped
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()

    import torch
    from comet import download_model, load_from_checkpoint

    run_dir = args.run_dir
    turns = {(t["dialogue_id"], t["turn_id"]): t for t in common.read_jsonl(run_dir / "dialogues.jsonl")}
    items = []
    for inst in common.read_jsonl(run_dir / "instances.jsonl"):
        prev = context_of(turns, inst["dialogue_id"], inst["target_turn_id"], 1)
        for kind, hyp in probes(inst, prev[0]["en"] if prev else None).items():
            items.append({"instance_id": inst["instance_id"], "probe": kind,
                          "dialogue_id": inst["dialogue_id"], "turn_id": inst["target_turn_id"],
                          "src": inst["current_source"], "mt": hyp, "ref": inst["reference_translation"]})
    print(f"[probes] {len(items)} probe hypotheses", flush=True)
    gpus = 1 if torch.cuda.is_available() else 0
    plain = [{"src": i["src"], "mt": i["mt"], "ref": i["ref"]} for i in items]

    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    comet = model.predict(plain, batch_size=16, gpus=gpus, progress_bar=False).scores
    doc_model = load_model("Unbabel/wmt22-comet-da")
    sep = doc_model.encoder.tokenizer.sep_token
    doc = []
    for i in items:
        prev = context_of(turns, i["dialogue_id"], i["turn_id"], args.window)
        doc.append({"src": with_context(i["src"], [t["ko"] for t in prev], sep),
                    "mt": with_context(i["mt"], [t["en"] for t in prev], sep),
                    "ref": with_context(i["ref"], [t["en"] for t in prev], sep)})
    doc_scores = doc_model.predict(doc, batch_size=16, gpus=gpus, progress_bar=False).scores
    del model, doc_model
    gc.collect()
    torch.cuda.empty_cache()
    xmodel = load_from_checkpoint(download_model("Unbabel/XCOMET-XL"))
    xcomet = xmodel.predict(plain, batch_size=8, gpus=gpus, progress_bar=False).scores

    for i, c, d, x in zip(items, comet, doc_scores, xcomet):
        i.update(comet=float(c), doc_comet_w2=float(d), xcomet=float(x))
    common.write_jsonl(run_dir / "probe_scores.jsonl", items)

    print(f"[probes] wrote {len(items)} rows to probe_scores.jsonl", flush=True)


if __name__ == "__main__":
    main()
