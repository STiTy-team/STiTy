"""Reference-based scores for every scorable row of a DialogueContext run.

Per row: COMET (Unbabel/wmt22-comet-da, src = current_source, ref = reference),
chrF++ (sacreBLEU, word_order=2), sentence BLEU (13a, effective order), and XCOMET-XL
(score + error spans) when the gated checkpoint is reachable. Corpus-level chrF++/BLEU
per condition are computed later by aggregate.py.

Identical (src, hyp, ref) triples are scored once. Triple scores are kept in
`reference_cache.jsonl`, so rerunning after more rows arrive only scores the new ones.
Whether XCOMET ran, and if not exactly why, goes to `xcomet_status.json`.

Runs in venv-metrics (transformers 4.57 + unbabel-comet 2.2.7):

    venv-metrics/bin/python evaluation/DialogueContext/scripts/score_reference.py \
        --run-dir evaluation/DialogueContext/results/<run_id>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import scoring_common as common

DEFAULT_COMET = "Unbabel/wmt22-comet-da"
DEFAULT_XCOMET = "Unbabel/XCOMET-XL"


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _gpus() -> int:
    try:
        import torch
        return 1 if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def run_comet(triples: list[dict], model_name: str, batch_size: int) -> list[float]:
    from core.utils.metrics.meaning import comet_score

    result = comet_score(
        [{"id": t["key"], "src": t["src"], "mt": t["mt"], "ref": t["ref"]} for t in triples],
        model_name=model_name, batch_size=batch_size)
    return [result["per_item"][t["key"]] for t in triples]


def _load_xcomet(model_name: str, token: str):
    """Download with an explicit token (never touches the machine's stored login)."""
    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download

    folder = snapshot_download(model_name, token=token or None)
    return load_from_checkpoint(os.path.join(folder, "checkpoints", "model.ckpt"))


def run_xcomet(triples: list[dict], model_name: str, batch_size: int,
               status_path: Path) -> list[dict] | None:
    """Score with XCOMET; on any failure write the reason and return None."""
    status = {"model": model_name, "at": datetime.now(timezone.utc).isoformat(),
              "n_requested": len(triples)}
    token = common.env_value("HF_TOKEN")
    started = time.time()
    try:
        model = _load_xcomet(model_name, token)
    except Exception as exc:  # gated repo, network, missing token, bad checkpoint
        status.update(status="unavailable", stage="load",
                      reason=f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1] if str(exc).strip() else ''}",
                      hf_token_present=bool(token))
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        print(f"[xcomet] unavailable: {status['reason']}", flush=True)
        return None
    samples = [{"src": t["src"], "mt": t["mt"], "ref": t["ref"]} for t in triples]
    output = None
    tried = []
    for size in dict.fromkeys([batch_size, max(1, batch_size // 4), 1]):
        tried.append(size)
        try:
            output = model.predict(samples, batch_size=size, gpus=_gpus())
            break
        except Exception as exc:
            oom = "out of memory" in str(exc).lower()
            print(f"[xcomet] batch_size={size} failed ({type(exc).__name__}"
                  f"{', OOM' if oom else ''})", flush=True)
            _free_gpu()
            if not oom:
                status.update(status="failed", stage="predict", batch_sizes_tried=tried,
                              reason=f"{type(exc).__name__}: {exc}")
                break
            status.update(status="failed", stage="predict", batch_sizes_tried=tried,
                          reason=f"CUDA out of memory at batch_size={size}")
    peak = None
    try:
        import torch
        if torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    del model
    _free_gpu()
    if output is None:
        status["peak_gpu_gb"] = peak
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        print(f"[xcomet] failed: {status.get('reason')}", flush=True)
        return None
    spans = getattr(getattr(output, "metadata", None), "error_spans", None) or [[]] * len(triples)
    results = []
    for score, span_list in zip(output.scores, spans):
        results.append({
            "xcomet": float(score),
            "xcomet_error_spans": [
                {"text": s.get("text"), "severity": s.get("severity"),
                 "confidence": round(float(s.get("confidence", 0.0)), 4),
                 "start": s.get("start"), "end": s.get("end")}
                for s in (span_list or [])
            ],
        })
    status.update(status="ok", n_scored=len(results), batch_size=tried[-1],
                  peak_gpu_gb=peak, seconds=round(time.time() - started, 1),
                  reason=None)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"[xcomet] scored {len(results)} triples in {status['seconds']}s", flush=True)
    return results


def _isolated(function, *args):
    """Run one GPU model in a fresh process, so its memory is fully returned afterwards.

    Measured: after COMET-DA ran in the same process, XCOMET-XL (14.1 GB peak alone)
    hit CUDA OOM even at batch size 1 on a 24 GB card with another 6 GB user.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # Not multiprocessing.Pool: its workers are daemonic and COMET's predict() starts
    # its own worker processes, which daemonic processes may not do.
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(function, *args).result()


def lexical(hyp: str, ref: str) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    chrf = CHRF(char_order=6, word_order=2, beta=2)
    bleu = BLEU(tokenize="13a", effective_order=True)
    return {"chrf_pp": float(chrf.sentence_score(hyp, [ref]).score),
            "bleu": float(bleu.sentence_score(hyp, [ref]).score)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--instances", help="instances.jsonl (default: run-dir copy, else data/)")
    parser.add_argument("--comet-model",
                        default=os.environ.get("STITY_COMET_MODEL") or DEFAULT_COMET)
    parser.add_argument("--xcomet-model", default=DEFAULT_XCOMET)
    parser.add_argument("--no-xcomet", action="store_true", help="skip XCOMET entirely")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--xcomet-batch-size", type=int, default=8)
    args = parser.parse_args()

    common.load_env()
    run_dir = args.run_dir
    instances = common.load_instances(common.resolve_instances_path(run_dir, args.instances))
    rows = list(common.latest_by_job(common.read_jsonl(run_dir / "translations.jsonl")).values())
    rows = common.scorable_rows(rows, instances or None)
    print(f"[reference] {len(rows)} scorable rows", flush=True)
    if not rows:
        return

    cache_path = run_dir / "reference_cache.jsonl"
    cache = {c["key"]: c for c in common.read_jsonl(cache_path)}
    triples: dict[str, dict] = {}
    row_key = {}
    for row in rows:
        src, hyp, ref = row["current_source"], row["hypothesis"], row["reference"]
        key = common.digest(src, hyp, ref)
        row_key[row["job_id"]] = key
        triples.setdefault(key, {"key": key, "src": src, "mt": hyp, "ref": ref})
    print(f"[reference] {len(triples)} unique (src, hyp, ref) triples, "
          f"{sum(k in cache for k in triples)} cached", flush=True)

    # COMET + lexical for triples not yet cached under this COMET model.
    todo = [t for k, t in triples.items()
            if k not in cache or cache[k].get("comet_model") != args.comet_model]
    if todo:
        scores = _isolated(run_comet, todo, args.comet_model, args.batch_size)
        for triple, score in zip(todo, scores):
            entry = cache.get(triple["key"], {"key": triple["key"], "src": triple["src"],
                                              "mt": triple["mt"], "ref": triple["ref"]})
            entry.update(comet=float(score), comet_model=args.comet_model,
                         **lexical(triple["mt"], triple["ref"]))
            cache[triple["key"]] = entry
            common.append_jsonl(cache_path, entry)

    # XCOMET for triples lacking it.
    status_path = run_dir / "xcomet_status.json"
    if args.no_xcomet:
        if not status_path.exists():
            status_path.write_text(json.dumps({"status": "skipped", "reason": "--no-xcomet"}))
    else:
        todo = [t for k, t in triples.items()
                if cache[k].get("xcomet_model") != args.xcomet_model]
        if todo:
            results = _isolated(run_xcomet, todo, args.xcomet_model,
                                args.xcomet_batch_size, status_path)
            if results is not None:
                for triple, result in zip(todo, results):
                    entry = cache[triple["key"]]
                    entry.update(result, xcomet_model=args.xcomet_model)
                    common.append_jsonl(cache_path, entry)
        else:
            print("[xcomet] all triples already scored", flush=True)

    out = []
    for row in rows:
        entry = cache[row_key[row["job_id"]]]
        out.append({
            "job_id": row["job_id"],
            "triple_key": entry["key"],
            "comet": entry.get("comet"),
            "comet_model": entry.get("comet_model"),
            "xcomet": entry.get("xcomet"),
            "xcomet_model": entry.get("xcomet_model"),
            "xcomet_error_spans": entry.get("xcomet_error_spans"),
            "chrf_pp": entry.get("chrf_pp"),
            "bleu": entry.get("bleu"),
        })
    common.write_jsonl(run_dir / "scores_reference.jsonl", out)
    # Compact the append-only cache: last entry per key.
    common.write_jsonl(cache_path, cache.values())
    print(f"[reference] wrote {len(out)} rows to {run_dir / 'scores_reference.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
