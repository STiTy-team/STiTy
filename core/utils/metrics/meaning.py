"""Meaning preservation: XCOMET, MetricX-24 and chrF++.

The learned metrics are deliberately lazy.  A normal bench run uses a local
checkpoint named by ``STITY_XCOMET_MODEL`` / ``STITY_METRICX_MODEL`` or consumes
precomputed per-item values from ``metric_inputs.meaning``.  It never downloads a
multi-gigabyte checkpoint merely because a summary is being written.  The public
scorers can still be called directly by an offline/GPU scoring job.
"""
from __future__ import annotations

import os
from statistics import mean


DEFAULT_XCOMET_MODEL = "Unbabel/XCOMET-XL"
DEFAULT_METRICX_MODEL = "google/metricx-24-hybrid-large-v2p6"
DEFAULT_METRICX_TOKENIZER = "google/mt5-xl"


def _xcomet_model(model_name: str):
    from comet import download_model, load_from_checkpoint

    return load_from_checkpoint(download_model(model_name))


def xcomet_score(samples, *, model_name: str = DEFAULT_XCOMET_MODEL,
                  batch_size: int = 8, gpus: int | None = None) -> dict:
    """Run the official Unbabel XCOMET implementation.

    ``samples`` contain ``src``, ``mt`` and ``ref``.  The result retains the
    sentence scores and MQM-like error spans because a corpus mean alone cannot
    diagnose omissions or additions.
    """
    rows = list(samples)
    if not rows:
        raise ValueError("XCOMET needs at least one sample")
    if gpus is None:
        try:
            import torch
            gpus = 1 if torch.cuda.is_available() else 0
        except ImportError:
            gpus = 0
    output = _xcomet_model(model_name).predict(
        [{"src": r["src"], "mt": r["mt"], "ref": r["ref"]} for r in rows],
        batch_size=batch_size,
        gpus=gpus,
    )
    scores = [float(v) for v in output.scores]
    metadata = getattr(output, "metadata", None)
    spans = getattr(metadata, "error_spans", None) if metadata is not None else None
    result = {
        "score": float(output.system_score),
        "model": model_name,
        "n_scored": len(scores),
        "per_item": {str(r.get("id", i)): s for i, (r, s) in enumerate(zip(rows, scores))},
    }
    if spans is not None:
        result["error_spans"] = {
            str(r.get("id", i)): value
            for i, (r, value) in enumerate(zip(rows, spans))
        }
    return result


def _metricx_runtime(model_name: str, tokenizer_name: str, device: str):
    # ``metricx24`` is the package from google-research/metricx.  Importing its
    # model class, rather than approximating it with AutoModel, is important: the
    # regression head reads vocabulary logit 250089 exactly as the official code.
    import torch
    from metricx24.models import MT5ForRegression
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = MT5ForRegression.from_pretrained(model_name, torch_dtype="auto")
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def metricx24_score(samples, *, model_name: str = DEFAULT_METRICX_MODEL,
                    tokenizer_name: str = DEFAULT_METRICX_TOKENIZER,
                    max_input_length: int = 1536, batch_size: int = 1,
                    device: str | None = None) -> dict:
    """Run Google's official MetricX-24 regression model (lower is better)."""
    rows = list(samples)
    if not rows:
        raise ValueError("MetricX-24 needs at least one sample")
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _metricx_runtime(model_name, tokenizer_name, device)
    predictions: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [
            f"source: {r['src']} candidate: {r['mt']} reference: {r['ref']}"
            for r in batch
        ]
        encoded = tokenizer(texts, max_length=max_input_length, truncation=True,
                            padding=True, return_tensors="pt")
        # The official predictor removes the tokenizer's terminal EOS before
        # inference.  Preserve padding and remove the last attended token per row.
        for index, length in enumerate(encoded["attention_mask"].sum(dim=1).tolist()):
            if length:
                encoded["attention_mask"][index, int(length) - 1] = 0
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
        predictions.extend(float(v) for v in output.predictions.detach().cpu())
    return {
        "score": mean(predictions),
        "model": model_name,
        "tokenizer": tokenizer_name,
        "max_input_length": max_input_length,
        "n_scored": len(predictions),
        "per_item": {str(r.get("id", i)): s
                     for i, (r, s) in enumerate(zip(rows, predictions))},
    }


def chrfpp_score(pairs) -> dict:
    """Corpus chrF++ using sacreBLEU's reproducible implementation."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("chrF++ needs at least one hypothesis/reference pair")
    from sacrebleu.metrics import CHRF

    metric = CHRF(char_order=6, word_order=2, beta=2)
    hyps = [p[0] for p in pairs]
    refs = [p[1] for p in pairs]
    score = metric.corpus_score(hyps, [refs])
    per_item = [metric.sentence_score(h, [r]).score for h, r in pairs]
    return {
        "score": float(score.score),
        "signature": str(metric.get_signature()),
        "n_scored": len(pairs),
        "per_item_scores": per_item,
    }


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
        return float(value["score"])
    return None


def _precomputed(items, name: str) -> dict | None:
    values = []
    per_item = {}
    spans = {}
    for item in items:
        raw = (item.metric_inputs.get("meaning") or {}).get(name)
        score = _number(raw)
        if score is None:
            continue
        values.append(score)
        per_item[item.id] = score
        if isinstance(raw, dict) and raw.get("error_spans") is not None:
            spans[item.id] = raw["error_spans"]
    if not values:
        return None
    result = {"score": mean(values), "n_scored": len(values), "per_item": per_item,
              "source": "metric_inputs"}
    if spans:
        result["error_spans"] = spans
    return result


def _release_accelerator() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def corpus(items, *, languages) -> tuple[dict, dict]:
    """Return the meaning scorecard and structured unavailability reasons."""
    items = list(items)
    samples = []
    pair_ids = []
    for item in items:
        target = languages.expected_target(item.src_lang)
        ref = item.reference_translation(target)
        if item.hypothesis_translation and ref:
            samples.append({"id": item.id, "src": item.hypothesis,
                            "mt": item.hypothesis_translation, "ref": ref})
            pair_ids.append(item.id)

    axis, unavailable = {}, {}
    if samples:
        try:
            result = chrfpp_score([(s["mt"], s["ref"]) for s in samples])
            result["per_item"] = dict(zip(pair_ids, result.pop("per_item_scores")))
            axis["chrfpp"] = result
        except Exception as exc:  # noqa: BLE001 - optional scorer failures are diagnostics
            unavailable["meaning.chrfpp"] = f"sacrebleu unavailable: {exc}"
    else:
        unavailable["meaning.chrfpp"] = "no candidate/reference translation pairs"

    learned = (("xcomet", "STITY_XCOMET_MODEL", xcomet_score),
               ("metricx_24", "STITY_METRICX_MODEL", metricx24_score))
    for name, env_name, scorer in learned:
        ready = _precomputed(items, name)
        model_name = os.environ.get(env_name)
        if ready is not None:
            axis[name] = ready
        elif not samples:
            unavailable[f"meaning.{name}"] = "no candidate/reference translation pairs"
        elif not model_name:
            unavailable[f"meaning.{name}"] = (
                f"no precomputed metric_inputs.meaning.{name} and {env_name} is not set")
        else:
            try:
                axis[name] = scorer(samples, model_name=model_name)
            except Exception as exc:  # noqa: BLE001 - model/cache/auth errors vary by backend
                unavailable[f"meaning.{name}"] = f"scorer unavailable: {exc}"
            finally:
                # XCOMET and MetricX are both multi-GB models. Keep them
                # sequential so a single-GPU evaluator does not retain both.
                _release_accelerator()
    return ({"meaning": axis} if axis else {}), unavailable


__all__ = ["xcomet_score", "metricx24_score", "chrfpp_score", "corpus"]
