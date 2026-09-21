"""Target-language spoken fluency metrics.

The judge and MQM metrics aggregate reviewed/externally-produced annotations.
Pseudo-perplexity has a complete masked-LM implementation and is activated in the
bench by precomputed values or ``STITY_FLUENCY_LM``.
"""
from __future__ import annotations

import math
import os
from statistics import mean


DEFAULT_SEVERITY_WEIGHTS = {"minor": 1.0, "major": 5.0, "critical": 25.0}


def spoken_fluency_judge_score(judgements) -> dict:
    """Mean fixed-rubric 1..5 score, retaining per-item rationales."""
    rows = list(judgements)
    scores = []
    per_item = {}
    reasons = {}
    for index, row in enumerate(rows):
        raw = row.get("score") if isinstance(row, dict) else row
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        score = float(raw)
        if not 1.0 <= score <= 5.0:
            raise ValueError("spoken fluency judge scores must be in [1, 5]")
        item_id = str(row.get("id", index)) if isinstance(row, dict) else str(index)
        scores.append(score)
        per_item[item_id] = score
        if isinstance(row, dict) and row.get("reason"):
            reasons[item_id] = str(row["reason"])
    if not scores:
        raise ValueError("no spoken fluency judge scores")
    result = {"score": mean(scores), "n_scored": len(scores), "per_item": per_item}
    if reasons:
        result["reasons"] = reasons
    return result


def mqm_fluency_error_rate(rows, *, severity_weights=None) -> dict:
    """Severity-weighted fluency/style errors per target-language token."""
    weights = dict(DEFAULT_SEVERITY_WEIGHTS)
    weights.update(severity_weights or {})
    weighted_errors = 0.0
    tokens = 0
    counts = {key: 0 for key in weights}
    per_item = {}
    for index, row in enumerate(rows):
        n_tokens = int(row.get("target_token_count") or 0)
        if n_tokens <= 0:
            continue
        item_weight = 0.0
        for error in row.get("errors") or []:
            severity = str(error.get("severity") or "minor").lower()
            weight = float(weights.get(severity, weights["minor"]))
            item_weight += weight
            counts[severity] = counts.get(severity, 0) + 1
        item_id = str(row.get("id", index))
        per_item[item_id] = item_weight / n_tokens
        weighted_errors += item_weight
        tokens += n_tokens
    if not tokens:
        raise ValueError("no MQM row has target_token_count")
    return {"error_rate": weighted_errors / tokens,
            "weighted_errors": weighted_errors, "target_tokens": tokens,
            "severity_counts": counts, "severity_weights": weights,
            "per_item": per_item}


def _masked_lm(model_name: str, device: str):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.mask_token_id is None:
        raise ValueError(f"{model_name} has no mask token; use a masked LM")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def target_lm_pseudo_perplexity(texts, *, model_name: str,
                                device: str | None = None,
                                max_length: int = 512,
                                batch_size: int = 16) -> dict:
    """Masked-token pseudo-perplexity (lower is better).

    Each non-special token is masked once and its negative log-probability is
    accumulated.  This is intentionally not causal-LM perplexity.
    """
    import torch

    texts = list(texts)
    if not texts:
        raise ValueError("pseudo-perplexity needs at least one text")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _masked_lm(model_name, device)
    total_loss = 0.0
    total_tokens = 0
    per_item = {}
    for index, raw in enumerate(texts):
        item_id = str(raw.get("id", index)) if isinstance(raw, dict) else str(index)
        text = str(raw.get("text") or "") if isinstance(raw, dict) else str(raw)
        encoded = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length, return_special_tokens_mask=True)
        input_ids = encoded["input_ids"][0]
        special = encoded["special_tokens_mask"][0].bool()
        positions = (~special).nonzero(as_tuple=False).flatten().tolist()
        if not positions:
            continue
        losses = []
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start:start + batch_size]
            masked = input_ids.repeat(len(batch_positions), 1)
            labels = []
            for batch_index, position in enumerate(batch_positions):
                labels.append(int(masked[batch_index, position]))
                masked[batch_index, position] = tokenizer.mask_token_id
            attention = encoded["attention_mask"].repeat(
                len(batch_positions), 1).to(device)
            with torch.inference_mode():
                logits = model(input_ids=masked.to(device), attention_mask=attention).logits
                selected = logits[
                    torch.arange(len(batch_positions), device=logits.device),
                    torch.tensor(batch_positions, device=logits.device),
                ]
                log_probs = torch.log_softmax(selected, dim=-1)
                batch_losses = -log_probs[
                    torch.arange(len(labels), device=logits.device),
                    torch.tensor(labels, device=logits.device),
                ]
            losses.extend(float(value) for value in batch_losses.detach().cpu())
        sentence_loss = sum(losses) / len(losses)
        per_item[item_id] = math.exp(sentence_loss)
        total_loss += sum(losses)
        total_tokens += len(losses)
    if not total_tokens:
        raise ValueError("no scoreable target-language tokens")
    return {"pseudo_perplexity": math.exp(total_loss / total_tokens),
            "model": model_name, "n_tokens": total_tokens,
            "n_scored": len(per_item), "per_item": per_item}


def corpus(items, **_) -> tuple[dict, dict]:
    items = list(items)
    blocks = [(item, item.metric_inputs.get("fluency") or {}) for item in items]
    axis, unavailable = {}, {}

    judgements = []
    mqm_rows = []
    ppls = {}
    ppl_targets = {}
    ppl_models = {}
    for item, block in blocks:
        if block.get("judge") is not None:
            judge = block["judge"]
            judgements.append({"id": item.id, **(judge if isinstance(judge, dict)
                                                  else {"score": judge})})
        if "mqm_errors" in block:
            token_count = block.get("target_token_count")
            if token_count is None:
                token_count = len(item.hypothesis_translation.split())
            mqm_rows.append({"id": item.id, "errors": block.get("mqm_errors") or [],
                             "target_token_count": token_count})
        raw_ppl = block.get("target_lm_pseudo_perplexity")
        if isinstance(raw_ppl, (int, float)) and not isinstance(raw_ppl, bool):
            ppls[item.id] = float(raw_ppl)
            ppl_targets[item.id] = str(block.get("target_lang") or "unknown")
            if block.get("target_lm_model"):
                ppl_models[item.id] = str(block["target_lm_model"])

    try:
        axis["spoken_fluency_judge"] = spoken_fluency_judge_score(judgements)
    except ValueError as exc:
        unavailable["fluency.spoken_fluency_judge"] = str(exc)
    try:
        axis["mqm_fluency_error_rate"] = mqm_fluency_error_rate(mqm_rows)
    except ValueError as exc:
        unavailable["fluency.mqm_fluency_error_rate"] = str(exc)

    if ppls:
        grouped = {}
        for item_id, value in ppls.items():
            grouped.setdefault(ppl_targets[item_id], []).append(value)
        result = {"n_scored": len(ppls), "per_item": ppls,
                  "source": "metric_inputs",
                  "by_target": {lang: {"pseudo_perplexity": mean(values),
                                       "n_scored": len(values)}
                                for lang, values in sorted(grouped.items())}}
        if len(grouped) == 1:
            result["pseudo_perplexity"] = mean(ppls.values())
        if ppl_models:
            result["models"] = sorted(set(ppl_models.values()))
        axis["target_lm_pseudo_perplexity"] = result
    else:
        model_name = os.environ.get("STITY_FLUENCY_LM")
        texts = [{"id": item.id, "text": item.hypothesis_translation}
                 for item in items if item.hypothesis_translation]
        if not model_name:
            unavailable["fluency.target_lm_pseudo_perplexity"] = (
                "no precomputed fluency value and STITY_FLUENCY_LM is not set")
        elif not texts:
            unavailable["fluency.target_lm_pseudo_perplexity"] = "no candidate translations"
        else:
            try:
                axis["target_lm_pseudo_perplexity"] = target_lm_pseudo_perplexity(
                    texts, model_name=model_name)
            except Exception as exc:  # noqa: BLE001 - optional model failures are diagnostics
                unavailable["fluency.target_lm_pseudo_perplexity"] = str(exc)
            finally:
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
    return ({"fluency": axis} if axis else {}), unavailable


__all__ = ["spoken_fluency_judge_score", "mqm_fluency_error_rate",
           "target_lm_pseudo_perplexity", "corpus"]
