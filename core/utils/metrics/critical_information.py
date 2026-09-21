"""Critical-information preservation metrics.

Extraction and canonicalisation are dataset responsibilities: an evaluator cannot
reliably infer that ``3 PM`` and ``15:00`` denote the same value without locale and
date context.  These functions score reviewed spans carrying ``type`` and a
``canonical_value`` (plus optional ``accepted_values``).
"""
from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict


VALUE_TYPES = {"number", "ordinal", "date", "time", "money", "unit"}


def _norm(value) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _forms(span: dict) -> set[str]:
    values = [span.get("canonical_value"), span.get("text")]
    values.extend(span.get("accepted_values") or [])
    return {_norm(v) for v in values if _norm(v)}


def _matches(expected: dict, predicted: dict) -> bool:
    return (_norm(expected.get("type")) == _norm(predicted.get("type"))
            and bool(_forms(expected) & _forms(predicted)))


def _match_counts(expected, predicted, *, allowed_types=None) -> tuple[int, int, int, dict]:
    refs = [dict(v) for v in expected
            if allowed_types is None or _norm(v.get("type")) in allowed_types]
    hyps = [dict(v) for v in predicted
            if allowed_types is None or _norm(v.get("type")) in allowed_types]
    used = set()
    correct = 0
    by_type = defaultdict(lambda: Counter(total=0, correct=0))
    for ref in refs:
        kind = _norm(ref.get("type")) or "unknown"
        by_type[kind]["total"] += 1
        found = next((i for i, hyp in enumerate(hyps)
                      if i not in used and _matches(ref, hyp)), None)
        if found is not None:
            used.add(found)
            correct += 1
            by_type[kind]["correct"] += 1
    detail = {kind: {"correct": counts["correct"], "total": counts["total"],
                     "accuracy": counts["correct"] / counts["total"]}
              for kind, counts in sorted(by_type.items()) if counts["total"]}
    return correct, len(refs), len(hyps), detail


def normalized_value_accuracy(expected, predicted) -> dict:
    """Accuracy after dataset-provided canonical value normalisation."""
    correct, total, _, by_type = _match_counts(
        expected, predicted, allowed_types=VALUE_TYPES)
    if not total:
        raise ValueError("no normalized value annotations")
    return {"accuracy": correct / total, "correct": correct, "total": total,
            "by_type": by_type}


def critical_span_f1(expected, predicted) -> dict:
    """Micro precision/recall/F1 over reviewed critical spans."""
    true_positive, n_ref, n_hyp, _ = _match_counts(expected, predicted)
    precision = true_positive / n_hyp if n_hyp else (1.0 if not n_ref else 0.0)
    recall = true_positive / n_ref if n_ref else (1.0 if not n_hyp else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": true_positive, "predicted": n_hyp, "reference": n_ref}


def critical_fact_error_rate(utterances) -> dict:
    """Fraction of critical-information-bearing utterances with any span error."""
    eligible = errors = 0
    per_item = {}
    for row in utterances:
        expected = row.get("reference_spans") or []
        if not expected:
            continue
        predicted = row.get("candidate_spans") or []
        match = critical_span_f1(expected, predicted)
        has_error = not (match["true_positive"] == match["reference"] == match["predicted"])
        eligible += 1
        errors += int(has_error)
        per_item[str(row.get("id") or eligible)] = has_error
    if not eligible:
        raise ValueError("no utterance contains reviewed critical spans")
    return {"error_rate": errors / eligible, "error_items": errors,
            "eligible_items": eligible, "per_item": per_item}


def corpus(items, **_) -> tuple[dict, dict]:
    rows = []
    reference_items = 0
    for item in items:
        block = item.metric_inputs.get("critical_information") or {}
        if "reference_spans" in block:
            reference_items += 1
        # Missing means "the new candidate has not been annotated"; an explicit
        # empty list means "annotation ran and found no spans". Conflating the two
        # would turn every translation-only replay into a false critical failure.
        if "reference_spans" in block and "candidate_spans" in block:
            rows.append({"id": item.id,
                         "reference_spans": block.get("reference_spans") or [],
                         "candidate_spans": block.get("candidate_spans") or []})
    if not rows:
        reason = ("missing metric_inputs.critical_information.candidate_spans"
                  if reference_items else
                  "missing metric_inputs.critical_information.reference_spans")
        return {}, {name: reason for name in (
            "critical_information.normalized_value_accuracy",
            "critical_information.critical_span_f1",
            "critical_information.critical_fact_error_rate")}

    values, unavailable = {}, {}
    # Match within an utterance. Flattening first would let the same value in a
    # different turn conceal a local omission.
    value_correct = value_total = 0
    value_types = defaultdict(lambda: Counter(total=0, correct=0))
    span_tp = span_ref = span_hyp = 0
    for row in rows:
        correct, total, _, detail = _match_counts(
            row["reference_spans"], row["candidate_spans"], allowed_types=VALUE_TYPES)
        value_correct += correct
        value_total += total
        for kind, cell in detail.items():
            value_types[kind]["correct"] += cell["correct"]
            value_types[kind]["total"] += cell["total"]
        tp, n_ref, n_hyp, _ = _match_counts(
            row["reference_spans"], row["candidate_spans"])
        span_tp += tp
        span_ref += n_ref
        span_hyp += n_hyp
    if value_total:
        values["normalized_value_accuracy"] = {
            "accuracy": value_correct / value_total,
            "correct": value_correct, "total": value_total,
            "n_annotated_items": len(rows),
            "n_missing_candidate_annotations": reference_items - len(rows),
            "by_type": {kind: {"correct": cell["correct"], "total": cell["total"],
                               "accuracy": cell["correct"] / cell["total"]}
                        for kind, cell in sorted(value_types.items())},
        }
    else:
        unavailable["critical_information.normalized_value_accuracy"] = (
            "no normalized value annotations")
    precision = span_tp / span_hyp if span_hyp else (1.0 if not span_ref else 0.0)
    recall = span_tp / span_ref if span_ref else (1.0 if not span_hyp else 0.0)
    values["critical_span_f1"] = {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positive": span_tp, "predicted": span_hyp, "reference": span_ref,
        "n_annotated_items": len(rows),
        "n_missing_candidate_annotations": reference_items - len(rows),
    }
    try:
        values["critical_fact_error_rate"] = critical_fact_error_rate(rows)
        values["critical_fact_error_rate"]["n_missing_candidate_annotations"] = (
            reference_items - len(rows))
    except ValueError as exc:
        unavailable["critical_information.critical_fact_error_rate"] = str(exc)
    return ({"critical_information": values} if values else {}), unavailable


__all__ = ["normalized_value_accuracy", "critical_span_f1",
           "critical_fact_error_rate", "corpus"]
