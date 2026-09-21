"""Speaker-intent preservation metrics."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean


def _label(value) -> str:
    return str(value or "").strip().lower()


def polarity_preservation_accuracy(rows) -> dict:
    """Polarity accuracy and the critical positive/negative reversal rate."""
    correct = reversals = total = 0
    confusion = Counter()
    per_item = {}
    for index, row in enumerate(rows):
        reference = _label(row.get("reference"))
        candidate = _label(row.get("candidate"))
        if not reference or not candidate:
            continue
        item_id = str(row.get("id", index))
        is_correct = reference == candidate
        is_reversal = {reference, candidate} == {"positive", "negative"}
        total += 1
        correct += int(is_correct)
        reversals += int(is_reversal)
        confusion[f"{reference}->{candidate}"] += 1
        per_item[item_id] = {"correct": is_correct, "reversal": is_reversal}
    if not total:
        raise ValueError("no paired polarity labels")
    return {"accuracy": correct / total, "polarity_reversal_rate": reversals / total,
            "correct": correct, "reversals": reversals, "total": total,
            "confusion": dict(sorted(confusion.items())), "per_item": per_item}


def _labels(value) -> set[str]:
    if isinstance(value, str):
        return {_label(value)} if _label(value) else set()
    if isinstance(value, (list, tuple, set)):
        return {_label(v) for v in value if _label(v)}
    return set()


def speech_act_macro_f1(rows) -> dict:
    """Macro-F1 for single- or multi-label speech acts."""
    counts = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    total = 0
    exact = 0
    for row in rows:
        reference = _labels(row.get("reference"))
        candidate = _labels(row.get("candidate"))
        if not reference and not candidate:
            continue
        total += 1
        exact += int(reference == candidate)
        for label in reference | candidate:
            if label in reference and label in candidate:
                counts[label]["tp"] += 1
            elif label in candidate:
                counts[label]["fp"] += 1
            else:
                counts[label]["fn"] += 1
    if not total or not counts:
        raise ValueError("no paired speech-act labels")
    per_class = {}
    f1s = []
    for label, c in sorted(counts.items()):
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": c["tp"] + c["fn"]}
    return {"macro_f1": mean(f1s), "exact_match_accuracy": exact / total,
            "n_items": total, "per_class": per_class}


EPISTEMIC_RANK = {"speculative": 0, "possible": 1, "probable": 2, "certain": 3}


def modality_stance_preservation_accuracy(rows) -> dict:
    """Exact accuracy plus ordinal distance for epistemic-strength labels."""
    correct = total = 0
    distances = []
    confusion = Counter()
    per_item = {}
    for index, row in enumerate(rows):
        reference = _label(row.get("reference"))
        candidate = _label(row.get("candidate"))
        if not reference or not candidate:
            continue
        item_id = str(row.get("id", index))
        is_correct = reference == candidate
        total += 1
        correct += int(is_correct)
        confusion[f"{reference}->{candidate}"] += 1
        cell = {"correct": is_correct}
        if reference in EPISTEMIC_RANK and candidate in EPISTEMIC_RANK:
            distance = abs(EPISTEMIC_RANK[reference] - EPISTEMIC_RANK[candidate])
            distances.append(distance)
            cell["ordinal_distance"] = distance
        per_item[item_id] = cell
    if not total:
        raise ValueError("no paired modality/stance labels")
    result = {"accuracy": correct / total, "correct": correct, "total": total,
              "confusion": dict(sorted(confusion.items())), "per_item": per_item}
    if distances:
        result["mean_ordinal_distance"] = mean(distances)
        result["n_ordinal_pairs"] = len(distances)
    return result


def corpus(items, **_) -> tuple[dict, dict]:
    polarity, acts, modality = [], [], []
    for item in items:
        block = item.metric_inputs.get("intent") or {}
        for key, target in (("polarity", polarity), ("speech_act", acts),
                            ("modality", modality)):
            raw = block.get(key)
            if isinstance(raw, dict):
                target.append({"id": item.id, **raw})
    axis, unavailable = {}, {}
    for name, function, rows in (
        ("polarity_preservation", polarity_preservation_accuracy, polarity),
        ("speech_act_macro_f1", speech_act_macro_f1, acts),
        ("modality_stance_preservation", modality_stance_preservation_accuracy, modality),
    ):
        try:
            axis[name] = function(rows)
        except ValueError as exc:
            unavailable[f"intent.{name}"] = str(exc)
    return ({"intent": axis} if axis else {}), unavailable


__all__ = ["polarity_preservation_accuracy", "speech_act_macro_f1",
           "modality_stance_preservation_accuracy", "corpus"]
