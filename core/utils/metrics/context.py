"""Dialogue-context metrics: Context-MQM, inconsistency rate and contrastives.

The expensive/context-sensitive judgement step is intentionally external to these
deterministic aggregators.  Store its versioned output under
``metric_inputs.context`` and the bench will preserve per-turn diagnostics.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean


def context_mqm_score(judgements) -> dict:
    """Aggregate context-conditioned MQM scores and optional no-context deltas."""
    scores = []
    baselines = []
    per_item = {}
    errors = {}
    for index, row in enumerate(judgements):
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        item_id = str(row.get("id", index))
        score = float(score)
        scores.append(score)
        per_item[item_id] = score
        baseline = row.get("score_without_context")
        if isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
            baselines.append(score - float(baseline))
        if row.get("errors"):
            errors[item_id] = list(row["errors"])
    if not scores:
        raise ValueError("no Context-MQM judgements")
    result = {"score": mean(scores), "n_scored": len(scores), "per_item": per_item}
    if baselines:
        result["mean_context_gain"] = mean(baselines)
        result["n_paired_without_context"] = len(baselines)
    if errors:
        result["errors"] = errors
    return result


def dialogue_inconsistency_rate(turns) -> dict:
    """Fraction of reviewed turns with one or more dialogue inconsistency."""
    eligible = inconsistent = 0
    counts = defaultdict(int)
    per_item = {}
    for index, row in enumerate(turns):
        if "inconsistencies" not in row:
            continue
        issues = row.get("inconsistencies") or []
        item_id = str(row.get("id", index))
        has_issue = bool(issues)
        eligible += 1
        inconsistent += int(has_issue)
        per_item[item_id] = has_issue
        for issue in issues:
            kind = issue.get("type", "unspecified") if isinstance(issue, dict) else str(issue)
            counts[str(kind)] += 1
    if not eligible:
        raise ValueError("no turns have dialogue inconsistency annotations")
    return {"error_rate": inconsistent / eligible, "inconsistent_turns": inconsistent,
            "eligible_turns": eligible, "by_type": dict(sorted(counts.items())),
            "per_item": per_item}


def context_contrastive_accuracy(examples) -> dict:
    """Accuracy choosing the contextual translation over controlled distractors."""
    correct = total = 0
    by_phenomenon = defaultdict(lambda: [0, 0])
    per_item = {}
    for index, row in enumerate(examples):
        selected = row.get("selected_correct")
        if not isinstance(selected, bool):
            right = row.get("correct_score")
            wrong = row.get("distractor_scores") or []
            if not isinstance(right, (int, float)) or not wrong:
                continue
            selected = float(right) > max(float(value) for value in wrong)
        item_id = str(row.get("id", index))
        phenomenon = str(row.get("phenomenon") or "unspecified")
        total += 1
        correct += int(selected)
        by_phenomenon[phenomenon][0] += int(selected)
        by_phenomenon[phenomenon][1] += 1
        per_item[item_id] = selected
    if not total:
        raise ValueError("no context contrastive examples")
    return {"accuracy": correct / total, "correct": correct, "total": total,
            "by_phenomenon": {key: {"accuracy": n / d, "correct": n, "total": d}
                              for key, (n, d) in sorted(by_phenomenon.items())},
            "per_item": per_item}


def corpus(items, **_) -> tuple[dict, dict]:
    judgements, turns, examples = [], [], []
    for item in items:
        block = item.metric_inputs.get("context") or {}
        if block.get("mqm") is not None:
            raw = block["mqm"]
            judgements.append({"id": item.id, **(raw if isinstance(raw, dict)
                                                   else {"score": raw})})
        if "inconsistencies" in block:
            turns.append({"id": item.id, "inconsistencies": block["inconsistencies"]})
        if block.get("contrastive") is not None:
            raw = block["contrastive"]
            examples.append({"id": item.id, **(raw if isinstance(raw, dict) else {})})

    axis, unavailable = {}, {}
    for name, function, rows in (
        ("context_mqm", context_mqm_score, judgements),
        ("dialogue_inconsistency_rate", dialogue_inconsistency_rate, turns),
        ("context_contrastive_accuracy", context_contrastive_accuracy, examples),
    ):
        try:
            axis[name] = function(rows)
        except ValueError as exc:
            unavailable[f"context.{name}"] = str(exc)
    return ({"context": axis} if axis else {}), unavailable


__all__ = ["context_mqm_score", "dialogue_inconsistency_rate",
           "context_contrastive_accuracy", "corpus"]
