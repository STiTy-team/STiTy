"""The six quality axes of `metric_inputs`, and which of their inputs are human gold."""
import copy

from .critical_information import gold_reference_spans


QUALITY_AXES = frozenset({"meaning", "critical_information", "fluency", "context",
                          "asr_robustness", "intent"})


def gold_inputs(metric_inputs) -> dict:
    source = copy.deepcopy(dict(metric_inputs or {}))
    result = {key: value for key, value in source.items() if key not in QUALITY_AXES}

    critical = source.get("critical_information") or {}
    if "reference_spans" in critical:
        gold = gold_reference_spans(critical)
        if gold or "annotation_source" not in critical:
            result["critical_information"] = {"reference_spans": gold}

    intent = {}
    for name, labels in (source.get("intent") or {}).items():
        if isinstance(labels, dict) and labels.get("reference") is not None:
            intent[name] = {"reference": copy.deepcopy(labels["reference"])}
    if intent:
        result["intent"] = intent
    return result


__all__ = ["QUALITY_AXES", "gold_inputs"]
