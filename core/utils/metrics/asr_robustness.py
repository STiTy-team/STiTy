"""Paired clean/noisy-ASR translation robustness metrics."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean


DEFAULT_INVARIANCE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _quality_map(value, default_name="quality") -> dict[str, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {default_name: float(value)}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {}


def asr_induced_quality_drop(pairs, *, lower_is_better=()) -> dict:
    """Mean clean-to-noisy degradation, direction-normalised so larger is worse."""
    lower = set(lower_is_better)
    grouped = defaultdict(list)
    per_item = defaultdict(dict)
    for index, row in enumerate(pairs):
        clean = _quality_map(row.get("clean_quality"))
        noisy = _quality_map(row.get("noisy_quality"))
        item_id = str(row.get("id", index))
        for metric in clean.keys() & noisy.keys():
            raw = noisy[metric] - clean[metric] if metric in lower else clean[metric] - noisy[metric]
            grouped[metric].append(raw)
            denom = abs(clean[metric])
            per_item[metric][item_id] = {
                "absolute_drop": raw,
                "relative_drop": raw / denom if denom else 0.0,
            }
    if not grouped:
        raise ValueError("no paired clean_quality/noisy_quality values")
    return {metric: {
                "absolute_drop": mean(values),
                "relative_drop": mean(v["relative_drop"] for v in per_item[metric].values()),
                "n_pairs": len(values), "lower_is_better": metric in lower,
                "per_item": per_item[metric],
            } for metric, values in sorted(grouped.items())}


def translation_invariance_score(pairs, *, scorer=None) -> dict:
    """Mean semantic similarity of clean/noisy translations.

    ``similarity`` may be precomputed by a fixed multilingual encoder.  Supplying
    ``scorer(clean, noisy)`` makes the protocol executable without coupling this
    module to a particular embedding model.
    """
    values = []
    per_item = {}
    for index, row in enumerate(pairs):
        score = row.get("similarity")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            if scorer is None or "clean_translation" not in row or "noisy_translation" not in row:
                continue
            score = scorer(row["clean_translation"], row["noisy_translation"])
        score = float(score)
        item_id = str(row.get("id", index))
        values.append(score)
        per_item[item_id] = score
    if not values:
        raise ValueError("no translation invariance similarities or scorer")
    return {"score": mean(values), "n_pairs": len(values), "per_item": per_item}


def multilingual_semantic_similarity(pairs, *, model_name=DEFAULT_INVARIANCE_MODEL,
                                     device=None, batch_size=32) -> dict:
    rows = list(pairs)
    usable = [row for row in rows
              if str(row.get("clean_translation") or "").strip()
              and str(row.get("noisy_translation") or "").strip()]
    if not usable:
        raise ValueError("no clean/noisy translation pairs")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    clean = [str(row["clean_translation"]) for row in usable]
    noisy = [str(row["noisy_translation"]) for row in usable]
    left = model.encode(clean, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    right = model.encode(noisy, batch_size=batch_size, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
    scores = (left * right).sum(axis=1).tolist()
    per_item = {str(row.get("id", index)): float(score)
                for index, (row, score) in enumerate(zip(usable, scores))}
    return {"score": mean(per_item.values()), "n_pairs": len(per_item),
            "per_item": per_item, "model": model_name}


def quality_noise_degradation(samples, *, lower_is_better=()) -> dict:
    """Linear degradation slope, normalised AUC and worst-noise-bucket quality."""
    lower = set(lower_is_better)
    grouped = defaultdict(list)
    for row in samples:
        noise = row.get("noise_level", row.get("wer"))
        if not isinstance(noise, (int, float)) or isinstance(noise, bool):
            continue
        for metric, quality in _quality_map(row.get("quality")).items():
            grouped[metric].append((float(noise), quality))
    if not any(len(points) >= 2 for points in grouped.values()):
        raise ValueError("degradation needs at least two noise/quality points per metric")

    result = {}
    for metric, points in sorted(grouped.items()):
        if len(points) < 2:
            continue
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_mean, y_mean = mean(xs), mean(ys)
        variance = sum((x - x_mean) ** 2 for x in xs)
        if variance == 0:
            continue
        raw_slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / variance
        degradation_slope = -raw_slope if metric not in lower else raw_slope
        width = xs[-1] - xs[0]
        auc = (sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2
                   for i in range(1, len(points))) / width) if width else ys[-1]
        max_noise = xs[-1]
        worst_bucket = mean(y for x, y in points if x == max_noise)
        result[metric] = {"degradation_slope": degradation_slope,
                          "raw_quality_slope": raw_slope,
                          "quality_auc": auc, "worst_bucket_quality": worst_bucket,
                          "worst_bucket_noise": max_noise, "n_points": len(points),
                          "lower_is_better": metric in lower}
    if not result:
        raise ValueError("noise levels have zero variance")
    return result


def corpus(items, **_) -> tuple[dict, dict]:
    rows = []
    for item in items:
        block = item.metric_inputs.get("asr_robustness") or {}
        if block:
            rows.append({"id": item.id, **block})
    lower = {"metricx_24", "mqm_error_rate", "critical_fact_error_rate"}
    axis, unavailable = {}, {}
    try:
        axis["quality_drop"] = asr_induced_quality_drop(rows, lower_is_better=lower)
    except ValueError as exc:
        unavailable["asr_robustness.quality_drop"] = str(exc)
    try:
        axis["translation_invariance"] = translation_invariance_score(rows)
    except ValueError as exc:
        unavailable["asr_robustness.translation_invariance"] = str(exc)

    degradation_rows = []
    for row in rows:
        if "quality_by_noise" in row:
            for point in row["quality_by_noise"] or []:
                degradation_rows.append(point)
        elif "noise_level" in row and "noisy_quality" in row:
            degradation_rows.append({"noise_level": row["noise_level"],
                                     "quality": row["noisy_quality"]})
    try:
        axis["quality_noise_degradation"] = quality_noise_degradation(
            degradation_rows, lower_is_better=lower)
    except ValueError as exc:
        unavailable["asr_robustness.quality_noise_degradation"] = str(exc)
    return ({"asr_robustness": axis} if axis else {}), unavailable


__all__ = ["asr_induced_quality_drop", "translation_invariance_score",
           "multilingual_semantic_similarity", "quality_noise_degradation",
           "DEFAULT_INVARIANCE_MODEL", "corpus"]
