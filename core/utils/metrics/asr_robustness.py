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
    clean_levels = defaultdict(list)
    per_item = defaultdict(dict)
    for index, row in enumerate(pairs):
        clean = _quality_map(row.get("clean_quality"))
        noisy = _quality_map(row.get("noisy_quality"))
        item_id = str(row.get("id", index))
        for metric in clean.keys() & noisy.keys():
            raw = noisy[metric] - clean[metric] if metric in lower else clean[metric] - noisy[metric]
            grouped[metric].append(raw)
            denom = abs(clean[metric])
            clean_levels[metric].append(denom)
            per_item[metric][item_id] = {
                "absolute_drop": raw,
                "relative_drop": raw / denom if denom else 0.0,
            }
    if not grouped:
        raise ValueError("no paired clean_quality/noisy_quality values")
    return {metric: {
                "absolute_drop": mean(values),
                "relative_drop": (mean(values) / mean(clean_levels[metric])
                                  if mean(clean_levels[metric]) else 0.0),
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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def quality_noise_degradation(samples, *, lower_is_better=()) -> dict:
    """Linear degradation slope, normalised AUC and worst-noise-bucket quality.

    ``noise_level`` is the noise each sample actually has (e.g. WER from the clean
    transcript). Samples sharing a ``requested_noise_level`` form one bucket; without
    it, samples with the same ``noise_level`` do. Every statistic is taken over the
    bucket means -- the curve a reader would draw -- so the spread of individual
    sentences inside a bucket cannot leak into the AUC. The slope is the least-squares
    fit of those means weighted by bucket size, which equals the pooled fit whenever
    every sample in a bucket has the same noise.
    """
    lower = set(lower_is_better)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in samples:
        noise = row.get("noise_level", row.get("wer"))
        if not _is_number(noise):
            continue
        bucket = row.get("requested_noise_level")
        bucket = float(bucket) if _is_number(bucket) else float(noise)
        for metric, quality in _quality_map(row.get("quality")).items():
            grouped[metric][bucket].append((float(noise), quality))

    result = {}
    for metric, buckets in sorted(grouped.items()):
        curve = sorted(({"bucket": bucket,
                         "noise_level": mean(x for x, _ in points),
                         "quality": mean(y for _, y in points),
                         "n": len(points)} for bucket, points in buckets.items()),
                       key=lambda point: (point["noise_level"], point["bucket"]))
        if len(curve) < 2:
            continue
        weights = [point["n"] for point in curve]
        xs = [point["noise_level"] for point in curve]
        ys = [point["quality"] for point in curve]
        total = sum(weights)
        x_mean = sum(w * x for w, x in zip(weights, xs)) / total
        y_mean = sum(w * y for w, y in zip(weights, ys)) / total
        variance = sum(w * (x - x_mean) ** 2 for w, x in zip(weights, xs))
        if variance == 0:
            continue
        raw_slope = sum(w * (x - x_mean) * (y - y_mean)
                        for w, x, y in zip(weights, xs, ys)) / variance
        degradation_slope = -raw_slope if metric not in lower else raw_slope
        width = xs[-1] - xs[0]
        auc = sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2
                  for i in range(1, len(curve))) / width
        result[metric] = {"degradation_slope": degradation_slope,
                          "raw_quality_slope": raw_slope,
                          "quality_auc": auc, "worst_bucket_quality": ys[-1],
                          "worst_bucket_noise": xs[-1], "n_points": total,
                          "curve": curve, "lower_is_better": metric in lower}
    if not result:
        raise ValueError("degradation needs at least two noise levels with quality "
                         "for one metric")
    return result


def corpus(items, **_) -> tuple[dict, dict]:
    rows = []
    for item in items:
        block = item.metric_inputs.get("asr_robustness") or {}
        if block:
            rows.append({"id": item.id, **block})
    lower = {"metricx_24", "mqm_fluency_error_rate", "critical_fact_error_rate",
             "target_lm_pseudo_perplexity"}
    axis, unavailable = {}, {}
    try:
        axis["quality_drop"] = asr_induced_quality_drop(rows, lower_is_better=lower)
    except ValueError as exc:
        unavailable["asr_robustness.quality_drop"] = str(exc)
    try:
        axis["translation_invariance"] = translation_invariance_score(rows)
    except ValueError as exc:
        unavailable["asr_robustness.translation_invariance"] = str(exc)
    # An empty ASR output or translation stays in the averages above (it is the worst
    # case of robustness, not missing data) and is also counted on its own.
    flagged = [row for row in rows if "catastrophic" in row]
    if flagged:
        failed = [row["id"] for row in flagged if row["catastrophic"]]
        axis["catastrophic_failures"] = {"count": len(failed), "n_pairs": len(flagged),
                                         "rate": len(failed) / len(flagged),
                                         "items": failed}

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
