"""Metric aggregation and the pre-flight requirement check.

Both metric libraries return None when their dependency is missing -- jiwer and
sacrebleu are not installed in this venv today -- so a WER run currently emits
{"wer": null} and looks like it succeeded. Every metric therefore declares what it
needs, and `check_requirements` runs before the model loads so a missing reference
or package fails at second zero instead of after hours on the GPU.
"""
from dataclasses import dataclass

from ..errors import BenchConfigError
from . import asr, commit, latency, routing, translation

NEEDS_TRANSCRIPT = "transcript"
NEEDS_TRANSLATION = "translation_reference"
NEEDS_SACREBLEU = "sacrebleu"
NEEDS_REALTIME = "realtime_pacing"


@dataclass(frozen=True)
class MetricSpec:
    requires: frozenset
    keys: tuple


SPECS = {
    "wer": MetricSpec(frozenset({NEEDS_TRANSCRIPT}),
                      ("wer", "wer_scored_only", "n_empty_hypothesis")),
    "cer": MetricSpec(frozenset({NEEDS_TRANSCRIPT}), ("cer",)),
    "fsl": MetricSpec(frozenset({NEEDS_REALTIME}),
                      ("avg_fsl_sec", "avg_fsl_normalized_sec", "n_seg_with_fsl")),
    "laal": MetricSpec(frozenset({NEEDS_TRANSLATION, NEEDS_REALTIME}),
                       ("laal_ms", "laal_ca_ms", "laal_uncapped_ms")),
    "bleu": MetricSpec(frozenset({NEEDS_TRANSLATION, NEEDS_SACREBLEU}),
                       ("bleu", "bleu_all", "bleu_by_target")),
    "commit": MetricSpec(frozenset(), ("commit_stats", "segments_per_item")),
    "routing": MetricSpec(frozenset(),
                          ("lang_detect_accuracy", "route_accuracy", "n_segments_routed",
                           "n_misrouted", "n_direction_refix", "confusion")),
}

# Aggregate key -> the metric that owns it. Makes "every null has a reason in
# diagnostics" mechanically checkable instead of a naming convention.
KEY_OWNER = {key: name for name, spec in SPECS.items() for key in spec.keys}


def check_requirements(metric_names, *, dataset, languages, realtime: bool) -> None:
    """Raises BenchConfigError naming every unmet requirement. Runs before the
    model loads.

    All problems are collected and reported together rather than one per run --
    otherwise a missing package hides a missing reference and you discover them
    one failed launch at a time.
    """
    problems: list[str] = []
    for name in metric_names:
        spec = SPECS[name]
        for requirement in sorted(spec.requires):
            if requirement == NEEDS_TRANSCRIPT and not dataset.has_transcript:
                problems.append(
                    f"metric {name!r} needs transcript references, but dataset "
                    f"{dataset.name!r} declares provides.transcript: false"
                )
            elif requirement == NEEDS_TRANSLATION:
                if not dataset.translation_langs:
                    problems.append(
                        f"metric {name!r} needs target-language references, but dataset "
                        f"{dataset.name!r} declares provides.translations: []"
                    )
                else:
                    # Only the directions this dataset's audio can actually
                    # trigger. languages.target_langs lists both ends of a pair,
                    # but an English-only corpus never exercises the ko->en leg,
                    # so requiring an English reference would block a scorable run.
                    needed = languages.targets_for(getattr(dataset, "languages", []))
                    missing = [t for t in needed if t not in dataset.translation_langs]
                    if missing:
                        problems.append(
                            f"metric {name!r}: dataset {dataset.name!r} has references "
                            f"for {dataset.translation_langs} but this run translates "
                            f"{getattr(dataset, 'languages', [])} into {needed} "
                            f"(missing: {missing})"
                        )
            elif requirement == NEEDS_REALTIME and not realtime:
                problems.append(
                    f"metric {name!r} needs real-time pacing (pacing.send_interval_ms > 0)"
                )
            elif requirement == NEEDS_SACREBLEU and not translation.available():
                problems.append(
                    f"metric {name!r} needs sacrebleu: pip install 'sacrebleu[ja,ko]'")
    if problems:
        raise BenchConfigError(
            "the requested metrics cannot be computed:\n  - " + "\n  - ".join(problems))


def _bleu_pairs(rows, languages, *, only_routed: bool):
    by_target: dict[str, list] = {}
    for row in rows:
        for seg in row.get("segments") or []:
            hyp = (seg.get("translation") or "").strip()
            target = (seg.get("target_lang") or "").strip().lower()
            if not target:
                continue
            if only_routed and not routing.is_correctly_routed(row, seg, languages):
                continue
            ref = (row.get("reference_translations") or {}).get(target, "")
            if not ref:
                continue
            by_target.setdefault(target, []).append((hyp, ref))
    return by_target


def run_metrics(rows, cfg) -> tuple[dict, dict]:
    """Returns (aggregate, diagnostics). A null in aggregate must have a reason
    in diagnostics.uncomputable -- otherwise it is a bug."""
    requested = list(cfg.metrics)
    aggregate: dict = {}
    uncomputable: dict = {}
    scored = [r for r in rows if r.get("status") == "ok"]
    segments = [s for r in scored for s in (r.get("segments") or [])]

    if "wer" in requested:
        aggregate["wer"] = asr.corpus_wer(scored)
        aggregate["wer_scored_only"] = asr.wer_scored_only(scored)
        aggregate["n_empty_hypothesis"] = asr.count_empty_hypotheses(scored)
        if aggregate["wer"] is None:
            uncomputable["wer"] = "no row had a non-empty transcript reference"

    if "cer" in requested:
        aggregate["cer"] = asr.corpus_cer(scored)
        if aggregate["cer"] is None:
            uncomputable["cer"] = "no row had a non-empty transcript reference"

    if "fsl" in requested:
        aggregate.update(latency.fsl_stats(segments,
                                           min_silence_ms=cfg.stity.vad.min_silence_ms))
        if aggregate.get("avg_fsl_sec") is None:
            uncomputable["fsl"] = "no segment carried fsl_sec"

    if "laal" in requested:
        per_item = [r["laal"] for r in scored if r.get("laal")]
        for key in ("laal_ms", "laal_ca_ms", "laal_uncapped_ms"):
            values = [d[key] for d in per_item if d.get(key) is not None]
            aggregate[key] = (sum(values) / len(values)) if values else None
        if aggregate.get("laal_ms") is None:
            uncomputable["laal"] = "no item produced a commit with decision_audio_sec"

    if "bleu" in requested:
        routed = translation.bleu_by_target(_bleu_pairs(scored, cfg.languages,
                                                        only_routed=True))
        everything = translation.bleu_by_target(_bleu_pairs(scored, cfg.languages,
                                                            only_routed=False))
        aggregate["bleu"] = routed["bleu"]
        aggregate["bleu_all"] = everything["bleu"]
        aggregate["bleu_by_target"] = routed["by_target"]
        if routed["bleu"] is None:
            uncomputable["bleu"] = (
                "no correctly-routed segment had a reference translation"
                if everything["bleu"] is not None else
                "no segment had a reference translation in its target language")

    if "commit" in requested:
        aggregate["commit_stats"] = commit.commit_stats(segments)
        aggregate["segments_per_item"] = commit.segments_per_item(
            len(r.get("segments") or []) for r in scored)

    if "routing" in requested:
        stats = routing.routing_stats(scored, cfg.languages)
        misrouted_items = stats.pop("misrouted_items")
        aggregate.update(stats)
        if stats["lang_detect_accuracy"] is None:
            uncomputable["routing"] = "no segment reported a detected language"
    else:
        misrouted_items = []

    diagnostics = {
        "uncomputable": uncomputable,
        "misrouted_items": misrouted_items,
    }
    return aggregate, diagnostics


def primary_key(metric_name: str) -> str:
    """The per-item field logs.rank_by sorts on."""
    return {"wer": "wer", "cer": "cer", "bleu": "sentence_bleu",
            "fsl": "avg_fsl_sec", "laal": "laal_ms",
            "routing": "route_errors", "commit": "n_segments"}[metric_name]
