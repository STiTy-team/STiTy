from statistics import mean

import numpy as np

from .common import macro, per_lang, succeeded, translated
from .latency import fsl, laal, longyaal, token_emission, yaal
from .asr import commit_reasons, language_detection
from .transcription import cer, wer
from .translation import bleu, bleu_pairs


NO_TRANSCRIPT = "no item carried a reference transcript"
NO_TRANSLATION = (
    "no item carried a reference translation in its target language, "
    "or every item was already in the target language"
)
SENTENCE_LEVEL = NO_TRANSLATION + ", or the run is long-form (see longyaal_ms)"
NOT_LONGFORM = (
    "the run is not long-form (set longform: true on the dataset), or no talk "
    "had a reference translation for every sentence"
)
NO_SEGMENT = "no segment was committed"
NO_ALIGNMENT = (
    "no item had both a reference alignment and recorded partials or commits, "
    "or no recognized word matched the reference"
)
UNAVAILABLE = {
    "wer": NO_TRANSCRIPT,
    "wer_by_lang": NO_TRANSCRIPT,
    "wer_scored_only": "every transcription output was empty",
    "cer": NO_TRANSCRIPT,
    "cer_by_lang": NO_TRANSCRIPT,
    "bleu": NO_TRANSLATION,
    "bleu_by_pair": NO_TRANSLATION,
    "avg_fsl_sec": NO_SEGMENT,
    "laal_ms": SENTENCE_LEVEL,
    "laal_ca_ms": SENTENCE_LEVEL,
    "yaal_ms": SENTENCE_LEVEL + ", or every item emitted after its source ended",
    "yaal_ca_ms": SENTENCE_LEVEL + ", or every item emitted after its source ended",
    "longyaal_ms": NOT_LONGFORM,
    "longyaal_ca_ms": NOT_LONGFORM,
    "token_emission_ms": NO_ALIGNMENT,
    "token_emission_ca_ms": NO_ALIGNMENT,
    "lang_detect_accuracy": "no segment reported a detected language",
    "commit_reasons": NO_SEGMENT,
}


def score_row(row: dict, target: str) -> dict:
    emission = token_emission([row], clock="audio")
    emission_ca = token_emission([row], clock="t")
    values = {
        "n_segments": len(translated(row["records"])),
        "wer": wer([row]),
        "cer": cer([row]),
        "sentence_bleu": bleu([row], target),
        "avg_fsl_sec": fsl([row]),
        "laal_ms": laal([row], target),
        "laal_ca_ms": laal([row], target, computation_aware=True),
        "yaal_ms": yaal([row], target),
        "yaal_ca_ms": yaal([row], target, computation_aware=True),
        "longyaal_ms": longyaal([row], target),
        "longyaal_ca_ms": longyaal([row], target, computation_aware=True),
        "token_emission_ms": mean(emission) if emission else None,
        "token_emission_ca_ms": mean(emission_ca) if emission_ca else None,
    }
    return {key: value for key, value in values.items() if value is not None}


def score_run(rows: list[dict], target: str) -> tuple[dict, dict]:
    rows = succeeded(rows)
    segments = [s for row in rows for s in translated(row["records"])]
    wer_by_lang = per_lang(wer, rows)
    cer_by_lang = per_lang(cer, rows)
    bleu_by_pair = bleu_pairs(rows, target)
    values = {
        "wer": macro(wer_by_lang),
        "wer_by_lang": wer_by_lang,
        "wer_scored_only": macro(
            per_lang(wer, [row for row in rows if row["transcription_output"]])
        ),
        "cer": macro(cer_by_lang),
        "cer_by_lang": cer_by_lang,
        "bleu": macro(bleu_by_pair),
        "bleu_by_pair": bleu_by_pair,
        "avg_fsl_sec": fsl(rows),
        "laal_ms": laal(rows, target),
        "laal_ca_ms": laal(rows, target, computation_aware=True),
        "yaal_ms": yaal(rows, target),
        "yaal_ca_ms": yaal(rows, target, computation_aware=True),
        "longyaal_ms": longyaal(rows, target),
        "longyaal_ca_ms": longyaal(rows, target, computation_aware=True),
        **language_detection(rows),
        "commit_reasons": commit_reasons(rows),
        "segments_per_item": len(segments) / len(rows) if rows else None,
    }
    for key, clock in (("token_emission", "audio"), ("token_emission_ca", "t")):
        delays = token_emission(rows, clock=clock)
        values[f"{key}_ms"] = mean(delays) if delays else None
        if delays:
            values[f"{key}_p50_ms"] = float(np.percentile(delays, 50))
            values[f"{key}_p90_ms"] = float(np.percentile(delays, 90))

    metrics = {key: value for key, value in values.items() if value is not None}
    unavailable = {
        key: UNAVAILABLE[key]
        for key, value in values.items()
        if value is None and key in UNAVAILABLE
    }
    return metrics, unavailable
