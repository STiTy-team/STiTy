"""Transcription accuracy: WER and CER.

`corpus_wer` charges an empty hypothesis as a full deletion. The older
`compute_wer_for_rows` drops any utterance whose hypothesis came back empty, and
callers tend to drop those rows again before scoring, so an utterance the model
failed on disappears twice and the number reads better than the system performed.

`wer_scored_only` reproduces that older number so the two can be compared. When
no hypothesis is empty they must agree -- that equality is what shows the
reimplementation is faithful rather than merely different.

Both are corpus figures: errors summed over every utterance divided by reference
units summed over every utterance, never a mean of per-utterance rates. Short
utterances would otherwise carry the same weight as long ones.
"""
from .text import levenshtein, normalize_words, strip_for_cer


def per_item(item) -> dict:
    """Per-utterance WER and CER. `None` where the reference is empty -- there is
    nothing to measure against, which is not the same as a score of zero."""
    return {"wer": _rate(normalize_words(item.reference), normalize_words(item.hypothesis)),
            "cer": _rate(strip_for_cer(item.reference), strip_for_cer(item.hypothesis))}


def corpus(items) -> tuple[dict, dict]:
    """(values, unavailable). A dataset of audio with no transcripts scores nothing
    here and says so; it does not fail, and it does not report a zero."""
    items = list(items)
    values = {"n_empty_hypothesis": count_empty_hypotheses(items)}
    unavailable = {}

    wer = corpus_wer(items)
    if wer is None:
        unavailable["wer"] = "no item carried a reference transcript"
    else:
        values["wer"] = wer
        legacy = wer_scored_only(items)
        if legacy is not None:
            values["wer_scored_only"] = legacy

    cer = corpus_cer(items)
    if cer is None:
        unavailable["cer"] = "no item carried a reference transcript"
    else:
        values["cer"] = cer
    return values, unavailable


def corpus_wer(items) -> float | None:
    return _corpus(items, normalize_words)


def corpus_cer(items) -> float | None:
    return _corpus(items, strip_for_cer)


def wer_scored_only(items) -> float | None:
    """The legacy number: utterances with an empty hypothesis are left out."""
    return corpus_wer([i for i in items if normalize_words(i.hypothesis)])


def count_empty_hypotheses(items) -> int:
    return sum(1 for i in items if not i.hypothesis)


def _rate(ref, hyp) -> float | None:
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)


def _corpus(items, unitize) -> float | None:
    total_err = total_len = 0
    for item in items:
        ref = unitize(item.reference)
        if not ref:
            continue
        total_err += levenshtein(ref, unitize(item.hypothesis))
        total_len += len(ref)
    if total_len == 0:
        return None
    return total_err / total_len
