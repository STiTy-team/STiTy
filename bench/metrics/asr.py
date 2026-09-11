"""Transcription accuracy.

The corpus WER here is bench's primary number, and it is deliberately not
harness.compute_wer_for_rows: that function drops any row whose hypothesis is
empty (scoring.py:19), and process_batch drops such rows again before scoring, so
an utterance the model failed on disappears twice and WER reads better than the
system performed. corpus_wer counts an empty hypothesis as a full deletion.

wer_scored_only reproduces the legacy number so the two can be compared; when no
hypothesis is empty they must agree, which is what proves this reimplementation
faithful.
"""
import re

from evaluation.harness.scoring import _levenshtein, _strip_for_cer, compute_corpus_cer

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    """Lowercase, drop punctuation, collapse whitespace.

    Same normalization as harness.compute_wer_for_rows so the two numbers are
    comparable.
    """
    lowered = (text or "").lower()
    lowered = _PUNCT.sub(" ", lowered)
    return lowered.split()


def corpus_wer(rows) -> float | None:
    """Word errors summed over all rows / reference words summed over all rows."""
    total_err = total_len = 0
    for row in rows:
        ref = normalize_words(row.get("reference"))
        if not ref:
            continue
        hyp = normalize_words(row.get("hypothesis"))
        total_err += _levenshtein(ref, hyp)
        total_len += len(ref)
    if total_len == 0:
        return None
    return total_err / total_len


def wer_scored_only(rows) -> float | None:
    """The legacy number: rows with an empty hypothesis are excluded."""
    scored = [r for r in rows if normalize_words(r.get("reference"))
              and normalize_words(r.get("hypothesis"))]
    if not scored:
        return None
    return corpus_wer(scored)


def corpus_cer(rows) -> float | None:
    """Character errors / reference characters, via the harness implementation.

    Unlike the WER path this needs no replacement: _levenshtein(ref, "") is
    len(ref), so compute_corpus_cer already charges an empty hypothesis as a full
    deletion.
    """
    return compute_corpus_cer(rows)


def per_item_cer(reference: str, hypothesis: str) -> float | None:
    ref = _strip_for_cer(reference)
    if not ref:
        return None
    return _levenshtein(ref, _strip_for_cer(hypothesis)) / len(ref)


def per_item_wer(reference: str, hypothesis: str) -> float | None:
    ref = normalize_words(reference)
    if not ref:
        return None
    return _levenshtein(ref, normalize_words(hypothesis)) / len(ref)


def count_empty_hypotheses(rows) -> int:
    return sum(1 for r in rows if not (r.get("hypothesis") or "").strip())
