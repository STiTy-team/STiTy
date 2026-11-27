import re
from collections import Counter

from core.utils import langs

from .common import sentence_lang, src_code, transcribed, translated


COMMIT_REASONS = ("vad", "seg", "dot", "always", "finish")


def language_detection(rows: list[dict]) -> dict:
    detected = [pair for row in rows for pair in _detections(row)]
    reported = any(got != "?" for _, got in detected)
    return {
        "lang_detect_accuracy": (
            sum(truth == got for truth, got in detected) / len(detected) if reported else None
        ),
        "confusion": (
            dict(sorted(Counter(f"{t}->{g}" for t, g in detected).items())) if reported else None
        ),
    }


def _detections(row: dict) -> list[tuple[str, str]]:
    sentences = row.get("reference_segmentation") or []
    pairs, heard_from = [], 0.0
    for commit in transcribed(row["records"]):
        heard_to = commit.get("decision_audio_sec") or heard_from
        truth = (
            _spoken_language(row, sentences, heard_from, heard_to) if sentences else src_code(row)
        )
        pairs.append((truth, langs.norm_code(commit.get("language") or "") or "?"))
        heard_from = heard_to
    return pairs


def language_routing(rows: list[dict]) -> dict:
    """Score the language each commit was translated from (the `translated` records, after
    langid) against the language of the reference sentence its text belongs to. The audio
    window between decisions is not used as truth: a commit is confirmed a chunk after it was
    spoken, so that window often holds the next utterance."""
    routed = [pair for row in rows for pair in _routes(row)]
    matched = [(truth, got) for truth, got in routed if truth]
    return {
        "lang_detect_route_accuracy": (
            sum(truth == got for truth, got in matched) / len(matched) if matched else None
        ),
        "lang_detect_route_confusion": (
            dict(sorted(Counter(f"{t}->{g}" for t, g in matched).items())) if matched else None
        ),
        "lang_detect_route_unmatched": (len(routed) - len(matched)) if routed else None,
    }


ROUTE_LOOKBACK_SEC = 8.0
ROUTE_MIN_OVERLAP = 0.3


def _routes(row: dict) -> list[tuple[str | None, str]]:
    sentences = row.get("reference_segmentation") or []
    if not sentences:
        return [
            (src_code(row), langs.norm_code(c.get("language") or "") or "?")
            for c in translated(row["records"])
        ]
    grams = [_bigrams(s.get("transcript") or "") for s in sentences]
    pairs = []
    for commit in translated(row["records"]):
        heard_to = commit.get("decision_audio_sec") or 0.0
        said = _bigrams(commit.get("original") or "")
        best, truth = 0.0, None
        for sentence, sentence_grams in zip(sentences, grams):
            start, end = sentence["offset"], sentence["offset"] + sentence["duration"]
            if end < heard_to - ROUTE_LOOKBACK_SEC or start > heard_to or not said:
                continue
            overlap = sum((said & sentence_grams).values()) / sum(said.values())
            if overlap > best:
                best, truth = overlap, sentence_lang(row, sentence)
        got = langs.norm_code(commit.get("language") or "") or "?"
        pairs.append((truth if best >= ROUTE_MIN_OVERLAP else None, got))
    return pairs


def _bigrams(text: str) -> Counter:
    letters = re.sub(r"[\W_]+", "", text.lower())
    return Counter(letters[i : i + 2] for i in range(len(letters) - 1))


def _spoken_language(row: dict, sentences: list[dict], start: float, end: float) -> str:
    spoken = max(
        sentences,
        key=lambda s: min(end, s["offset"] + s["duration"]) - max(start, s["offset"]),
    )
    return sentence_lang(row, spoken)


def commit_reasons(rows: list[dict]) -> dict | None:
    reasons = Counter(
        s.get("commit_reason") or "unknown" for row in rows for s in translated(row["records"])
    )
    total = sum(reasons.values())
    if not total:
        return None
    return {
        reason: reasons[reason] / total for reason in dict.fromkeys([*COMMIT_REASONS, *reasons])
    }
