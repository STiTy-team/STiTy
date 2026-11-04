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
