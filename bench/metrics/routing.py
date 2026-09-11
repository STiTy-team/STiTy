"""Language detection and translation routing.

In a multi-language session the translation target is chosen per commit from the
*detected* source language, so two different failures can produce the same bad
BLEU: a poor translation, or a correctly translated utterance sent to the wrong
language. These metrics separate them.
"""
from collections import Counter


def _norm(value: str) -> str:
    return (value or "").strip().lower()


def routing_stats(rows, languages) -> dict:
    """rows carry src_lang and segments[] with `language` and `target_lang`."""
    detect_hit = detect_total = 0
    route_hit = route_total = 0
    refix = 0
    confusion: Counter[tuple[str, str]] = Counter()
    misrouted_items: list[str] = []

    for row in rows:
        truth = _norm(row.get("src_lang"))
        expected_target = languages.expected_target(truth)
        item_misrouted = False
        for seg in row.get("segments") or []:
            detected = _norm(seg.get("language"))
            if detected:
                detect_total += 1
                if detected == truth:
                    detect_hit += 1
                confusion[(truth, detected)] += 1
            used_target = _norm(seg.get("target_lang"))
            if used_target and expected_target:
                route_total += 1
                if used_target == expected_target:
                    route_hit += 1
                else:
                    item_misrouted = True
            if seg.get("direction_refixed"):
                refix += 1
        if item_misrouted:
            misrouted_items.append(row.get("id"))

    return {
        "lang_detect_accuracy": (detect_hit / detect_total) if detect_total else None,
        "route_accuracy": (route_hit / route_total) if route_total else None,
        "n_segments_routed": route_total,
        "n_misrouted": route_total - route_hit if route_total else 0,
        "n_direction_refix": refix,
        "misrouted_items": misrouted_items,
        "confusion": {f"{truth}->{got}": n for (truth, got), n in sorted(confusion.items())},
    }


def is_correctly_routed(row, seg, languages) -> bool:
    expected = languages.expected_target(_norm(row.get("src_lang")))
    used = _norm(seg.get("target_lang"))
    if not expected or not used:
        return True
    return used == expected
