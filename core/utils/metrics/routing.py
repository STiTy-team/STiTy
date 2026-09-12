"""Language detection and translation routing.

In a multi-language session the translation target is chosen per commit from the
*detected* source language, so two unrelated failures produce the same poor BLEU: a
bad translation, or a good translation sent to the wrong language. Separating them
is the whole point of these numbers -- without them a detection regression looks
like a translation regression.

Comparison is on the raw lowercased codes the server reported, not canonicalized
ones: a server that answers with something outside the expected vocabulary should
count as a miss here rather than being quietly mapped onto a hit.
"""
from collections import Counter


def corpus(items, *, languages) -> tuple[dict, dict]:
    """(values, unavailable). `misrouted_items` rides along in values; it is a list
    of ids for the replay, not a score, so a caller moves it out of the report.

    Detection accuracy and route accuracy are separately available: a server can
    report a detected language without the run knowing what target to expect (no
    language policy), and vice versa.
    """
    detect_hit = detect_total = 0
    route_hit = route_total = 0
    refix = n_segments = 0
    confusion: Counter = Counter()
    misrouted_items: list[str] = []

    for item in items:
        truth = item.src_lang
        expected = languages.expected_target(truth)
        misrouted = False
        for segment in item.segments:
            n_segments += 1
            if segment.language:
                detect_total += 1
                detect_hit += segment.language == truth
                confusion[(truth, segment.language)] += 1
            if segment.target_lang and expected:
                route_total += 1
                if segment.target_lang == expected:
                    route_hit += 1
                else:
                    misrouted = True
            refix += bool(segment.direction_refixed)
        if misrouted:
            misrouted_items.append(item.id)

    values: dict = {"misrouted_items": misrouted_items}
    unavailable: dict = {}

    if not n_segments:
        return values, {"routing": "no segment was committed"}

    values["n_direction_refix"] = refix
    if detect_total:
        values["lang_detect_accuracy"] = detect_hit / detect_total
        values["confusion"] = {f"{truth}->{got}": n
                               for (truth, got), n in sorted(confusion.items())}
    else:
        unavailable["lang_detect_accuracy"] = "no segment reported a detected language"

    if route_total:
        values["route_accuracy"] = route_hit / route_total
        values["n_segments_routed"] = route_total
        values["n_misrouted"] = route_total - route_hit
    else:
        unavailable["route_accuracy"] = (
            "no segment reported a target language the run had an expectation for")
    return values, unavailable


def is_correctly_routed(item, segment, languages) -> bool:
    """Unknown either way counts as routed: an utterance with no expected target
    cannot be said to have gone to the wrong one."""
    expected = languages.expected_target(item.src_lang)
    if not expected or not segment.target_lang:
        return True
    return segment.target_lang == expected


def route_errors(item, *, target_lang: str) -> int:
    """Per-utterance count, for ranking which utterances to inspect."""
    return sum(1 for s in item.segments
               if s.target_lang and s.target_lang != target_lang)
