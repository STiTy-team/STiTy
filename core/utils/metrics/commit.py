"""Which trigger closed each segment.

A high `finish` share is the tell that the commit path under test is not firing at
all: finish is the safety net that closes whatever is open when the stream ends, so
when it carries most segments the policy being measured is effectively not running.
One recorded case had 5,793 of 18,655 segments fall through to finish on the seg
axis -- the numbers looked plausible and described nothing.
"""
REASONS = ("vad", "seg", "dot", "always", "finish")


def commit_stats(segments) -> dict:
    counts = {reason: 0 for reason in REASONS}
    for segment in segments:
        reason = segment.commit_reason or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    total = sum(counts.values())
    ratios = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {"counts": counts, "total": total, "ratios": ratios,
            "finish_ratio": ratios.get("finish", 0.0)}


def segments_per_item(items) -> float | None:
    counts = [len(i.segments) for i in items]
    if not counts:
        return None
    return sum(counts) / len(counts)


def corpus(items) -> tuple[dict, dict]:
    """(values, unavailable). Needs no references at all -- an audio-only run still
    reports which trigger closed each segment, which is the point of running one."""
    items = list(items)
    segments = [s for i in items for s in i.segments]
    if not segments:
        return {}, {"commit": "no segment was committed"}
    return {"commit_stats": commit_stats(segments),
            "segments_per_item": segments_per_item(items)}, {}
