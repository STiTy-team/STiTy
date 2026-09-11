"""Commit-reason statistics.

A high `finish` ratio is the tell that the axis's own commit path is not firing:
finish is the safety net, and when it carries most segments the policy under test
is effectively not running. One recorded case had 5,793 of 18,655 segments fall
through to finish on the seg axis.
"""
REASONS = ("vad", "seg", "dot", "finish", "always", "timeout")


def commit_stats(segments) -> dict:
    counts = {reason: 0 for reason in REASONS}
    for seg in segments:
        reason = (seg.get("commit_reason") or "").strip() or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    total = sum(counts.values())
    ratios = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {
        "counts": counts,
        "total": total,
        "ratios": ratios,
        "finish_ratio": ratios.get("finish", 0.0),
    }


def segments_per_item(per_item_counts) -> float | None:
    counts = list(per_item_counts)
    if not counts:
        return None
    return sum(counts) / len(counts)
