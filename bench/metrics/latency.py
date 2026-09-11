"""Latency: first-sentence latency (FSL) and length-adaptive average lag (LAAL).

LAAL's per-segment delay d_i is `decision_audio_sec` -- the source audio read at
the moment the commit was decided. It is NOT `audio_end_sec`, which is the
content boundary (back-estimated from the token ratio for SEG commits). Using the
latter has produced LAAL values off by tens of seconds.
"""
from evaluation.ast.metrics_ast import compute_laal, expand_delays, mean_or_none


def _vad_normalized(fsl_sec: float, commit_reason: str, min_silence_ms: int) -> float:
    """VAD commits wait out the silence window before they can fire.

    Bound to the configured min_silence_ms rather than a literal 0.8, so the
    number does not quietly lie when --vad-min-silence moves.
    """
    if commit_reason == "vad":
        return fsl_sec + min_silence_ms / 1000.0
    return fsl_sec


def fsl_stats(segments, *, min_silence_ms: int = 800) -> dict:
    raw = [s["fsl_sec"] for s in segments if s.get("fsl_sec") is not None]
    normalized = [
        _vad_normalized(s["fsl_sec"], s.get("commit_reason") or "", min_silence_ms)
        for s in segments if s.get("fsl_sec") is not None
    ]
    return {
        "avg_fsl_sec": mean_or_none(raw),
        "avg_fsl_normalized_sec": mean_or_none(normalized),
        "n_seg_with_fsl": len(raw),
    }


def first_token_latency_sec(segments) -> float | None:
    """Elapsed time to the first commit that actually carried a translation.

    A commit with an empty translation does not count -- matching test_ast.py.
    """
    for seg in segments:
        if (seg.get("translation") or "").strip() and seg.get("recv_elapsed_sec") is not None:
            return seg["recv_elapsed_sec"]
    return None


def _delay_pairs(segments, key: str, *, src_duration_ms: float, cap: bool):
    pairs = []
    for seg in segments:
        value = seg.get(key)
        if value is None:
            continue
        delay_ms = float(value) * 1000.0
        if cap:
            delay_ms = min(delay_ms, src_duration_ms)
        pairs.append(((seg.get("translation") or ""), delay_ms))
    return pairs


def laal_for_item(segments, *, src_duration_sec: float, ref_text: str,
                  unit: str = "word", cap_source: bool = True) -> dict:
    """Three d variants, as the AST track computes them.

    laal_ms       non-computation-aware, capped at the source length. Primary:
                  it measures the policy, so it reproduces across GPUs.
    laal_uncapped non-computation-aware, uncapped. Audit only.
    laal_ca_ms    computation-aware -- real elapsed time to receipt.
    """
    src_ms = src_duration_sec * 1000.0
    n_ref = len((ref_text or "").split()) if unit == "word" else len(
        (ref_text or "").replace(" ", ""))
    n_ref = n_ref or None

    def laal(pairs):
        if not pairs:
            return None
        return compute_laal(expand_delays(pairs, unit), src_ms, n_ref)

    return {
        "laal_ms": laal(_delay_pairs(segments, "decision_audio_sec",
                                     src_duration_ms=src_ms, cap=cap_source)),
        "laal_uncapped_ms": laal(_delay_pairs(segments, "decision_audio_sec",
                                              src_duration_ms=src_ms, cap=False)),
        "laal_ca_ms": laal(_delay_pairs(segments, "recv_elapsed_sec",
                                        src_duration_ms=src_ms, cap=False)),
    }
