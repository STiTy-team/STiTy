"""Latency: first-sentence latency (FSL) and length-adaptive average lagging (LAAL).

LAAL (Papi et al. 2022, "Over-Generation Cannot Be Rewarded"):

    LAAL = (1/tau) * sum_{i=1..tau} [ d_i - (i-1) * T / max(|Y_hyp|, |Y_ref|) ]
    tau  = min{ i : d_i >= T }, or |Y_hyp| when no such i exists

`d_i` is the delay at which target unit i appeared and `T` is the source audio
length. The single difference from average lagging is the denominator
max(|Y_hyp|, |Y_ref|): plain AL divides by |Y_hyp| alone, so generating less
shrinks the (i-1)*gamma term and under-generation is rewarded with a better score.

Commits happen per segment, not per token, so every target unit inside one segment
shares that segment's delay -- the standard chunk-level treatment.

The delay must come from `decision_audio_sec`, the source audio read at the moment
the commit was decided. It is *not* `audio_end_sec`, which is the content boundary
and is back-estimated from the token ratio for SEG commits; using it has produced
LAAL values wrong by tens of seconds.
"""
from .text import count_units, mean_or_none

LAAL_KEYS = ("laal_ms", "laal_ca_ms", "laal_uncapped_ms")


def fsl_stats(segments) -> dict:
    """Averaged over the segments given, so a corpus figure must pass every segment
    in the run rather than averaging per-utterance averages.

    No VAD normalization: the silence a detector waits out before it decides sits in
    both clocks `Segment.fsl_sec` is derived from, so it cancels. Adding the window
    back -- as a figure computed from a single clock had to -- would count it twice.
    """
    values = [s.fsl_sec for s in segments if s.fsl_sec is not None]
    return {"avg_fsl_sec": mean_or_none(values), "n_seg_with_fsl": len(values)}


def laal_for_item(item, *, target_lang: str, unit: str = "word",
                  cap_source: bool = True) -> dict:
    """Three variants of d for one utterance. Empty when LAAL does not apply.

    laal_ms        delay is source audio read at the commit decision, capped at the
                   source length. The primary number: it measures the policy, so it
                   reproduces on different hardware.
    laal_uncapped  the same, uncapped. Audit only -- a commit decided after the
                   audio ended reports a lag longer than the utterance.
    laal_ca_ms     computation-aware: real elapsed time until the client had it.
                   What a listener actually waits.

    Check: laal_ca_ms - laal_ms should be about mean(fsl). A large gap means the
    timing is wired wrong.

    Returns nothing without a reference translation. The denominator is
    max(|Y_hyp|, |Y_ref|), so dropping |Y_ref| does not make LAAL approximate -- it
    makes it average lagging, a different metric whose denominator |Y_hyp| alone
    rewards under-generation. Reporting that as `laal_ms` would put two
    incomparable numbers in one column.
    """
    src_ms = item.duration_sec * 1000.0
    n_ref = count_units(item.reference_translation(target_lang), unit)
    if not n_ref or src_ms <= 0:
        return {}

    def laal(key: str, *, cap: bool):
        delays = expand_delays(item.segments, key, unit=unit,
                              src_duration_ms=src_ms if cap else None)
        return compute_laal(delays, src_ms, n_ref)

    out = {"laal_ms": laal("decision_audio_sec", cap=cap_source),
           "laal_uncapped_ms": laal("decision_audio_sec", cap=False),
           "laal_ca_ms": laal("recv_elapsed_sec", cap=False)}
    return {k: v for k, v in out.items() if v is not None}


def expand_delays(segments, key: str, *, unit: str = "word",
                  src_duration_ms: float | None = None) -> list[float]:
    """Segment delays to one delay per target unit, in milliseconds.

    Segments whose delay was never recorded are skipped; an empty translation
    contributes no units and so drops out on its own.
    """
    delays: list[float] = []
    for segment in segments:
        value = getattr(segment, key)
        if value is None:
            continue
        n = count_units(segment.translation, unit)
        if n <= 0:
            continue
        delay_ms = value * 1000.0
        if src_duration_ms is not None:
            delay_ms = min(delay_ms, src_duration_ms)
        delays.extend([delay_ms] * n)
    return delays


def compute_laal(delays_ms, src_duration_ms: float,
                 n_ref_units: int | None = None) -> float | None:
    """LAAL in milliseconds, or None when it cannot be computed.

    `n_ref_units` None makes the denominator |Y_hyp| alone, i.e. plain average
    lagging.
    """
    n_hyp = len(delays_ms)
    if n_hyp == 0 or src_duration_ms <= 0:
        return None

    gamma = src_duration_ms / max(n_hyp, n_ref_units or 0)

    tau = n_hyp
    for i, delay in enumerate(delays_ms, start=1):
        if delay >= src_duration_ms:
            tau = i
            break

    total = sum(delays_ms[i - 1] - (i - 1) * gamma for i in range(1, tau + 1))
    return total / tau


def corpus(items, *, languages, unit_for) -> tuple[dict, dict]:
    """(values, unavailable).

    FSL and LAAL aggregate differently on purpose. FSL is a property of a commit, so
    every commit in the run counts once. LAAL is defined per utterance -- its gamma
    term is relative to one source length -- so it can only be averaged afterwards.

    FSL needs no references, so an audio-only run still reports it. LAAL needs a
    reference translation and therefore does not.
    """
    items = list(items)
    segments = [s for i in items for s in i.segments]
    values, unavailable = {}, {}

    fsl = fsl_stats(segments)
    if fsl["n_seg_with_fsl"]:
        values.update(fsl)
    else:
        unavailable["fsl"] = ("no segment carried fsl_sec" if segments
                              else "no segment was committed")

    per_item = [laal_for_item(item, target_lang=target, unit=unit_for(target))
                for item, target in ((i, languages.expected_target(i.src_lang))
                                     for i in items)]
    for key in LAAL_KEYS:
        value = mean_or_none(d.get(key) for d in per_item)
        if value is not None:
            values[key] = value
    if "laal_ms" not in values:
        unavailable["laal"] = _why_no_laal(items, segments, languages)
    return values, unavailable


def _why_no_laal(items, segments, languages) -> str:
    if not segments:
        return "no segment was committed"
    if not any(i.reference_translation(languages.expected_target(i.src_lang))
               for i in items):
        return "no item carried a reference translation in its expected target"
    if not any(s.decision_audio_sec is not None for s in segments):
        return "no commit carried decision_audio_sec"
    if not any(i.duration_sec > 0 for i in items):
        return "no item reported an audio duration"
    return "no committed segment produced a target unit to measure"
