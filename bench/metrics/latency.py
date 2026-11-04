import math
from difflib import SequenceMatcher
from statistics import mean

from omnisteval import Instance, YAALScorer
from omnisteval.scoring import LAALScorer

from core.utils import langs

from .common import resegmented, translated, translating, units


ALIGNER_STEP_SEC = 0.08


def fsl(rows: list[dict]) -> float | None:
    delays = [
        s["committed_elapsed_sec"] - s["decision_audio_sec"]
        for row in rows
        for s in translated(row["records"])
        if s.get("committed_elapsed_sec") is not None and s.get("decision_audio_sec") is not None
    ]
    return mean(delays) if delays else None


def laal(rows: list[dict], target: str, *, computation_aware: bool = False) -> float | None:
    score = LAALScorer(computation_aware=computation_aware)(_latency_inputs(rows, target))
    return None if math.isnan(score) else score


def yaal(rows: list[dict], target: str, *, computation_aware: bool = False) -> float | None:
    score = YAALScorer(computation_aware=computation_aware)(_latency_inputs(rows, target))
    return None if math.isnan(score) else score


def longyaal(rows: list[dict], target: str, *, computation_aware: bool = False) -> float | None:
    instances = [
        instance
        for row in translating(rows, target)
        if row.get("reference_segmentation")
        for _, instance in resegmented(row, target)
    ]
    score = YAALScorer(computation_aware=computation_aware, is_longform=True)(instances)
    return None if math.isnan(score) else score


def token_emission(rows: list[dict], *, clock: str) -> list[float]:
    delays = []
    for row in rows:
        if not row.get("reference_alignment") or not row.get("records"):
            continue
        unit = langs.laal_unit(langs.norm_code(row["src_lang"]))
        spoken_words, spoken_at = _spoken_ends(row, unit)
        shown_words, shown_at = _shown_at(row["records"], unit)
        matcher = SequenceMatcher(None, spoken_words, shown_words, autojunk=False)
        for a, b, n in matcher.get_matching_blocks():
            for k in range(n):
                spoken, shown = spoken_at[a + k], shown_at[b + k]
                if spoken is not None and shown is not None and shown.get(clock) is not None:
                    delays.append((shown[clock] - spoken) * 1000.0)
    return delays


def _latency_inputs(rows: list[dict], target: str) -> list[Instance]:
    inputs = []
    unit = langs.laal_unit(target)
    separator = " " if unit == "word" else ""
    for row in translating(rows, target):
        reference = separator.join(row["reference_translations"].get(target, "").split())
        source_ms = (row["duration_sec"] or 0.0) * 1000.0
        if not reference or source_ms <= 0 or row.get("reference_segmentation"):
            continue

        emitted, emitted_ca = [], []
        for s in translated(row["records"]):
            if (s.get("target_lang") or target) != target:
                continue
            words = (s.get("translation") or "").split()
            n_units = len(words) if unit == "word" else len("".join(words))
            if s.get("decision_audio_sec") is not None:
                emitted += [min(s["decision_audio_sec"] * 1000.0, source_ms)] * n_units
            if s.get("translated_elapsed_sec") is not None:
                emitted_ca += [s["translated_elapsed_sec"] * 1000.0] * n_units
        inputs.append(
            Instance(
                reference=reference,
                latency_unit=unit,
                source_length=source_ms,
                emission_cu=emitted,
                emission_ca=emitted_ca,
            )
        )
    return inputs


def _spoken_ends(row: dict, unit: str) -> tuple[list[str], list[float | None]]:
    words = units(row["reference"], unit)
    chars, owner = [], []
    for index, word in enumerate(words):
        for c in word:
            if c.isalnum():
                chars.append(c)
                owner.append(index)

    aligned, ends = [], []
    for w in row["reference_alignment"]:
        if row["duration_sec"] and w["end"] > row["duration_sec"] + ALIGNER_STEP_SEC:
            continue
        for c in w["word"].lower():
            if c.isalnum():
                aligned.append(c)
                ends.append(float(w["end"]))

    spoken_at: list[float | None] = [None] * len(words)
    for a, b, n in SequenceMatcher(None, chars, aligned, autojunk=False).get_matching_blocks():
        for k in range(n):
            index = owner[a + k]
            spoken_at[index] = max(spoken_at[index] or 0.0, ends[b + k])
    return words, spoken_at


def _shown_at(records: list[dict], unit: str) -> tuple[list[str], list[dict | None]]:
    committed: list[str] = []
    screens = []
    for record in records:
        if record["type"] == "transcribed":
            committed = committed + units(record["original"], unit)
            moment = {"t": record["committed_elapsed_sec"], "audio": record["decision_audio_sec"]}
            screens.append((moment, committed))
        elif record["type"] == "partial":
            screens.append((record, committed + units(record["text"], unit)))

    shown_at: list[dict | None] = [None] * len(committed)
    stable = len(committed)
    for moment, screen in reversed(screens):
        prefix = 0
        while prefix < min(len(screen), stable) and screen[prefix] == committed[prefix]:
            prefix += 1
        stable = prefix
        for k in range(stable):
            shown_at[k] = moment
    return committed, shown_at
