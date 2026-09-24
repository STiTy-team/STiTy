import math
import re
from collections import Counter
from difflib import SequenceMatcher
from statistics import mean

import jiwer
import numpy as np
from omnisteval import Instance, Word, YAALScorer
from omnisteval.resegment import resegment
from omnisteval.scoring import LAALScorer
from sacrebleu.metrics import BLEU
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from core.utils import langs

BLEU_TOKENIZE = {"zh": "zh", "ja": "ja-mecab", "ko": "ko-mecab"}
COMMIT_REASONS = ("vad", "seg", "dot", "always", "finish")
ALIGNER_STEP_SEC = 0.08

NONSPEECH = re.compile(
    r"[\(\[]\s*(?:laughter and applause|laughter|laughs|applause|cheers|cheering|music|"
    r"singing|video|audio|recording|silence|sighs|beat|sniffs|clapping|boos|gasps|"
    r"gelächter|lachen|applaus|beifall|musik|gesang|jubel|stille|seufzt|klatschen)\s*[\)\]]",
    re.IGNORECASE,
)

NO_TRANSCRIPT = "no item carried a reference transcript"
NO_TRANSLATION = "no item carried a reference translation in its target language"
SENTENCE_LEVEL = NO_TRANSLATION + ", or the run is long-form (see longyaal_ms)"
NOT_LONGFORM = ("the run is not long-form (set longform: true on the dataset), or no talk "
                "had a reference translation for every sentence")
NO_SEGMENT = "no segment was committed"
NO_ALIGNMENT = ("no item had both a reference alignment and recorded emissions, "
                "or no recognized word matched the reference")
UNAVAILABLE = {
    "wer": NO_TRANSCRIPT,
    "wer_by_lang": NO_TRANSCRIPT,
    "wer_scored_only": "every hypothesis was empty",
    "cer": NO_TRANSCRIPT,
    "cer_by_lang": NO_TRANSCRIPT,
    "bleu": NO_TRANSLATION,
    "bleu_by_pair": NO_TRANSLATION,
    "avg_fsl_sec": NO_SEGMENT,
    "laal_ms": SENTENCE_LEVEL,
    "laal_ca_ms": SENTENCE_LEVEL,
    "yaal_ms": SENTENCE_LEVEL + ", or every item emitted after its source ended",
    "yaal_ca_ms": SENTENCE_LEVEL + ", or every item emitted after its source ended",
    "longyaal_ms": NOT_LONGFORM,
    "longyaal_ca_ms": NOT_LONGFORM,
    "token_emission_ms": NO_ALIGNMENT,
    "token_emission_ca_ms": NO_ALIGNMENT,
    "lang_detect_accuracy": "no segment reported a detected language",
    "route_accuracy": "no segment reported a target language",
    "commit_reasons": NO_SEGMENT,
}

_english = EnglishTextNormalizer()
_basic = BasicTextNormalizer(preserve_marks=True)


def score_row(row: dict, languages) -> dict:
    target = languages.expected_target(row["src_lang"])
    emission = token_emission([row], clock="audio")
    emission_ca = token_emission([row], clock="t")
    values = {
        "n_segments": len(row["segments"]),
        "route_errors": sum(1 for s in row["segments"]
                            if s.get("target_lang") and s["target_lang"] != target),
        "wer": wer([row]),
        "cer": cer([row]),
        "sentence_bleu": bleu([row], languages),
        "avg_fsl_sec": fsl([row]),
        "laal_ms": laal([row], languages),
        "laal_ca_ms": laal([row], languages, computation_aware=True),
        "yaal_ms": yaal([row], languages),
        "yaal_ca_ms": yaal([row], languages, computation_aware=True),
        "longyaal_ms": longyaal([row], languages),
        "longyaal_ca_ms": longyaal([row], languages, computation_aware=True),
        "token_emission_ms": mean(emission) if emission else None,
        "token_emission_ca_ms": mean(emission_ca) if emission_ca else None,
    }
    return {key: value for key, value in values.items() if value is not None}


def score_run(rows: list[dict], languages) -> tuple[dict, dict]:
    segments = [s for row in rows for s in row["segments"]]
    wer_by_lang = per_lang(wer, rows)
    cer_by_lang = per_lang(cer, rows)
    bleu_by_pair = bleu_pairs(rows, languages)
    values = {
        "wer": macro(wer_by_lang),
        "wer_by_lang": wer_by_lang,
        "wer_scored_only": macro(per_lang(wer, [row for row in rows if row["hypothesis"]])),
        "cer": macro(cer_by_lang),
        "cer_by_lang": cer_by_lang,
        "bleu": macro(bleu_by_pair),
        "bleu_by_pair": bleu_by_pair,
        "avg_fsl_sec": fsl(rows),
        "laal_ms": laal(rows, languages),
        "laal_ca_ms": laal(rows, languages, computation_aware=True),
        "yaal_ms": yaal(rows, languages),
        "yaal_ca_ms": yaal(rows, languages, computation_aware=True),
        "longyaal_ms": longyaal(rows, languages),
        "longyaal_ca_ms": longyaal(rows, languages, computation_aware=True),
        **routing(rows, languages),
        "commit_reasons": commit_reasons(rows),
        "segments_per_item": len(segments) / len(rows) if rows else None,
    }
    for key, clock in (("token_emission", "audio"), ("token_emission_ca", "t")):
        delays = token_emission(rows, clock=clock)
        values[f"{key}_ms"] = mean(delays) if delays else None
        if delays:
            values[f"{key}_p50_ms"] = float(np.percentile(delays, 50))
            values[f"{key}_p90_ms"] = float(np.percentile(delays, 90))

    metrics = {key: value for key, value in values.items() if value is not None}
    unavailable = {key: UNAVAILABLE[key] for key, value in values.items()
                   if value is None and key in UNAVAILABLE}
    return metrics, unavailable


def src_code(row: dict) -> str:
    return langs.norm_code(row["src_lang"]) or row["src_lang"]


def per_lang(score, rows: list[dict]) -> dict[str, float] | None:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(src_code(row), []).append(row)
    scores = {lang: score(group) for lang, group in sorted(groups.items())}
    return {lang: value for lang, value in scores.items() if value is not None} or None


def macro(scores: dict[str, float] | None) -> float | None:
    return mean(scores.values()) if scores else None


def wer(rows: list[dict]) -> float | None:
    transcripts = _transcripts(rows)
    if not transcripts:
        return None
    return jiwer.wer([ref for ref, _ in transcripts], [hyp for _, hyp in transcripts])


def cer(rows: list[dict]) -> float | None:
    transcripts = _transcripts(rows)
    if not transcripts:
        return None
    return jiwer.cer(["".join(ref.split()) for ref, _ in transcripts],
                     ["".join(hyp.split()) for _, hyp in transcripts])


def bleu(rows: list[dict], languages) -> float | None:
    return macro(bleu_pairs(rows, languages))


def bleu_pairs(rows: list[dict], languages) -> dict[str, float] | None:
    by_pair: dict[str, list[dict]] = {}
    for sentence in translation_sentences(rows, languages):
        by_pair.setdefault(sentence["pair"], []).append(sentence)

    scores = {}
    for pair, sentences in sorted(by_pair.items()):
        target = pair.rsplit("-", 1)[1]
        metric = BLEU(tokenize=BLEU_TOKENIZE.get(target, "13a"),
                      effective_order=len(sentences) == 1)
        score = metric.corpus_score([s["mt"] for s in sentences], [[s["ref"] for s in sentences]])
        scores[pair] = score.score
    return scores or None


def translation_sentences(rows: list[dict], languages) -> list[dict]:
    sentences = []
    for row in rows:
        target = languages.expected_target(row["src_lang"])
        if row.get("reference_segmentation"):
            ordered = sorted(row["reference_segmentation"], key=lambda s: s["offset"])
            found = [(sentence.get("transcript") or "", instance.prediction, instance.reference)
                     for sentence, instance in zip(ordered, _resegmented(row, languages))]
        elif row["reference_translations"].get(target):
            found = [(row["reference"] or "", _translation(row, target),
                      row["reference_translations"][target])]
        else:
            found = []
        pair = f"{src_code(row)}-{target}"
        sentences += [{"pair": pair, "src": _clean(src), "mt": _clean(mt), "ref": _clean(ref)}
                      for src, mt, ref in found]
    return sentences


def fsl(rows: list[dict]) -> float | None:
    delays = [s["recv_elapsed_sec"] - s["decision_audio_sec"]
              for row in rows for s in row["segments"]
              if s.get("recv_elapsed_sec") is not None and s.get("decision_audio_sec") is not None]
    return mean(delays) if delays else None


def laal(rows: list[dict], languages, *, computation_aware: bool = False) -> float | None:
    score = LAALScorer(computation_aware=computation_aware)(_latency_inputs(rows, languages))
    return None if math.isnan(score) else score


def yaal(rows: list[dict], languages, *, computation_aware: bool = False) -> float | None:
    score = YAALScorer(computation_aware=computation_aware)(_latency_inputs(rows, languages))
    return None if math.isnan(score) else score


def longyaal(rows: list[dict], languages, *, computation_aware: bool = False) -> float | None:
    instances = [instance for row in rows if row.get("reference_segmentation")
                 for instance in _resegmented(row, languages)]
    score = YAALScorer(computation_aware=computation_aware, is_longform=True)(instances)
    return None if math.isnan(score) else score


def token_emission(rows: list[dict], *, clock: str) -> list[float]:
    delays = []
    for row in rows:
        if not row.get("reference_alignment") or not row.get("emissions"):
            continue
        unit = langs.laal_unit(langs.norm_code(row["src_lang"]))
        spoken_words, spoken_at = _spoken_ends(row, unit)
        shown_words, shown_at = _shown_at(row["emissions"], unit)
        matcher = SequenceMatcher(None, spoken_words, shown_words, autojunk=False)
        for a, b, n in matcher.get_matching_blocks():
            for k in range(n):
                spoken, shown = spoken_at[a + k], shown_at[b + k]
                if spoken is not None and shown is not None and shown.get(clock) is not None:
                    delays.append((shown[clock] - spoken) * 1000.0)
    return delays


def routing(rows: list[dict], languages) -> dict:
    detected = [(row["src_lang"], s["language"])
                for row in rows for s in row["segments"] if s.get("language")]
    routed = [s["target_lang"] == languages.expected_target(row["src_lang"])
              for row in rows for s in row["segments"] if s.get("target_lang")]
    return {
        "lang_detect_accuracy": (sum(truth == got for truth, got in detected) / len(detected)
                                 if detected else None),
        "confusion": dict(sorted(Counter(f"{t}->{g}" for t, g in detected).items())) or None,
        "route_accuracy": sum(routed) / len(routed) if routed else None,
    }


def commit_reasons(rows: list[dict]) -> dict | None:
    reasons = Counter(s.get("commit_reason") or "unknown" for row in rows for s in row["segments"])
    total = sum(reasons.values())
    if not total:
        return None
    return {reason: reasons[reason] / total for reason in dict.fromkeys([*COMMIT_REASONS, *reasons])}


def _transcripts(rows: list[dict]) -> list[tuple[str, str]]:
    transcripts = []
    for row in rows:
        normalize = _english if langs.norm_code(row["src_lang"]) == "en" else _basic
        reference = normalize(row["reference"] or "").strip()
        if reference:
            transcripts.append((reference, normalize(row["hypothesis"] or "").strip()))
    return transcripts


def _clean(text: str) -> str:
    return " ".join(NONSPEECH.sub(" ", text).split())


def _translation(row: dict, target: str) -> str:
    return " ".join(s["translation"] for s in row["segments"]
                    if s.get("translation") and (s.get("target_lang") or target) == target)


def _latency_inputs(rows: list[dict], languages) -> list[Instance]:
    inputs = []
    for row in rows:
        target = languages.expected_target(row["src_lang"])
        unit = langs.laal_unit(target)
        separator = " " if unit == "word" else ""
        reference = separator.join(row["reference_translations"].get(target, "").split())
        source_ms = (row["duration_sec"] or 0.0) * 1000.0
        if not reference or source_ms <= 0 or row.get("reference_segmentation"):
            continue

        emitted, emitted_ca = [], []
        for s in row["segments"]:
            words = (s.get("translation") or "").split()
            n_units = len(words) if unit == "word" else len("".join(words))
            if s.get("decision_audio_sec") is not None:
                emitted += [min(s["decision_audio_sec"] * 1000.0, source_ms)] * n_units
            if s.get("recv_elapsed_sec") is not None:
                emitted_ca += [s["recv_elapsed_sec"] * 1000.0] * n_units
        inputs.append(Instance(reference=reference, latency_unit=unit, source_length=source_ms,
                               emission_cu=emitted, emission_ca=emitted_ca))
    return inputs


def _resegmented(row: dict, languages) -> list[Instance]:
    target = languages.expected_target(row["src_lang"])
    char_level = langs.laal_unit(target) == "char"
    sentences = sorted(row["reference_segmentation"], key=lambda s: s["offset"])
    references = [s["translations"].get(target, "") for s in sentences]
    if not all(references):
        return []
    if char_level:
        references = ["".join(r.split()) for r in references]

    recording_end_ms = max(row["duration_sec"] or 0.0,
                           *(s["offset"] + s["duration"] for s in sentences)) * 1000.0
    segmentation, reference_words = [], []
    for index, (sentence, reference) in enumerate(zip(sentences, references)):
        offset_ms = sentence["offset"] * 1000.0
        segmentation.append({"offset": offset_ms, "duration": sentence["duration"] * 1000.0,
                             "time_to_recording_end": recording_end_ms - offset_ms,
                             "doc_id": 0, "seg_id": index})
        units = list(reference.lower()) if char_level else reference.lower().split()
        reference_words += [Word(unit, emission_cu=offset_ms, seq_id=index) for unit in units]

    hypothesis_words = []
    for s in row["segments"]:
        if s.get("decision_audio_sec") is None or (s.get("target_lang") or target) != target:
            continue
        text = s.get("translation") or ""
        units = list("".join(text.split())) if char_level else text.split()
        emitted_ca = s["recv_elapsed_sec"] * 1000.0 if s.get("recv_elapsed_sec") is not None else None
        hypothesis_words += [Word(unit, emission_cu=s["decision_audio_sec"] * 1000.0,
                                  emission_ca=emitted_ca) for unit in units]

    instances, _ = resegment([reference_words], [hypothesis_words], segmentation, references,
                             char_level=char_level, lang=None if char_level else target)
    return instances


def _spoken_ends(row: dict, unit: str) -> tuple[list[str], list[float | None]]:
    words = _units(row["reference"], unit)
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


def _shown_at(emissions: list[dict], unit: str) -> tuple[list[str], list[dict | None]]:
    committed: list[str] = []
    screens = []
    for emission in emissions:
        if emission["kind"] == "commit":
            committed = committed + _units(emission["text"], unit)
            screens.append((emission, committed))
        else:
            screens.append((emission, committed + _units(emission["text"], unit)))

    shown_at: list[dict | None] = [None] * len(committed)
    stable = len(committed)
    for emission, screen in reversed(screens):
        prefix = 0
        while prefix < min(len(screen), stable) and screen[prefix] == committed[prefix]:
            prefix += 1
        stable = prefix
        for k in range(stable):
            shown_at[k] = emission
    return committed, shown_at


def _units(text: str, unit: str) -> list[str]:
    if unit == "char":
        return [c for c in (text or "").lower() if c.isalnum()]
    return _basic(text or "").split()
