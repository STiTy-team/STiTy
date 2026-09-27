from statistics import mean

from omnisteval import Instance, Word
from omnisteval.resegment import resegment
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from core.utils import langs


english = EnglishTextNormalizer()
basic = BasicTextNormalizer(preserve_marks=True)


def succeeded(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "ok"]


def transcribed(records: list[dict]) -> list[dict]:
    return [record for record in records if record["type"] == "transcribed"]


def translated(records: list[dict]) -> list[dict]:
    return [record for record in records if record["type"] == "translated"]


def translating(rows: list[dict], target: str) -> list[dict]:
    return [row for row in rows if src_code(row) != target and not row.get("partial")]


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


def units(text: str, unit: str) -> list[str]:
    if unit == "char":
        return [c for c in (text or "").lower() if c.isalnum()]
    return basic(text or "").split()


def resegmented(row: dict, target: str) -> list[tuple[dict, Instance]]:
    char_level = langs.laal_unit(target) == "char"
    sentences = sorted(row["reference_segmentation"], key=lambda s: s["offset"])
    references = [s["translations"].get(target, "") for s in sentences]
    if not all(references):
        return []
    if char_level:
        references = ["".join(r.split()) for r in references]

    recording_end_ms = (
        max(row["duration_sec"] or 0.0, *(s["offset"] + s["duration"] for s in sentences)) * 1000.0
    )
    segmentation, reference_words = [], []
    for index, (sentence, reference) in enumerate(zip(sentences, references)):
        offset_ms = sentence["offset"] * 1000.0
        segmentation.append(
            {
                "offset": offset_ms,
                "duration": sentence["duration"] * 1000.0,
                "time_to_recording_end": recording_end_ms - offset_ms,
                "doc_id": 0,
                "seg_id": index,
            }
        )
        units = list(reference.lower()) if char_level else reference.lower().split()
        reference_words += [Word(unit, emission_cu=offset_ms, seq_id=index) for unit in units]

    hypothesis_words = []
    for s in translated(row["records"]):
        if s.get("decision_audio_sec") is None or (s.get("target_lang") or target) != target:
            continue
        text = s.get("translation") or ""
        units = list("".join(text.split())) if char_level else text.split()
        emitted_ca = (
            s["translated_elapsed_sec"] * 1000.0
            if s.get("translated_elapsed_sec") is not None
            else None
        )
        hypothesis_words += [
            Word(unit, emission_cu=s["decision_audio_sec"] * 1000.0, emission_ca=emitted_ca)
            for unit in units
        ]

    instances, _ = resegment(
        [reference_words],
        [hypothesis_words],
        segmentation,
        references,
        char_level=char_level,
        lang=None if char_level else target,
    )
    return [
        (sentence, instance)
        for sentence, instance in zip(sentences, instances)
        if sentence_lang(row, sentence) != target and not sentence.get("partial")
    ]


def sentence_lang(row: dict, sentence: dict) -> str:
    code = sentence.get("src_lang") or row["src_lang"]
    return langs.norm_code(code) or code
