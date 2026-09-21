from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol


QUALITY_ANNOTATOR_VERSION = "2026-09-21.1"
CRITICAL_PROMPT_VERSION = "ko-en-critical-v1"
FLUENCY_PROMPT_VERSION = "spoken-fluency-mqm-v1"

FLUENCY_SYSTEM_PROMPT = """You are a strict evaluator of spoken-language fluency.
Judge only the candidate text as conversation in the target language. Do not infer or judge source meaning or translation faithfulness.
Use this 1-5 rubric: 5 fully natural spoken language; 4 natural with a small awkwardness; 3 understandable but noticeably awkward; 2 difficult or repeatedly ungrammatical; 1 unusable.
Also annotate MQM fluency/style errors. Allowed categories are grammar, word_order, word_form, spelling, punctuation, register, awkwardness, repetition, untranslated_fragment, and consistency. Severity is minor, major, or critical. Critical is reserved for text that is effectively unusable as target-language conversation.
Every error must identify an exact substring using zero-based start and exclusive end character offsets in the candidate. Do not create an error merely because a different wording would be preferable.
Return one JSON object with keys score, reason, and errors. errors is an array of objects with category, severity, start, end, text, and explanation."""

CRITICAL_SYSTEM_PROMPT = """You annotate critical information in Korean-English conversational translation.
The source and reference establish which facts should be preserved. Extract only person, location, organization, product, domain term, postal address, email, and URL spans; numeric values are supplied separately and must not be repeated.
For semantically corresponding reference and candidate spans, use exactly the same language-neutral canonical_value. Transliteration, translation, abbreviation, and original-script retention may be equivalent. Candidate-only invented facts must still be extracted with their own canonical_value.
Every span must use an exact substring and zero-based start and exclusive end character offsets in its own text. Return JSON with reference_spans, candidate_spans, and alignments. Each span has type, text, start, end, canonical_value, and accepted_values. Each alignment has reference_index, candidate_index, and relation, where relation is equivalent or conflicting."""


class JsonJudge(Protocol):
    model: str

    async def ask(self, *, purpose: str, system: str, payload: dict) -> dict:
        ...


class OpenAIJsonJudge:
    def __init__(self, *, model: str = "gpt-5.4-mini", api_key: str | None = None,
                 max_retries: int = 3):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for automatic quality annotation")
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=key)
        self.model = model
        self.max_retries = max_retries

    async def ask(self, *, purpose: str, system: str, payload: dict) -> dict:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                value = json.loads(response.choices[0].message.content)
                if not isinstance(value, dict):
                    raise ValueError(f"{purpose} did not return a JSON object")
                return value
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"{purpose} failed after {self.max_retries} attempts: {last_error}")


@dataclass(frozen=True)
class Span:
    type: str
    text: str
    start: int
    end: int
    canonical_value: str

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "canonical_value": self.canonical_value,
        }


_EN_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_KO_NUMBERS = {
    "영": 0, "공": 0, "일": 1, "하나": 1, "한": 1, "첫": 1,
    "이": 2, "둘": 2, "두": 2, "삼": 3, "셋": 3, "세": 3,
    "사": 4, "넷": 4, "네": 4, "오": 5, "다섯": 5,
    "육": 6, "여섯": 6, "칠": 7, "일곱": 7, "팔": 8, "여덟": 8,
    "구": 9, "아홉": 9, "십": 10, "열": 10, "열한": 11,
    "열두": 12, "열세": 13, "열네": 14, "스물": 20,
}

_UNIT_ALIASES = {
    "km": "km", "kilometer": "km", "kilometers": "km", "킬로미터": "km",
    "m": "m", "meter": "m", "meters": "m", "미터": "m",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "센티미터": "cm",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "밀리미터": "mm",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg", "킬로그램": "kg",
    "g": "g", "gram": "g", "grams": "g", "그램": "g",
    "l": "L", "liter": "L", "liters": "L", "litre": "L", "litres": "L", "리터": "L",
    "ml": "mL", "milliliter": "mL", "milliliters": "mL", "밀리리터": "mL",
    "°c": "°C", "celsius": "°C", "섭씨": "°C",
    "°f": "°F", "fahrenheit": "°F", "화씨": "°F",
    "%": "%", "percent": "%", "퍼센트": "%",
}

_CURRENCY_ALIASES = {
    "₩": "KRW", "원": "KRW", "원화": "KRW", "won": "KRW", "krw": "KRW",
    "$": "USD", "달러": "USD", "dollar": "USD", "dollars": "USD", "usd": "USD",
    "€": "EUR", "유로": "EUR", "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "£": "GBP", "파운드": "GBP", "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
}


def _decimal(raw: str) -> str | None:
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _add(spans: list[Span], occupied: list[tuple[int, int]], span: Span) -> None:
    if span.start >= span.end or any(span.start < end and start < span.end
                                     for start, end in occupied):
        return
    spans.append(span)
    occupied.append((span.start, span.end))


def _digit_values(text: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    for match in re.finditer(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])", text):
        canonical = _decimal(match.group())
        if canonical is not None:
            _add(spans, occupied, Span("number", match.group(), match.start(), match.end(), canonical))


def _dates(text: str, lang: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    patterns = [
        re.compile(r"(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})"),
        re.compile(r"(?P<y>\d{4})년\s*(?P<m>\d{1,2})월\s*(?P<d>\d{1,2})일"),
    ]
    if lang == "en":
        patterns.append(re.compile(r"(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4})"))
    for pattern in patterns:
        for match in pattern.finditer(text):
            y, m, d = int(match["y"]), int(match["m"]), int(match["d"])
            if 1 <= m <= 12 and 1 <= d <= 31:
                _add(spans, occupied, Span("date", match.group(), match.start(), match.end(),
                                           f"{y:04d}-{m:02d}-{d:02d}"))


def _times(text: str, lang: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    digit = re.compile(r"(?<!\d)(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<p>am|pm)?\b", re.I)
    for match in digit.finditer(text):
        hour, minute = int(match["h"]), int(match["m"])
        marker = (match["p"] or "").lower()
        if marker and 1 <= hour <= 12:
            hour = (hour % 12) + (12 if marker == "pm" else 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            _add(spans, occupied, Span("time", match.group(), match.start(), match.end(),
                                       f"{hour:02d}:{minute:02d}"))
    meridiem = re.compile(r"(?<!\d)(?P<h>\d{1,2})\s*(?P<p>a\.?m\.?|p\.?m\.?)(?![A-Za-z])", re.I)
    for match in meridiem.finditer(text):
        hour = int(match["h"])
        marker = match["p"].lower().replace(".", "")
        if not 1 <= hour <= 12:
            continue
        hour = (hour % 12) + (12 if marker == "pm" else 0)
        _add(spans, occupied, Span("time", match.group(), match.start(), match.end(),
                                   f"{hour:02d}:00"))
    if lang == "ko":
        pattern = re.compile(r"(?:(?P<p>오전|오후)\s*)?(?P<h>\d{1,2}|[가-힣]+)\s*시(?:\s*(?P<m>\d{1,2})\s*분)?")
        for match in pattern.finditer(text):
            raw_hour = match["h"]
            hour = int(raw_hour) if raw_hour.isdigit() else _KO_NUMBERS.get(raw_hour)
            minute = int(match["m"] or 0)
            if hour is None or not 0 <= minute <= 59:
                continue
            if match["p"] == "오후" and hour < 12:
                hour += 12
            if match["p"] == "오전" and hour == 12:
                hour = 0
            if 0 <= hour <= 23:
                _add(spans, occupied, Span("time", match.group(), match.start(), match.end(),
                                           f"{hour:02d}:{minute:02d}"))
    else:
        words = "|".join(sorted(_EN_NUMBERS, key=len, reverse=True))
        pattern = re.compile(rf"\b(?P<h>{words})\s*(?P<p>a\.?m\.?|p\.?m\.?|o'clock)\b", re.I)
        for match in pattern.finditer(text):
            hour = _EN_NUMBERS[match["h"].lower()]
            marker = match["p"].lower().replace(".", "")
            if marker == "pm" and hour < 12:
                hour += 12
            if marker == "am" and hour == 12:
                hour = 0
            _add(spans, occupied, Span("time", match.group(), match.start(), match.end(),
                                       f"{hour:02d}:00"))


def _money_and_units(text: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    currencies = "|".join(re.escape(v) for v in sorted(_CURRENCY_ALIASES, key=len, reverse=True))
    money = re.compile(rf"(?:(?P<pre>{currencies})\s*)?(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<post>{currencies})?", re.I)
    for match in money.finditer(text):
        currency = match["pre"] or match["post"]
        if not currency:
            continue
        amount = _decimal(match["n"])
        code = _CURRENCY_ALIASES[currency.casefold()]
        _add(spans, occupied, Span("money", match.group(), match.start(), match.end(),
                                   f"{code}:{amount}"))
    units = "|".join(re.escape(v) for v in sorted(_UNIT_ALIASES, key=len, reverse=True))
    pattern = re.compile(rf"(?P<n>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?P<u>{units})(?![A-Za-z가-힣])", re.I)
    for match in pattern.finditer(text):
        number = _decimal(match["n"])
        unit = _UNIT_ALIASES[match["u"].casefold()]
        _add(spans, occupied, Span("unit", match.group(), match.start(), match.end(),
                                   f"{number} {unit}"))


def _phones(text: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    pattern = re.compile(r"(?<!\d)(?:\+?\d{1,3}[- .])?(?:\d{2,3}[- .])\d{3,4}[- .]\d{4}(?!\d)")
    for match in pattern.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 9 <= len(digits) <= 15:
            _add(spans, occupied, Span("phone", match.group(), match.start(), match.end(), digits))


def _ordinals(text: str, lang: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    if lang == "en":
        names = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                 "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
        pattern = re.compile(r"\b(?:\d+(?:st|nd|rd|th)|" + "|".join(names) + r")\b", re.I)
        for match in pattern.finditer(text):
            raw = match.group().lower()
            value = names.get(raw) or int(re.match(r"\d+", raw).group())
            _add(spans, occupied, Span("ordinal", match.group(), match.start(), match.end(), str(value)))
    else:
        pattern = re.compile(r"(?:\d+|첫|두|세|네)\s*번(?:째)?")
        for match in pattern.finditer(text):
            head = re.match(r"\d+|첫|두|세|네", match.group()).group()
            value = int(head) if head.isdigit() else {"첫": 1, "두": 2, "세": 3, "네": 4}[head]
            _add(spans, occupied, Span("ordinal", match.group(), match.start(), match.end(), str(value)))


def _word_numbers(text: str, lang: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    table = _KO_NUMBERS if lang == "ko" else _EN_NUMBERS
    boundary = r"(?<![가-힣])({})(?![가-힣])" if lang == "ko" else r"\b({})\b"
    pattern = re.compile(boundary.format("|".join(re.escape(v) for v in sorted(table, key=len, reverse=True))),
                         0 if lang == "ko" else re.I)
    for match in pattern.finditer(text):
        raw = match.group()
        value = table.get(raw if lang == "ko" else raw.lower())
        _add(spans, occupied, Span("number", raw, match.start(), match.end(), str(value)))


def extract_critical_values(text: str, lang: str) -> list[dict]:
    if lang not in {"ko", "en"}:
        raise ValueError("critical-information extraction supports only ko and en")
    spans: list[Span] = []
    occupied: list[tuple[int, int]] = []
    _phones(text, spans, occupied)
    _dates(text, lang, spans, occupied)
    _times(text, lang, spans, occupied)
    _money_and_units(text, spans, occupied)
    _ordinals(text, lang, spans, occupied)
    _digit_values(text, spans, occupied)
    _word_numbers(text, lang, spans, occupied)
    return [span.as_dict() for span in sorted(spans, key=lambda value: value.start)]


def _validated_span(text: str, raw: dict, allowed_types: set[str]) -> dict | None:
    kind = str(raw.get("type") or "").strip().lower()
    surface = str(raw.get("text") or "")
    canonical = str(raw.get("canonical_value") or "").strip()
    if kind not in allowed_types or not surface or not canonical:
        return None
    start, end = raw.get("start"), raw.get("end")
    valid = (isinstance(start, int) and isinstance(end, int)
             and 0 <= start < end <= len(text) and text[start:end] == surface)
    if not valid:
        matches = [match.start() for match in re.finditer(re.escape(surface), text)]
        if len(matches) != 1:
            return None
        start, end = matches[0], matches[0] + len(surface)
    accepted = [str(value) for value in raw.get("accepted_values") or [] if str(value).strip()]
    return {"type": kind, "text": surface, "start": start, "end": end,
            "canonical_value": canonical, "accepted_values": accepted}


def _merge_spans(primary: list[dict], secondary: list[dict]) -> list[dict]:
    output = [dict(value) for value in primary]
    occupied = [(value["start"], value["end"]) for value in output
                if isinstance(value.get("start"), int) and isinstance(value.get("end"), int)]
    identities = {(str(value.get("type") or "").lower(),
                   str(value.get("canonical_value") or "").casefold()) for value in output}
    for value in secondary:
        identity = (str(value.get("type") or "").lower(),
                    str(value.get("canonical_value") or "").casefold())
        if identity in identities:
            existing = next(item for item in output
                            if (str(item.get("type") or "").lower(),
                                str(item.get("canonical_value") or "").casefold()) == identity)
            for key in ("text", "start", "end"):
                if existing.get(key) is None and value.get(key) is not None:
                    existing[key] = value[key]
            accepted = list(dict.fromkeys((existing.get("accepted_values") or [])
                                          + (value.get("accepted_values") or [])))
            if accepted:
                existing["accepted_values"] = accepted
            continue
        located = (isinstance(value.get("start"), int)
                   and isinstance(value.get("end"), int))
        if located and any(value["start"] < end and start < value["end"]
                           for start, end in occupied):
            continue
        output.append(value)
        identities.add(identity)
        if located:
            occupied.append((value["start"], value["end"]))
    return sorted(output, key=lambda value: (value.get("start", len(output) + 1),
                                             value.get("end", len(output) + 1)))


async def annotate_critical_information(*, judge: JsonJudge, source: str,
                                        reference: str, candidate: str,
                                        source_lang: str, target_lang: str,
                                        gold_reference_spans: list[dict] | None = None) -> dict:
    if {source_lang, target_lang} != {"ko", "en"}:
        raise ValueError("critical-information annotation supports only ko<->en")
    deterministic_reference = extract_critical_values(reference, target_lang)
    deterministic_candidate = extract_critical_values(candidate, target_lang)
    supplied_gold = [dict(value) for value in gold_reference_spans or []]
    reference_seed = _merge_spans(supplied_gold, deterministic_reference)
    result = await judge.ask(
        purpose="critical_information",
        system=CRITICAL_SYSTEM_PROMPT,
        payload={
            "source_language": source_lang,
            "target_language": target_lang,
            "source": source,
            "reference": reference,
            "candidate": candidate,
            "known_reference_spans": reference_seed,
            "known_candidate_value_spans": deterministic_candidate,
        },
    )
    types = {"person", "location", "organization", "product", "term", "address", "email", "url"}
    llm_reference = [value for raw in result.get("reference_spans") or []
                     if isinstance(raw, dict)
                     for value in [_validated_span(reference, raw, types)] if value]
    llm_candidate = [value for raw in result.get("candidate_spans") or []
                     if isinstance(raw, dict)
                     for value in [_validated_span(candidate, raw, types)] if value]
    reference_spans = _merge_spans(reference_seed, llm_reference)
    candidate_spans = _merge_spans(deterministic_candidate, llm_candidate)
    return {
        "reference_spans": reference_spans,
        "candidate_spans": candidate_spans,
        "alignment": result.get("alignments") or [],
        "annotation_source": "human_gold+automatic" if supplied_gold else "automatic",
        "annotator_version": QUALITY_ANNOTATOR_VERSION,
        "prompt_version": CRITICAL_PROMPT_VERSION,
        "judge_model": judge.model,
        "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    }


def target_token_count(text: str, lang: str) -> int:
    if lang == "ko":
        return len(re.findall(r"\S+", text))
    return len(re.findall(r"\b\w+(?:['’-]\w+)*\b", text, re.UNICODE))


async def annotate_fluency(*, judge: JsonJudge, candidate: str,
                           target_lang: str, previous_turns: list[str] | None = None) -> dict:
    result = await judge.ask(
        purpose="spoken_fluency_mqm",
        system=FLUENCY_SYSTEM_PROMPT,
        payload={
            "target_language": target_lang,
            "preceding_target_turns": list(previous_turns or [])[-3:],
            "candidate": candidate,
        },
    )
    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= float(score) <= 5:
        raise ValueError("fluency judge returned a score outside [1, 5]")
    allowed = {"grammar", "word_order", "word_form", "spelling", "punctuation",
               "register", "awkwardness", "repetition", "untranslated_fragment", "consistency"}
    errors = []
    for raw in result.get("errors") or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("category") or "").lower()
        surface = str(raw.get("text") or "")
        start, end = raw.get("start"), raw.get("end")
        if kind not in allowed or not surface:
            continue
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(candidate) and candidate[start:end] == surface):
            matches = [match.start() for match in re.finditer(re.escape(surface), candidate)]
            if len(matches) != 1:
                continue
            start, end = matches[0], matches[0] + len(surface)
        severity = str(raw.get("severity") or "minor").lower()
        if severity not in {"minor", "major", "critical"}:
            severity = "minor"
        errors.append({"category": kind, "severity": severity, "start": start, "end": end,
                       "text": surface, "explanation": str(raw.get("explanation") or "")})
    return {
        "judge": {"score": float(score), "reason": str(result.get("reason") or "")},
        "mqm_errors": errors,
        "target_token_count": target_token_count(candidate, target_lang),
        "target_lang": target_lang,
        "annotator_version": QUALITY_ANNOTATOR_VERSION,
        "prompt_version": FLUENCY_PROMPT_VERSION,
        "judge_model": judge.model,
        "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    }
