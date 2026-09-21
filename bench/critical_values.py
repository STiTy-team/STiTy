"""Rule-based extraction of numeric critical values with language-neutral canonical forms.

Numbers, ordinals, dates, clock times, money, measurements and phone numbers. The
reference translation and the candidate are both in the target language and go
through the same rules, so a blind spot of a rule is shared by both sides. What must
not happen is a rule firing on words that are not numbers at all -- a Korean "네"
(yes) read as 4, the "19" of "COVID-19", the pronoun "one" -- because each of those
turns a correct translation into a critical failure. Forms that are also ordinary
words are therefore read as numbers only next to a counter, unit or currency.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


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


def _alternation(words) -> str:
    return "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))


def _canonical(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


_NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"

_EN_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9}
_EN_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
             "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
             "nineteen": 19}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90}
_EN_SCALES = {"hundred": 100, "thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}
_EN_HOURS = {**{k: v for k, v in _EN_UNITS.items() if v}, "ten": 10, "eleven": 11,
             "twelve": 12}

_KO_SINO_DIGITS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7,
                   "팔": 8, "구": 9}
_KO_SMALL = {"십": 10, "백": 100, "천": 1000}
_KO_BIG = {"만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12}
_KO_NATIVE_TENS = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40, "쉰": 50,
                   "예순": 60, "일흔": 70, "여든": 80, "아흔": 90}
_KO_NATIVE_UNITS = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "넷": 4,
                    "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
# Determiner forms ("두 개") are also everyday words ("네" yes, "한국", "세상"), and
# 열/쉰 are also verbs/adjectives; they count only in front of a counter.
_KO_NEEDS_COUNTER = {"한", "두", "세", "네", "스무", "열", "쉰"}
_KO_HOURS = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7,
             "여덟": 8, "아홉": 9, "열": 10, "열한": 11, "열두": 12}
_KO_COUNTERS = {
    "개", "명", "분", "번", "시간", "살", "마리", "잔", "권", "장", "대", "달", "가지",
    "사람", "병", "그릇", "쪽", "곳", "군데", "벌", "켤레", "송이", "척", "채", "통",
    "봉지", "줄", "판", "끼", "개국", "바퀴", "배", "초", "년", "월", "일", "층", "호",
    "개월", "주", "주일", "주년", "인분", "호선", "박", "회", "위",
}
_KO_PARTICLES = {
    "을", "를", "이", "가", "은", "는", "에", "의", "도", "만", "씩", "째", "로", "으로",
    "와", "과", "랑", "이랑", "하고", "이나", "나", "쯤", "정도", "가량", "이상", "이하",
    "이내", "까지", "부터", "밖에", "뿐", "동안", "전", "후", "반", "짜리", "마다",
    "보다", "처럼", "에게", "한테", "요", "이요", "입니다", "이에요", "예요", "이고",
}
# What may follow 시/분 of a clock time: a particle or a copula ending, never the
# rest of a word such as 시간 (hours), 시작, 시장.
_KO_TIME_TAIL = _KO_PARTICLES | {
    "경", "께", "정각", "야", "였", "이후", "이전", "사이", "즈음", "다", "엔", "라",
    "면", "지", "죠", "니", "인가", "인데", "인지", "일",
}

_UNIT_ALIASES = {
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km",
    "kilometres": "km", "킬로미터": "km",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m", "미터": "m",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "센티미터": "cm", "센티": "cm",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "밀리미터": "mm",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg", "킬로그램": "kg",
    "g": "g", "gram": "g", "grams": "g", "그램": "g",
    "l": "L", "liter": "L", "liters": "L", "litre": "L", "litres": "L", "리터": "L",
    "ml": "mL", "milliliter": "mL", "milliliters": "mL", "밀리리터": "mL",
    "°c": "°C", "celsius": "°C", "섭씨": "°C",
    "°f": "°F", "fahrenheit": "°F", "화씨": "°F",
    "%": "%", "percent": "%", "per cent": "%", "퍼센트": "%", "프로": "%",
}
_CURRENCY_ALIASES = {
    "₩": "KRW", "원": "KRW", "원화": "KRW", "won": "KRW", "krw": "KRW",
    "$": "USD", "us$": "USD", "달러": "USD", "dollar": "USD", "dollars": "USD", "usd": "USD",
    "€": "EUR", "유로": "EUR", "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "£": "GBP", "파운드": "GBP", "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
    "yen": "JPY", "jpy": "JPY", "엔화": "JPY",
}


def _is_latin(word: str) -> bool:
    return bool(re.fullmatch(r"[a-z ]+", word))


# Latin unit and currency words must end a word ("5 m" but not "5 more"); Hangul and
# symbols may be followed by a particle ("70킬로미터를", "3만 원이").
_UNIT_AFTER = re.compile(
    rf"\s?(?P<u>(?:{_alternation(w for w in _UNIT_ALIASES if _is_latin(w))})(?![A-Za-z])"
    rf"|{_alternation(w for w in _UNIT_ALIASES if not _is_latin(w))})", re.I)
_CURRENCY_AFTER = re.compile(
    rf"\s?(?P<c>(?:{_alternation(w for w in _CURRENCY_ALIASES if _is_latin(w))})(?![A-Za-z])"
    rf"|(?:{_alternation(w for w in _CURRENCY_ALIASES if re.fullmatch('[가-힣]+', w))}))",
    re.I)
# A digit run ends at a word boundary, or runs straight into a unit ("35mm").
_DIGITS_END = (rf"(?:(?![\w])|(?=(?:{_alternation(w for w in _UNIT_ALIASES if _is_latin(w))})"
               rf"(?![A-Za-z])))")
_CURRENCY_BEFORE = re.compile(r"(?P<c>us\$|[$₩€£]|(?<![A-Za-z])(?:usd|krw|eur|gbp|jpy))\s?$",
                              re.I)
_KO_COUNTER_AFTER = re.compile(
    rf"(?P<gap>\s?)(?:{_alternation(_KO_COUNTERS)})"
    rf"(?=$|[^가-힣]|(?:{_alternation(_KO_PARTICLES)}))")

_EN_DIGITS = re.compile(
    rf"(?:(?<![^\s(])[-+](?=\d)|(?<![\w.])(?<![^\W\d_]-))(?P<n>{_NUMBER}){_DIGITS_END}"
    rf"(?:\s+(?P<scale>{_alternation(_EN_SCALES)})\b)?", re.I)
_EN_WORD = _alternation([*_EN_UNITS, *_EN_TEENS, *_EN_TENS, *_EN_SCALES])
_EN_WORDS = re.compile(
    rf"\b(?:a\s+(?=(?:{_alternation(_EN_SCALES)})\b))?(?:{_EN_WORD})\b"
    rf"(?:(?:[\s-]+|\s+and\s+)(?:{_EN_WORD})\b)*", re.I)
_KO_TOKEN = rf"(?:{_NUMBER}|[{''.join(_KO_SINO_DIGITS)}{''.join(_KO_SMALL)}{''.join(_KO_BIG)}])"
# A scale may be spaced from its number ("6 만 원") and a new group may follow a big
# scale after a space ("1억 2천만").
# A spaced scale must end the word or run into another scale, a currency or a counter,
# so "3 백화점" stays 3.
_KO_SCALE = f"[{''.join(_KO_SMALL)}{''.join(_KO_BIG)}]"
_KO_SCALE_END = (rf"(?=$|[^가-힣]|{_KO_SCALE}|(?:"
                 rf"{_alternation([w for w in _CURRENCY_ALIASES if re.fullmatch('[가-힣]+', w)] + list(_KO_COUNTERS))}))")
_KO_SINO = re.compile(
    rf"(?<![\w.])(?<![^\W\d_]-){_KO_TOKEN}"
    rf"(?:{_KO_TOKEN}|\s{_KO_SCALE}{_KO_SCALE_END}|(?<=[{''.join(_KO_BIG)}])\s{_KO_TOKEN})*"
    rf"(?:(?![A-Za-z])|(?=(?:{_alternation(w for w in _UNIT_ALIASES if _is_latin(w))})"
    rf"(?![A-Za-z])))", re.I)
_KO_NATIVE = re.compile(
    rf"(?<![가-힣])(?:(?P<tens>{_alternation(_KO_NATIVE_TENS)})"
    rf"(?:\s?(?P<unit>{_alternation(_KO_NATIVE_UNITS)}))?"
    rf"|(?P<solo>{_alternation(_KO_NATIVE_UNITS)}))")


@dataclass(frozen=True)
class _Quantity:
    start: int
    end: int
    value: Decimal
    # "none": a number by itself; "counter": only before a counter, unit or currency;
    # "spaced_counter": the same, with exactly one space between (single-syllable
    # Sino numerals such as "오 분", which written together are words: "사원");
    # "boundary": a number when the word ends there or a counter follows;
    # "measure": only before a unit or currency (the English pronoun "one").
    needs: str = "none"


def _en_category(word: str) -> str:
    if word in ("a", "and"):
        return word
    if word in _EN_UNITS:
        return "unit"
    if word in _EN_TEENS:
        return "teen"
    if word in _EN_TENS:
        return "tens"
    return "hundred" if word == "hundred" else "big"


_EN_FOLLOWS = {
    None: {"unit", "teen", "tens", "a"},
    "a": {"hundred", "big"},
    "unit": {"hundred", "big"},
    "teen": {"hundred", "big"},
    "tens": {"unit", "hundred", "big"},
    "hundred": {"unit", "teen", "tens", "big", "and"},
    "big": {"unit", "teen", "tens", "and"},
    "and": {"unit", "teen", "tens"},
}


def _en_word_quantities(text: str) -> list[_Quantity]:
    """Split each run of number words into well-formed numbers ("two three" is two)."""
    output = []
    for match in _EN_WORDS.finditer(text):
        group = None
        for token in re.finditer(r"[A-Za-z]+", match.group()):
            word = token.group().lower()
            start = match.start() + token.start()
            end = match.start() + token.end()
            category = _en_category(word)
            if group is None or category not in _EN_FOLLOWS[group["last"]]:
                if group is not None:
                    output.append(group)
                group = None
                if category not in _EN_FOLLOWS[None]:
                    continue
                group = {"start": start, "end": end, "total": 0, "current": 0,
                         "last": None, "words": []}
            group["last"] = category
            if category == "and":
                continue
            group["end"] = end
            group["words"].append(word)
            if category == "a":
                group["current"] = 1
            elif category == "hundred":
                group["current"] = (group["current"] or 1) * 100
            elif category == "big":
                group["total"] += (group["current"] or 1) * _EN_SCALES[word]
                group["current"] = 0
            else:
                group["current"] += {**_EN_UNITS, **_EN_TEENS, **_EN_TENS}[word]
        if group is not None:
            output.append(group)
    return [_Quantity(g["start"], g["end"], Decimal(g["total"] + g["current"]),
                      "measure" if g["words"] == ["one"] else "none")
            for g in output if g["words"] and g["words"] != ["a"]]


def _en_digit_quantities(text: str) -> list[_Quantity]:
    output = []
    for match in _EN_DIGITS.finditer(text):
        value = _decimal(match["n"])
        if value is None:
            continue
        if match.group().startswith("-"):
            value = -value
        if match["scale"]:
            value *= _EN_SCALES[match["scale"].lower()]
        output.append(_Quantity(match.start(), match.end(), value))
    return output


def _ko_value(expression: str) -> Decimal | None:
    total = section = Decimal(0)
    current = None
    for token in re.finditer(rf"{_NUMBER}|\S", expression):
        raw = token.group()
        if raw[0].isdigit() or raw in _KO_SINO_DIGITS:
            if current is not None:
                return None
            current = _decimal(raw) if raw[0].isdigit() else Decimal(_KO_SINO_DIGITS[raw])
            if current is None:
                return None
        elif raw in _KO_SMALL:
            section += (current if current is not None else 1) * _KO_SMALL[raw]
            current = None
        elif raw in _KO_BIG:
            base = section + (current or 0)
            total += (base if base else 1) * _KO_BIG[raw]
            section, current = Decimal(0), None
        else:
            return None
    return total + section + (current or 0)


def _ko_sino_quantities(text: str) -> list[_Quantity]:
    output = []
    for match in _KO_SINO.finditer(text):
        expression = match.group()
        value = _ko_value(expression)
        if value is None:
            continue
        if any(char.isdigit() for char in expression):
            needs = "none"
        elif len(expression) == 1:
            # "이" is far more often "this" (이 분, 이 번) than two.
            if expression == "이":
                continue
            needs = "spaced_counter"
        else:
            needs = "counter"
        output.append(_Quantity(match.start(), match.end(), value, needs))
    return output


def _ko_native_quantities(text: str) -> list[_Quantity]:
    output = []
    for match in _KO_NATIVE.finditer(text):
        last = match["solo"] or match["unit"] or match["tens"]
        value = (_KO_NATIVE_TENS.get(match["tens"] or "", 0)
                 + _KO_NATIVE_UNITS.get(match["unit"] or match["solo"] or "", 0))
        needs = "counter" if last in _KO_NEEDS_COUNTER else "boundary"
        output.append(_Quantity(match.start(), match.end(), Decimal(value), needs))
    return output


def _add(spans: list[Span], occupied: list[tuple[int, int]], span: Span) -> bool:
    if span.start >= span.end or any(span.start < end and start < span.end
                                     for start, end in occupied):
        return False
    spans.append(span)
    occupied.append((span.start, span.end))
    return True


def _quantities(text: str, lang: str, spans: list[Span],
                occupied: list[tuple[int, int]]) -> None:
    found = (_en_digit_quantities(text) + _en_word_quantities(text) if lang == "en"
             else _ko_native_quantities(text) + _ko_sino_quantities(text))
    for quantity in sorted(found, key=lambda q: (q.start, q.start - q.end)):
        start, end, number = quantity.start, quantity.end, _canonical(quantity.value)
        spaced = quantity.needs == "spaced_counter"
        if spaced and not text.startswith(" ", end):
            continue
        before = _CURRENCY_BEFORE.search(text[:start])
        after = _CURRENCY_AFTER.match(text, end)
        if before or after:
            currency = (before or after)["c"].casefold()
            span_start = before.start() if before else start
            span_end = after.end() if after and not before else end
            _add(spans, occupied, Span("money", text[span_start:span_end], span_start,
                                       span_end, f"{_CURRENCY_ALIASES[currency]}:{number}"))
            continue
        unit = _UNIT_AFTER.match(text, end)
        if unit:
            _add(spans, occupied, Span("unit", text[start:unit.end()], start, unit.end(),
                                       f"{number} {_UNIT_ALIASES[unit['u'].casefold()]}"))
            continue
        if quantity.needs == "measure":
            continue
        if quantity.needs != "none":
            counter = _KO_COUNTER_AFTER.match(text, end)
            if counter and spaced and counter["gap"] != " ":
                counter = None
            ends_word = end == len(text) or not re.match(r"[가-힣]", text[end])
            if not counter and not (quantity.needs == "boundary" and ends_word):
                continue
        _add(spans, occupied, Span("number", text[start:end], start, end, number))


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


# "p.m." keeps its closing dot; an undotted "PM." leaves the sentence's full stop alone.
_MERIDIEM = r"(?:[ap]\.\s?m\b\.?|[ap]\s?m\b)"


def _meridiem_hour(hour: int, marker: str) -> int:
    marker = re.sub(r"[.\s]", "", marker.lower())
    if marker == "pm":
        return hour % 12 + 12
    if marker == "am":
        return hour % 12
    return hour


def _times(text: str, lang: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    clock = re.compile(rf"(?<![\d:])(?P<h>\d{{1,2}}):(?P<m>\d{{2}})(?![\d:])"
                       rf"(?:\s?(?P<p>{_MERIDIEM}))?", re.I)
    for match in clock.finditer(text):
        hour, minute = int(match["h"]), int(match["m"])
        if match["p"]:
            if not 1 <= hour <= 12:
                continue
            hour = _meridiem_hour(hour, match["p"])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            _add(spans, occupied, Span("time", match.group(), match.start(), match.end(),
                                       f"{hour:02d}:{minute:02d}"))
    meridiem = re.compile(rf"(?<![\d:.])(?P<h>\d{{1,2}})\s?(?P<p>{_MERIDIEM})", re.I)
    for match in meridiem.finditer(text):
        hour = int(match["h"])
        if 1 <= hour <= 12:
            _add(spans, occupied, Span("time", match.group(), match.start(), match.end(),
                                       f"{_meridiem_hour(hour, match['p']):02d}:00"))
    if lang == "ko":
        pattern = re.compile(
            rf"(?:(?<![가-힣])(?P<p>오전|오후|새벽|아침|낮|저녁|밤)\s?|(?<![가-힣]))"
            rf"(?P<h>\d{{1,2}}|{_alternation(_KO_HOURS)})\s?시"
            rf"(?:\s?(?P<m>\d{{1,2}})\s?분|\s?(?P<half>반))?"
            rf"(?=$|[^가-힣]|(?:{_alternation(_KO_TIME_TAIL)}))")
        for match in pattern.finditer(text):
            raw = match["h"]
            hour = int(raw) if raw.isdigit() else _KO_HOURS[raw]
            minute = 30 if match["half"] else int(match["m"] or 0)
            period = match["p"]
            if period in ("오후", "저녁") and hour < 12:
                hour += 12
            elif period == "낮" and hour < 6:
                hour += 12
            elif period == "밤" and 6 <= hour < 12:
                hour += 12
            elif period in ("밤", "오전", "새벽", "아침") and hour == 12:
                hour = 0
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                _add(spans, occupied, Span("time", match.group(), match.start(), match.end(),
                                           f"{hour:02d}:{minute:02d}"))
    else:
        pattern = re.compile(
            rf"\b(?P<h>{_alternation(_EN_HOURS)})\s?(?P<p>{_MERIDIEM}|o['’]clock\b)", re.I)
        for match in pattern.finditer(text):
            hour = _meridiem_hour(_EN_HOURS[match["h"].lower()], match["p"])
            _add(spans, occupied, Span("time", match.group(), match.start(), match.end(),
                                       f"{hour:02d}:00"))


def _phones(text: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    pattern = re.compile(r"(?<!\d)(?:\+?\d{1,3}[- .])?(?:\d{2,3}[- .])\d{3,4}[- .]\d{4}(?!\d)")
    for match in pattern.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 9 <= len(digits) <= 15:
            _add(spans, occupied, Span("phone", match.group(), match.start(), match.end(), digits))


_EN_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
_KO_ORDINAL_HEADS = {"첫": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7,
                     "여덟": 8, "아홉": 9, "열": 10}
_KO_ORDINAL_NOUNS = {"첫": 1, "둘": 2, "셋": 3, "넷": 4}


def _ordinals(text: str, lang: str, spans: list[Span], occupied: list[tuple[int, int]]) -> None:
    if lang == "en":
        pattern = re.compile(rf"\b(?:\d+(?:st|nd|rd|th)|{_alternation(_EN_ORDINALS)})\b", re.I)
        for match in pattern.finditer(text):
            raw = match.group().lower()
            # "a second", "per second": the unit of time, not the ordinal.
            if raw == "second" and re.search(r"\b(?:a|one|per|split)\s+$",
                                             text[:match.start()], re.I):
                continue
            value = _EN_ORDINALS.get(raw) or int(re.match(r"\d+", raw).group())
            _add(spans, occupied, Span("ordinal", match.group(), match.start(), match.end(),
                                       str(value)))
    else:
        # "두 번" is twice, not second; only 번째/째/제N are ordinals.
        pattern = re.compile(
            rf"(?<![가-힣])(?:(?P<head>\d+|{_alternation(_KO_ORDINAL_HEADS)})\s?번째"
            rf"|(?P<noun>{_alternation(_KO_ORDINAL_NOUNS)})째|제\s?(?P<je>\d+))")
        for match in pattern.finditer(text):
            head = match["head"] or match["je"]
            if head:
                value = int(head) if head.isdigit() else _KO_ORDINAL_HEADS[head]
            else:
                value = _KO_ORDINAL_NOUNS[match["noun"]]
            _add(spans, occupied, Span("ordinal", match.group(), match.start(), match.end(),
                                       str(value)))


def extract_critical_values(text: str, lang: str) -> list[dict]:
    if lang not in {"ko", "en"}:
        raise ValueError("critical-information extraction supports only ko and en")
    spans: list[Span] = []
    occupied: list[tuple[int, int]] = []
    _phones(text, spans, occupied)
    _dates(text, lang, spans, occupied)
    _times(text, lang, spans, occupied)
    _ordinals(text, lang, spans, occupied)
    _quantities(text, lang, spans, occupied)
    return [span.as_dict() for span in sorted(spans, key=lambda value: value.start)]


__all__ = ["Span", "extract_critical_values"]
