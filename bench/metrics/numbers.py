"""Write numbers one way (Arabic digits) before scoring. Applied to reference and output alike.

References spell Korean numbers out (이십 년, 삼십칠 퍼센트) while models mostly write digits
(20년, 37%), so without this a correct transcript is scored as wrong.

- Sino-Korean numerals: 이십 년 -> 20 년, 사백오십 명 -> 450 명, 이천삼백만원 -> 23000000 원.
  A run of two or more numeral characters is converted when a unit follows it, three or more
  always. A single one (이, 사, 오 ...) only before a unit in `SINGLE_OK`, so ordinary words such
  as 이번, 오늘, 사이 stay as they are.
- Digits followed by 억/만/천/백 become one number: 18만 7천 500 -> 187500. Commas between digits
  are dropped (187,500), so both spellings meet.
- A digit and the Hangul after it are split (20년 -> 20 년) so WER sees the same words.
- Native numerals: 하나, 둘, 셋 ... when they stand alone (열 excluded: 열 나? is "fever"),
  한, 두, 세, 네 ... only before a unit (두 병 -> 2 병; 한 열흘 stays).
- % becomes 퍼센트 next to Hangul and percent elsewhere.
- English number words in Latin-letter spans go through whisper's `EnglishNumberNormalizer`
  (twenty one -> 21), except the interjection "oh", which it would read as 0.
"""

import re

from whisper_normalizer.english import EnglishNumberNormalizer

from core.utils.langs import has_hangul
from core.utils.numerics import (
    AMOUNT,
    SINO_CHARS,
    amount_to_int,
    native_determiner_to_int,
    native_to_int,
    sino_to_int,
)

UNITS = (
    "퍼센트", "인분", "시간", "년", "명", "시", "분", "원", "병", "개", "잔", "살", "번", "층",
    "월", "일", "초", "달", "마리", "등",
)  # fmt: skip
SINGLE_OK = ("년", "명", "퍼센트", "인분", "병", "원", "층")
_english_numbers = EnglishNumberNormalizer()


def normalize_numbers(text: str) -> str:
    if not text:
        return text
    text = "".join(_percent(text, i) if c == "%" else c for i, c in enumerate(text))
    text = re.sub(r"[A-Za-z][A-Za-z0-9'\-– —]*[A-Za-z0-9]|[A-Za-z]", _english, text)
    tokens = text.split()
    out = []
    for i, token in enumerate(tokens):
        core = re.sub(r"^[^\w]+|[^\w]+$", "", token)
        if not core:
            out.append(token)
            continue
        tail = token[len(token.rstrip(".,!?…~")) :]
        following = re.sub(r"^[^\w]+", "", tokens[i + 1]) if i + 1 < len(tokens) else ""
        out.append(_korean_token(core, following) + tail)
    text = " ".join(out)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = AMOUNT.sub(_amount, text)
    text = re.sub(r"(\d)([가-힣])", r"\1 \2", text)
    return " ".join(text.split())


def _percent(text: str, at: int) -> str:
    return " 퍼센트" if has_hangul(text[max(0, at - 12) : at + 12]) else " percent"


def _korean_token(token: str, following: str) -> str:
    m = re.fullmatch(f"([{SINO_CHARS}]+)({'|'.join(UNITS)})?(.*)", token)
    if m:
        numeral, unit, rest = m.groups()
        unit = unit or ""
        single_ok = unit in SINGLE_OK or (not unit and following.startswith(SINGLE_OK))
        before_unit = bool(unit) or following.startswith(UNITS)
        value = sino_to_int(numeral)
        if value is not None and (
            (len(numeral) >= 2 and before_unit) or len(numeral) >= 3 or single_ok
        ):
            return f"{value} {unit}{rest}".strip()
    native = native_to_int(token)
    if native is not None and token != "열":
        return str(native)
    determiner = native_determiner_to_int(token)
    if determiner is not None and following.startswith(UNITS):
        return str(determiner)
    return token


def _amount(m: re.Match) -> str:
    return f"{amount_to_int(m.group(0))} "


def _english(m: re.Match) -> str:
    # A span with no number word is left exactly as written: spacing it out would split a
    # glossed name off its Hangul (키르치네르cristina) and add words that only the reference has.
    words = re.sub(r"[-–—]", " ", m.group(0).lower()).split()
    out, run = [], []
    for word in [*words, None]:
        if word is None or word.strip(".,!?'") == "oh":
            if run:
                out.append(_english_numbers(" ".join(run)))
                run = []
            if word is not None:
                out.append(word)
        else:
            run.append(word)
    if " ".join(out).split() == words:
        return m.group(0)
    return " " + " ".join(out) + " "
