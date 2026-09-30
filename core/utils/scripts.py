"""Which writing system a piece of text is in, and what that says about its language.

The ASR model labels a whole decode window with one language, so a sentence spoken in
another language inside that window keeps the old label. The letters themselves are a
better witness for most pairs: Hangul is Korean, kana is Japanese, Latin is one of the
Latin-script languages. This module only reads letters; it never guesses between two
languages that share a script -- the caller says which ones are possible.
"""
from collections import Counter

SCRIPT_RANGES = (
    ("hangul", 0xAC00, 0xD7A3),
    ("hangul", 0x1100, 0x11FF),
    ("hangul", 0x3130, 0x318F),
    ("kana", 0x3040, 0x30FF),
    ("han", 0x4E00, 0x9FFF),
    ("han", 0x3400, 0x4DBF),
    ("cyrillic", 0x0400, 0x04FF),
    ("arabic", 0x0600, 0x06FF),
    ("thai", 0x0E00, 0x0E7F),
    ("devanagari", 0x0900, 0x097F),
    ("georgian", 0x10A0, 0x10FF),
)

LANGUAGE_SCRIPT = {
    "ko": "hangul",
    "ja": "kana",
    "zh": "han",
    "ru": "cyrillic",
    "ar": "arabic",
    "th": "thai",
    "hi": "devanagari",
}

SCRIPT_LANGUAGE = {"hangul": "ko", "kana": "ja", "cyrillic": "ru", "arabic": "ar",
                   "thai": "th", "devanagari": "hi"}


def script_of_char(char: str) -> str | None:
    if not char.isalpha():
        return None
    point = ord(char)
    if point < 0x0250:
        return "latin"
    for name, low, high in SCRIPT_RANGES:
        if low <= point <= high:
            return name
    return "other"


def script_counts(text: str) -> Counter:
    return Counter(s for s in map(script_of_char, text or "") if s is not None)


def dominant_script(text: str, *, min_share: float = 0.5) -> str | None:
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return None
    if counts.get("kana"):
        counts["kana"] += counts.pop("han", 0)
    script, count = counts.most_common(1)[0]
    return script if count / total >= min_share else None


def script_for_language(code: str) -> str:
    return LANGUAGE_SCRIPT.get(code, "latin")


def language_for_script(script: str | None, *, candidates: list[str] | tuple = (),
                        fallback: str = "") -> str:
    """The language `script` points at, among `candidates` when they are known.

    Latin and Han are shared by several languages; they resolve only when exactly one
    candidate writes that script, otherwise to `fallback`.
    """
    if script is None:
        return fallback
    candidates = [c for c in candidates if c]
    if script in SCRIPT_LANGUAGE:
        code = SCRIPT_LANGUAGE[script]
        return code if not candidates or code in candidates else fallback
    sharing = [c for c in candidates if script_for_language(c) == script]
    if script == "han" and not sharing and "ja" in candidates:
        sharing = ["ja"]
    if len(sharing) == 1:
        return sharing[0]
    if script == "han" and not candidates:
        return "zh"
    return fallback


def matches_language(text: str, code: str, *, min_share: float = 0.5) -> bool:
    script = dominant_script(text, min_share=min_share)
    if script is None:
        return True
    expected = script_for_language(code)
    if code == "ja":
        return script in ("kana", "han")
    return script == expected


def foreign_share(text: str, code: str) -> float:
    """How much of the letters in `text` are outside the script `code` is written in.

    Latin inside Korean or Japanese text is not counted: names and acronyms are
    written that way on purpose.
    """
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return 0.0
    allowed = {script_for_language(code)}
    if code == "ja":
        allowed |= {"han"}
    if code in ("ko", "ja", "zh"):
        allowed |= {"latin"}
    return sum(n for s, n in counts.items() if s not in allowed) / total
