"""Supported language codes, and the facts that follow from them.

A fixed table rather than `langcodes`, deliberately. Callers here validate against
a supported set and have to reject what falls outside it. The open-ended case --
accepting whatever target language someone typed -- is autoseg's `to_lang_code`,
which uses `langcodes` for exactly that reason (35a8108).
"""
import re

CODE_TO_NAME = {
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "hi": "Hindi",
}

NAME_TO_CODE = {v.lower(): k for k, v in CODE_TO_NAME.items()}

# Scripts written without spaces between words.
CHAR_UNIT_LANGS = {"zh", "ja", "th"}


def get_lang_name(code: str | None) -> str:
    return CODE_TO_NAME.get(code or "", code or "the source language")


def norm_code(value: str) -> str:
    """A code or a name to its code. `""` for anything unrecognised, `auto` included."""
    if not isinstance(value, str):
        return ""
    v = value.strip()
    if not v or v == "auto":
        return ""
    if v in CODE_TO_NAME:
        return v
    return NAME_TO_CODE.get(v.lower(), "")


def parse_map(raw: object) -> dict[str, str]:
    """`{"ko": "en", "ja": "ko"}`: where each detected language is translated to.

    Unknown codes and entries that point at themselves are dropped, so what comes back
    only ever names a real direction.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for source, target in raw.items():
        src, dst = norm_code(source), norm_code(target)
        if src and dst and src != dst:
            out[src] = dst
    return out


def pick_target(detected: str, *, lang: str = "", target_lang: str = "",
                lang_map: dict[str, str] | None = None) -> str:
    """The language a segment spoken in `detected` is translated into.

    `lang_map` wins. Otherwise it is the pair rule between a speaker of `lang` and a
    listener of `target_lang`: the speaker's own language goes to the listener, and
    anything else comes back to the speaker. An unknown `detected` goes to the
    listener. `""` means there is nowhere to translate to.
    """
    code = norm_code(detected)
    if lang_map and code in lang_map:
        return lang_map[code]
    if not lang or not code or code == lang:
        return target_lang
    return lang


def laal_unit(code: str) -> str:
    """Latency counts characters where the script has no word boundaries."""
    return "char" if code in CHAR_UNIT_LANGS else "word"


# Languages whose letters are unique enough that seeing them at all identifies the
# language -- unlike the Latin alphabet, which English, Spanish, German, ... all share.
# ("Alphabet" here is shorthand for "set of letters a language is written with" --
# Japanese and Chinese are not alphabets in the strict sense, but kana and Han
# characters are exactly as identifying as Hangul is for Korean.) `detect_by_alphabet`
# can never return a Latin-alphabet language because of this: there is no pattern for
# one to match.
ALPHABET_LANGS: tuple[str, ...] = ("ko", "ja", "zh")
ALPHABET_THRESHOLD = 0.6

_ALPHABETS: dict[str, re.Pattern] = {
    "ko": re.compile(r"[가-힣]"),
    "ja": re.compile(r"[぀-ヿ]"),
    "zh": re.compile(r"[一-鿿]"),
}


def alphabet_ratio(text: str, lang: str) -> float:
    """The share of `text`'s letters written in `lang`'s alphabet. `0.0` for an unknown lang."""
    pattern = _ALPHABETS.get(lang)
    if pattern is None:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if pattern.match(c)) / len(letters)


def detect_by_alphabet(text: str, *, langs: tuple[str, ...] = ALPHABET_LANGS,
                       threshold: float = ALPHABET_THRESHOLD) -> str | None:
    """The language `text` is written in, judged by its letters alone.

    `langs` is checked in order, and each letter counts toward the first language in
    `langs` whose alphabet it belongs to -- a Japanese sentence is mostly kanji plus some
    kana, so `ja` has to be tried before `zh` or every Japanese sentence would look
    Chinese. Returns the earliest language in `langs` whose share of the text's letters
    reaches `threshold`, or `None` if nothing does, there are no letters, or `text` is
    written in an alphabet this function does not know (including any Latin one).
    """
    if not text:
        return None
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    counts = dict.fromkeys(langs, 0)
    for c in letters:
        for lang in langs:
            pattern = _ALPHABETS.get(lang)
            if pattern is not None and pattern.match(c):
                counts[lang] += 1
                break
    total = len(letters)
    for lang in langs:
        if counts[lang] / total >= threshold:
            return lang
    return None
