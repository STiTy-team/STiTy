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


def parse_map(raw: object) -> dict[str, tuple[str, ...]]:
    """`{"ko": "en", "ja": ["ko", "en"]}`: where each detected language is translated to.

    A value may be one code or a list of them; either way the entry becomes the tuple
    of that source's targets, in the order given. Unknown codes are dropped. An entry
    that points at itself (or an empty list) keeps its key with no targets: the source
    stays a language the ASR listens for, and its segments go out transcribed only.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for source, target in raw.items():
        src = norm_code(source)
        raw_targets = target if isinstance(target, (list, tuple)) else [target]
        codes = [norm_code(t) for t in raw_targets]
        targets = tuple(dict.fromkeys(dst for dst in codes if dst and dst != src))
        if src and (targets or src in codes or not raw_targets):
            out[src] = targets
    return out


def pick_targets(detected: str, *, lang: str = "", target_lang: str = "",
                 lang_map: dict[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    """The languages a segment spoken in `detected` is translated into.

    `lang_map` wins, and is the only way to more than one target. Otherwise it is the
    pair rule between a speaker of `lang` and a listener of `target_lang`: the speaker's
    own language goes to the listener, and anything else comes back to the speaker. An
    unknown `detected` goes to the listener. `()` means there is nowhere to translate to.
    """
    code = norm_code(detected)
    if lang_map and code in lang_map:
        return lang_map[code]
    if not lang or not code or code == lang:
        return (target_lang,) if target_lang else ()
    return (lang,)


def pick_target(detected: str, *, lang: str = "", target_lang: str = "",
                lang_map: dict[str, tuple[str, ...]] | None = None) -> str:
    """`pick_targets`' first answer, for callers that translate each segment once."""
    targets = pick_targets(detected, lang=lang, target_lang=target_lang, lang_map=lang_map)
    return targets[0] if targets else ""


def laal_unit(code: str) -> str:
    """Latency counts characters where the script has no word boundaries."""
    return "char" if code in CHAR_UNIT_LANGS else "word"


# Languages whose letters are unique enough that seeing them at all identifies the
# language -- unlike the Latin alphabet, which English, Spanish, German, ... all share.
# ("Alphabet" here is shorthand for "set of letters a language is written with" --
# Japanese and Chinese are not alphabets in the strict sense, but kana and Han
# characters are exactly as identifying as Hangul is for Korean.) `en` matches any
# Latin letter, so it only identifies English when the caller has narrowed `langs` to
# languages where English is the only Latin-script one (e.g. `("ko", "en")`); it is
# left out of the default for that reason.
ALPHABET_LANGS: tuple[str, ...] = ("ko", "ja", "zh")
ALPHABET_THRESHOLD = 0.6

_HANGUL = re.compile(r"[가-힣]")

_ALPHABETS: dict[str, re.Pattern] = {
    "ko": _HANGUL,
    "ja": re.compile(r"[぀-ヿ]"),
    "zh": re.compile(r"[一-鿿]"),
    "en": re.compile(r"[A-Za-z]"),
}


def has_hangul(text: str) -> bool:
    return _HANGUL.search(text) is not None


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
