"""Supported language codes, and the facts that follow from them.

A fixed table rather than `langcodes`, deliberately. Callers here validate against
a supported set and have to reject what falls outside it. The open-ended case --
accepting whatever target language someone typed -- is autoseg's `to_lang_code`,
which uses `langcodes` for exactly that reason (35a8108).
"""

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
