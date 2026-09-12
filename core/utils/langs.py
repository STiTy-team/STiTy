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


def laal_unit(code: str) -> str:
    """Latency counts characters where the script has no word boundaries."""
    return "char" if code in CHAR_UNIT_LANGS else "word"
