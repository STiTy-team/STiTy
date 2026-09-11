from .errors import BenchConfigError

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

CHAR_UNIT_LANGS = {"zh", "ja", "th"}


def norm_code(value: str) -> str:
    if not isinstance(value, str):
        return ""
    v = value.strip()
    if not v or v == "auto":
        return ""
    if v in CODE_TO_NAME:
        return v
    return NAME_TO_CODE.get(v.lower(), "")


def require_code(value: str, *, field: str) -> str:
    code = norm_code(value)
    if not code:
        raise BenchConfigError(
            f"{field}: unknown language {value!r} (known: {sorted(CODE_TO_NAME)})"
        )
    return code


def laal_unit(code: str) -> str:
    return "char" if code in CHAR_UNIT_LANGS else "word"
