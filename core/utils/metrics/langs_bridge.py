"""The one language fact the metrics need: how to count a target language's units.

Separate from `core.utils.langs` only so importing the metrics does not pull the
supported-language table in, and so a caller scoring a language outside that table
still gets a sensible unit instead of an error.
"""
from core.utils import langs


def laal_unit(code: str) -> str:
    """Latency counts characters where the script has no word boundaries."""
    return langs.laal_unit(langs.norm_code(code) or (code or "").strip().lower())
