from dataclasses import replace

from core.utils import langs, logging, scripts

from ..registry import Partial, Transcribed
from . import labelers
from .base import Labeler

log = logging.getLogger(__name__)


def _share(value) -> float:
    share = float(value)
    if not 0.0 < share <= 1.0:
        raise ValueError(f"min_share must be in (0, 1], got {share}")
    return share


@labelers.register("script")
class ScriptLabeler(Labeler):
    SETTINGS = {
        "min_share": ("min_share", _share),
        "latin_default": ("latin_default", langs.norm_code),
        "partials": ("partials", bool),
    }

    def start(self, languages: list[str] | None = None, **_) -> None:
        self.candidates = [code for code in (langs.norm_code(c) for c in languages or ()) if code]

    def label(self, records: list) -> list:
        return [self._relabel(record) for record in records]

    def _relabel(self, record):
        if isinstance(record, Transcribed):
            text = record.original
        elif isinstance(record, Partial) and self.settings.get("partials", True):
            text = record.text
        else:
            return record
        found = self.language_of(text, asr_label=record.language)
        if not found or found == record.language:
            return record
        if isinstance(record, Transcribed):
            log.info("[RELABEL] from=%s to=%s text=%r", record.language or "?", found, text)
        return replace(record, language=found)

    def language_of(self, text: str, *, asr_label: str) -> str:
        script = scripts.dominant_script(text, min_share=self.settings.get("min_share", 0.5))
        if script is None:
            return asr_label
        if asr_label and scripts.script_for_language(asr_label) == script:
            return asr_label
        latin_default = self.settings.get("latin_default", "en")
        fallback = latin_default if script == "latin" else asr_label
        return scripts.language_for_script(script, candidates=self.candidates,
                                           fallback=fallback) or asr_label
