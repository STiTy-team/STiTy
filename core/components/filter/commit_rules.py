import re
from collections import deque

from core.utils import langs, logging, scripts

from ..registry import Transcribed
from . import filters
from .base import Filter

log = logging.getLogger(__name__)

FILLERS = {
    "ko": {"음", "어", "으", "아", "에", "흠", "그", "그거", "뭐", "다", "이", "자", "으으",
           "으으으", "어어", "음음"},
    "en": {"um", "uh", "hmm", "mm", "ah", "er", "erm", "uhm", "mhm", "hm"},
    "ja": {"えー", "あの", "えっと", "うーん", "ま"},
    "zh": {"啊", "嗯", "呃", "哦", "那个"},
}
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def _key(text: str) -> str:
    return "".join(TOKEN_RE.findall((text or "").lower()))


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _word_lists(value) -> dict[str, set[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping of language -> words, got {type(value).__name__}")
    out = {}
    for code, words in value.items():
        lang = langs.norm_code(str(code))
        if not lang:
            raise ValueError(f"unknown language {code!r}")
        out[lang] = {w.lower() for w in words}
    return out


def _phrases(value) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"expected a list of phrases, got {type(value).__name__}")
    return [_key(p) for p in value if _key(p)]


@filters.register("commit-rules")
class CommitRules(Filter):
    SETTINGS = {
        "fillers": ("fillers", bool),
        "extra_fillers": ("extra_fillers", _word_lists),
        "canned_phrases": ("canned_phrases", _phrases),
        "foreign_script": ("foreign_script", bool),
        "repeats": ("repeats", int),
        "repeat_window_sec": ("repeat_window_sec", float),
        "repeat_min_chars": ("repeat_min_chars", int),
    }

    def start(self, languages: list[str] | None = None, **_) -> None:
        self.languages = [code for code in (langs.norm_code(c) for c in languages or ()) if code]
        self.recent: deque = deque(maxlen=max(1, self.settings.get("repeats", 0) or 1))
        extra = self.settings.get("extra_fillers", {})
        self.fillers = {lang: FILLERS.get(lang, set()) | extra.get(lang, set())
                        for lang in set(FILLERS) | set(extra)}

    def filter(self, records: list) -> list:
        kept = []
        for record in records:
            if isinstance(record, Transcribed):
                rule = self._rule_against(record)
                if rule:
                    log.info("[DROP] rule=%s gate=filter text=%r", rule, record.original)
                    continue
                self._remember(record)
            kept.append(record)
        return kept

    def _rule_against(self, record: Transcribed) -> str | None:
        text = record.original
        if self.settings.get("fillers") and self._only_fillers(text, record.language):
            return "filler"
        if _key(text) in self.settings.get("canned_phrases", ()):
            return "canned-phrase"
        if self.settings.get("foreign_script") and self._foreign(text):
            return "foreign-script"
        if self.settings.get("repeats") and self._repeated(record):
            return "repeat"
        return None

    def _only_fillers(self, text: str, language: str) -> bool:
        tokens = _tokens(text)
        if not tokens:
            return False
        words = set().union(*(self.fillers.get(lang, set())
                              for lang in {language, *self.languages} if lang))
        return all(token in words for token in tokens)

    def _foreign(self, text: str) -> bool:
        if not self.languages:
            return False
        script = scripts.dominant_script(text)
        if script is None:
            return False
        allowed = {scripts.script_for_language(code) for code in self.languages}
        if "ja" in self.languages:
            allowed.add("han")
        return script not in allowed

    def _repeated(self, record: Transcribed) -> bool:
        key = _key(record.original)
        if len(key) < self.settings.get("repeat_min_chars", 4):
            return False
        window = self.settings.get("repeat_window_sec", 15.0)
        return any(key == seen and record.decision_audio_sec - at <= window
                   for seen, at in self.recent)

    def _remember(self, record: Transcribed) -> None:
        self.recent.append((_key(record.original), record.decision_audio_sec))
