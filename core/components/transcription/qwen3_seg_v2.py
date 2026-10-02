import re

from core.utils import logging

from . import transcribers
from ..registry import Transcribed
from .qwen3_seg import (
    DROP_PUNCT_SPACE_RE,
    SEG_TAG,
    SHORT_TAIL_MAX_WORDS,
    Qwen3SegTranscription,
    bare_word,
    strip_asr_text,
    strip_lang_headers,
)

log = logging.getLogger(__name__)

KO_RISKY_FILLER_RE = re.compile(r"(?:(?:아니|응|그렇지|그거)[,.]?\s*)+[!?…]?")
WORD_RE = re.compile(r"\S+")
NON_WORD_RE = re.compile(r"[^\w]+")
LOOP_MIN_REPEATS = 3
LOOP_MAX_UNIT_WORDS = 8
MIN_SPEECH_SEC = 0.3


def find_tail_loop(text: str) -> int | None:
    spaced = (text or "").replace(SEG_TAG, " " * len(SEG_TAG))
    words = [(match.end(), NON_WORD_RE.sub("", match.group()).lower())
             for match in WORD_RE.finditer(spaced)]
    words = [(end, key) for end, key in words if key]
    keys = [key for _, key in words]
    for unit in range(1, LOOP_MAX_UNIT_WORDS + 1):
        body = keys
        if (len(keys) > unit and keys[-1] != keys[-1 - unit]
                and keys[-1 - unit].startswith(keys[-1])):
            body = keys[:-1]
        if len(body) < unit * LOOP_MIN_REPEATS:
            continue
        k = len(body) - 1
        while k >= unit and body[k] == body[k - unit]:
            k -= 1
        start = k + 1 - unit if k >= unit else 0
        if (len(body) - start) // unit >= LOOP_MIN_REPEATS:
            return words[start + unit - 1][0]
    return None


def repeats_tail(previous: str, sentence: str) -> bool:
    previous_words = [bare_word(w) for w in previous.split()]
    new_words = [bare_word(w) for w in sentence.split()]
    if not 1 <= len(new_words) <= SHORT_TAIL_MAX_WORDS or not all(new_words):
        return False
    return previous_words[-len(new_words):] == new_words


def squash(text: str) -> str:
    return DROP_PUNCT_SPACE_RE.sub("", strip_lang_headers(text or "")).lower()


@transcribers.register("qwen-seg:v2")
class Qwen3SegTranscriptionV2(Qwen3SegTranscription):

    SETTINGS = {
        **Qwen3SegTranscription.SETTINGS,
        "silence_window_grace_sec": ("silence_window_grace_sec", float),
        "min_speech_sec": ("min_speech_sec", float),
    }

    _text_first_heard_sec: float | None = None
    _dropped_texts: tuple = ()

    def _offer_partial(self, *, force: bool = False) -> None:
        seq = self._partial_seq
        super()._offer_partial(force=force)
        if (self._text_first_heard_sec is None and self._partial_seq != seq
                and self._partial_text):
            self._text_first_heard_sec = self._audio_sec()

    def _clear_partial(self) -> None:
        super()._clear_partial()
        self._text_first_heard_sec = None

    def _emit_final(self, original: str, reason: str) -> None:
        emitted = len(self._out)
        super()._emit_final(original, reason)
        self._text_first_heard_sec = None
        if not any(isinstance(item, Transcribed) for item in self._out[emitted:]):
            self._dropped_texts = (*self._dropped_texts, original)

    async def _decode_stream(self, chunk, state, **callbacks) -> None:
        await super()._decode_stream(chunk, state, **callbacks)
        self._cut_tail_loop(state)

    @staticmethod
    def _cut_tail_loop(state) -> None:
        text = state.text or ""
        cut = find_tail_loop(text)
        if cut is None:
            return
        kept = text[:cut].strip()
        log.info("[LOOP-CUT] text=%r kept=%r", strip_asr_text(text)[-80:], kept)
        state.text = kept
        state._last_nonempty_text = kept
        state.hallucination_detected = True

    def _find_duplicate(self, sentence: str, *, stage: str,
                        trigger: str | None = None,
                        batch_last: str | None = None,
                        batch_repeat: int = 0) -> tuple[str, str] | None:
        slot = self.slot
        if (sentence and not self.always_commit
                and stage in self.GUARD_STAGES["seg-boundary-dedup"]
                and "seg_reset_last_committed" in slot):
            previous = slot.pop("seg_reset_last_committed")
            if repeats_tail(previous, sentence):
                return "seg-boundary-dedup", previous
        return super()._find_duplicate(sentence, stage=stage, trigger=trigger,
                                       batch_last=batch_last, batch_repeat=batch_repeat)

    def _reset_slot(self, seed_text: str = "") -> None:
        super()._reset_slot(seed_text)
        self._dropped_texts = ()

    def _restart_with_audio(self, carry, last_committed: str, *,
                            header_reset: bool = False, dot_switch: bool = False) -> None:
        dropped = self._dropped_texts
        super()._restart_with_audio(carry, last_committed, header_reset=header_reset,
                                    dot_switch=dot_switch)
        previous = squash(self.slot.get("seg_reset_last_committed", ""))
        if previous and any(squash(text) and previous.endswith(squash(text))
                            for text in dropped):
            self.slot.pop("seg_reset_last_committed")
            log.info("[SEG-RESET-PREVIOUS] cause=dropped-commit previous=%r", last_committed)

    @staticmethod
    def _last_vad_reset_sec(spans: list) -> float:
        for span_start, span_end in reversed(spans):
            if span_end is not None:
                return span_end
        return 0.0

    def _silence_window_grace_sec(self) -> float:
        grace = self.settings.get("silence_window_grace_sec")
        if grace is not None:
            return float(grace)
        return float(self.settings.get("chunk_size_sec") or 2.0)

    def _min_speech_sec(self) -> float:
        value = self.settings.get("min_speech_sec")
        return MIN_SPEECH_SEC if value is None else float(value)

    def _speech_spans(self) -> list | None:
        if self.vad is None or getattr(self.vad, "disabled", False):
            return None
        settings = getattr(self.vad, "settings", None) or {}
        lag = (float(settings.get("min_silence_ms") or 800)
               - float(settings.get("speech_pad_ms") or 160)) / 1000
        return [(start, None if end is None else max(start, end - lag))
                for start, end in self.vad.spans]

    def _is_silence_hallucination(self, original: str, reason: str,
                                  audio_end_sec: float) -> bool:
        spans = self._speech_spans()
        if spans is None or not original.strip():
            return False
        reset = 0.0
        if any(end is not None for _, end in spans):
            reset = self._last_vad_reset_sec(spans) + self._silence_window_grace_sec()
        heard = self._text_first_heard_sec
        if heard is not None and heard < reset:
            return False
        start = max(self._last_final_end_sec, reset)
        end = audio_end_sec
        if end <= start:
            return False
        now = self._audio_sec()
        speech = sum(max(0.0, min(end, now if span_end is None else span_end)
                         - max(start, span_start))
                     for span_start, span_end in spans)
        if speech >= self._min_speech_sec():
            return False
        log.info("[DROP] rule=no-speech gate=emit reason=%s span=(%.1f~%.1f) "
                 "speech_sec=%.2f text=%r", reason, start, end, speech, original)
        return True

    def _drop_commit_candidate(self, original: str, reason: str) -> bool:
        dropped = self._should_drop_candidate(original, reason)
        if dropped:
            self._dropped_texts = (*self._dropped_texts, original)
        return dropped

    def _should_drop_candidate(self, original: str, reason: str) -> bool:
        if super()._drop_commit_candidate(original, reason):
            return True
        if reason not in ("vad", "finish"):
            return False
        stripped = strip_lang_headers(original or "").strip()
        if not KO_RISKY_FILLER_RE.fullmatch(stripped):
            return False
        if not self._is_silence_hallucination(stripped, reason, self._audio_sec()):
            return False
        self._log_drop("tail-filler-risky", "candidate", reason, stripped)
        return True
