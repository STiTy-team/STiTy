import difflib
import re
from collections import deque

import numpy as np

from core.utils import audio as audio_mod
from core.utils import clock
from core.utils import langs
from core.utils import logging
from core.utils.glossary import Glossary

from . import transcribers
from ..registry import Partial, Speech, Transcribed
from .qwen3_seg import (
    DROP_PUNCT_SPACE_RE,
    PARTIAL_MIN_INTERVAL_SEC,
    RETRY_LEAD_PAD_SEC,
    REP_DEDUP_MAX_REPEATS,
    SEG_HEADER_TAIL_RE,
    SEG_TAG,
    Qwen3SegTranscription,
    asr_key,
    bare_word,
    committed_cursor,
    display_of,
    fuzzy_key,
    strip_asr_text,
    strip_lang_headers,
    tail_overlaps,
    uncommitted_from,
)

log = logging.getLogger(__name__)

MAX_CARRY_SEC = 30.0
LOOP_MIN_REPEATS = 4
LOOP_MAX_PERIOD = 4
COVER_RATIO = 0.8
COVER_SLACK = 2
UNIT_END_RE = re.compile(r"<SEG>|[.!?。？！]+(?=\s|<SEG>|$)")
NORM_DROP_RE = re.compile(r"[\W_]+")
QWEN_LANGUAGE_NAMES = (
    "Chinese English Cantonese Arabic German French Spanish Portuguese Indonesian Italian "
    "Korean Russian Thai Vietnamese Japanese Turkish Hindi Malay Dutch Swedish Danish Finnish "
    "Polish Czech Filipino Persian Greek Hungarian Macedonian Romanian None").split()
LANGUAGE_NAME_RE = "(?:" + "|".join(QWEN_LANGUAGE_NAMES) + ")"
LANGUAGE_LIST_RE = re.compile(
    rf"^\s*(?:language\s+)?{LANGUAGE_NAME_RE}(?:\s*[,;:.…]*\s*(?:language\s+)?{LANGUAGE_NAME_RE})+"
    rf"\s*[,;:.!?…]*\s*$", re.IGNORECASE)
PARTIAL_HEADER_STUB_RE = re.compile(
    r"(?:^|(?<=[.,!?;:。？！]))\s*(?:l|la|lan|lang|langu|langua|languag|language)"
    r"(?:\s+[A-Z][A-Za-z]*)?\s*$")

FLAGS = (
    "no_speech_since_vad_reset",
    "keep_held_fragment",
    "strict_boundary_dedup",
    "carry_uncommitted_audio",
    "resync_cursor",
    "loop_guard",
    "carry_on_loop_reset",
    "drop_language_lists",
    "clean_partials",
    "dedup_dot_carry",
)


def norm_key(text: str) -> str:
    return NORM_DROP_RE.sub("", (text or "").replace(SEG_TAG, "")).lower()


def text_units(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in UNIT_END_RE.finditer(text):
        end = match.start() if match.group() == SEG_TAG else match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def covered_prefix(text: str, committed_display: str) -> int:
    target = norm_key(committed_display)
    pointer, pos = 0, 0
    for start, end in text_units(text):
        if pointer >= len(target):
            break
        key = norm_key(text[start:end])
        if not key:
            pos = end
            continue
        found = target.find(key, pointer)
        if found != -1 and found - pointer <= (COVER_SLACK if len(key) >= 3 else 0):
            pointer, pos = found + len(key), end
            continue
        window = target[pointer:pointer + len(key)]
        if len(key) >= 4 and difflib.SequenceMatcher(None, key, window).ratio() >= COVER_RATIO:
            pointer, pos = pointer + len(window), end
            continue
        break
    return pos


def prefix_matches(text: str, committed_display: str) -> bool:
    return committed_cursor(text, committed_display, 0) != -1


def find_loop(text: str, *, min_repeats: int = LOOP_MIN_REPEATS,
              max_period: int = LOOP_MAX_PERIOD) -> int | None:
    tokens = [(m.end(), NORM_DROP_RE.sub("", m.group()).lower())
              for m in re.finditer(r"\S+", text.replace(SEG_TAG, " " * len(SEG_TAG)))]
    tokens = [(end, key) for end, key in tokens if key]
    keys = [key for _, key in tokens]
    for i in range(len(keys)):
        for period in range(1, max_period + 1):
            base = keys[i:i + period]
            if len(base) < period:
                break
            repeats, j = 1, i + period
            while keys[j:j + period] == base:
                repeats += 1
                j += period
            if repeats >= min_repeats:
                return tokens[i + period - 1][0]
    return None


def repeats_tail(previous: str, sentence: str) -> bool:
    new_words = [bare_word(w) for w in sentence.split()]
    old_words = [bare_word(w) for w in previous.split()]
    return (bool(new_words) and len(old_words) >= len(new_words)
            and old_words[-len(new_words):] == new_words)


def clean_partial(text: str) -> str:
    text = strip_lang_headers(text)
    return PARTIAL_HEADER_STUB_RE.sub("", text).strip()


@transcribers.register("qwen-seg:v2")
class Qwen3SegTranscriptionV2(Qwen3SegTranscription):
    SETTINGS = {
        **Qwen3SegTranscription.SETTINGS,
        **{flag: (flag, bool) for flag in FLAGS},
        "tail_keep_sec": ("tail_keep_sec", float),
        "bias_glossary": ("bias_glossary", str),
        "bias_max_terms": ("bias_max_terms", int),
        "bias_recent_commits": ("bias_recent_commits", int),
    }

    async def load(self) -> None:
        await super().load()
        name = self.settings.get("bias_glossary")
        self.glossary = Glossary.load(name) if name else None

    def start(self, languages: list[str] | None = None, vad=None, **_) -> None:
        self.flags = {flag: bool(self.settings.get(flag, False)) for flag in FLAGS}
        self.tail_keep_sec = float(self.settings.get("tail_keep_sec", 0.1))
        self._languages = tuple(languages or ())
        remembered = max(1, self.settings.get("bias_recent_commits", 0))
        self._recent_finals: deque = deque(maxlen=remembered)
        self._vad_reset_sec = 0.0
        super().start(languages=languages, vad=vad, **_)
        enabled = sorted(flag for flag, on in self.flags.items() if on)
        log.info("[TRANSCRIBE-V2] flags=%s tail_keep_sec=%.2f bias=%s", enabled or "none",
                 self.tail_keep_sec, self.settings.get("bias_glossary") or "-")

    def _biasing(self) -> bool:
        return bool(getattr(self, "glossary", None) or self.settings.get("bias_recent_commits"))

    def _bias_context(self) -> str:
        parts = []
        if getattr(self, "glossary", None) is not None:
            codes = [langs.norm_code(c) for c in self._languages]
            forms = self.glossary.spoken_forms([c for c in codes if c])
            forms = forms[:self.settings.get("bias_max_terms", 20)]
            if forms:
                parts.append("Names and terms that may be spoken: " + ", ".join(forms) + ".")
        recent = [text for text in getattr(self, "_recent_finals", ()) if text]
        if self.settings.get("bias_recent_commits") and recent:
            parts.append("Earlier in this conversation: " + " ".join(recent))
        return "\n".join(parts)

    def _new_state(self, seed_text: str = ""):
        if not self._biasing():
            return super()._new_state(seed_text)
        kw = self.settings
        state = self.model.init_streaming_state(
            context=self._bias_context(),
            chunk_size_sec=float(kw.get("chunk_size_sec", 2.0)),
            unfixed_chunk_num=int(kw.get("unfixed_chunk_num", 2)),
            unfixed_token_num=int(kw.get("unfixed_token_num", 5)),
            allowed_languages=self._allowed_language_names(),
        )
        if seed_text:
            state._raw_decoded = seed_text
            state.text = seed_text
            state.chunk_id = state.unfixed_chunk_num
        return state

    def _flag(self, name: str) -> bool:
        return getattr(self, "flags", {}).get(name, False)

    def _resync(self, current_text: str) -> None:
        if not self._flag("resync_cursor"):
            return
        slot = self.slot
        committed = slot["committed_display"]
        if not committed or not current_text or prefix_matches(current_text, committed):
            return
        pos = covered_prefix(current_text, committed)
        display = display_of(current_text[:pos])
        segs = current_text[:pos].count(SEG_TAG)
        if display == committed and segs == slot["committed_seg_count"]:
            return
        log.info("[CURSOR-RESYNC] was=%r now=%r", committed, display)
        slot["committed_display"] = display
        slot["committed_seg_count"] = segs

    def _current_text(self) -> str:
        return strip_asr_text((self.slot["state"].text or "").strip())

    def _uncommitted_display(self, text_snapshot: str | None = None) -> str:
        if text_snapshot is None:
            self._resync(self._current_text())
        return super()._uncommitted_display(text_snapshot)

    def _log_seg(self) -> None:
        self._resync(self._current_text())
        super()._log_seg()

    async def _process_updates(self, force_reason: str | None = None,
                               chunk_end: bool = False,
                               final: bool = False) -> str | None:
        self._resync(self._current_text())
        return await super()._process_updates(force_reason=force_reason, chunk_end=chunk_end,
                                              final=final)

    async def _flush_uncommitted(self, reason: str) -> None:
        self._resync(self._current_text())
        await super()._flush_uncommitted(reason)

    async def _decode_stream(self, chunk, state, **callbacks) -> None:
        if not self._flag("loop_guard"):
            await super()._decode_stream(chunk, state, **callbacks)
            return
        self._looping = False
        guarded = {name: None if callback is None else self._unless_looping(callback)
                   for name, callback in callbacks.items()}
        await super()._decode_stream(chunk, state, **guarded)
        if state.hallucination_detected and not self._looping:
            return
        self._cut_loop(state)

    def _unless_looping(self, callback):
        async def guarded(state) -> None:
            if not self._cut_loop(state):
                await callback(state)

        return guarded

    def _cut_loop(self, state) -> bool:
        text = strip_asr_text((state.text or "").strip())
        cut = find_loop(text)
        if cut is None:
            return self._looping
        if not self._looping:
            log.info("[LOOP-GUARD] kept=%r dropped=%r", text[:cut], text[cut:])
        self._looping = True
        state.text = text[:cut]
        state.hallucination_detected = True
        return True

    async def _recover_from_hallucination(self) -> None:
        if not self._flag("carry_on_loop_reset"):
            await super()._recover_from_hallucination()
            return
        state = self.slot["state"]
        state.hallucination_detected = False
        if not (state.text or "").strip() and getattr(state, "_last_nonempty_text", ""):
            state.text = state._last_nonempty_text
        chunk_samples = self._chunk_samples()
        accum = state.audio_accum
        carry = accum[-chunk_samples:].copy() if accum is not None and accum.shape[0] else None
        log.info("[SLOT-RESET] cause=repetition-loop carry_sec=%.2f text=%r",
                 0.0 if carry is None else carry.shape[0] / audio_mod.SAMPLING_RATE,
                 strip_asr_text((state.text or "").strip()))
        await self._flush_uncommitted(reason="vad")
        self._reset_slot()
        if carry is not None:
            self.slot["state"].audio_accum = carry
            self.slot["real_audio"] = True
        self._offer_partial(force=True)

    def _chunk_samples(self) -> int:
        return int(round(float(self.settings.get("chunk_size_sec", 2.0))
                         * audio_mod.SAMPLING_RATE))

    def _is_silence_hallucination(self, original: str, reason: str,
                                  audio_end_sec: float) -> bool:
        if not self._flag("no_speech_since_vad_reset"):
            return super()._is_silence_hallucination(original, reason, audio_end_sec)
        saved = self._last_final_end_sec
        self._last_final_end_sec = max(saved, self._vad_reset_sec)
        try:
            return super()._is_silence_hallucination(original, reason, audio_end_sec)
        finally:
            self._last_final_end_sec = saved

    def _merge_comma_fragments(self, committed_items: list, closing: bool) -> list:
        merged = super()._merge_comma_fragments(committed_items, closing)
        if self._flag("keep_held_fragment") and self._deferred_fragment:
            self._forget(self._deferred_fragment)
        return merged

    def _forget(self, text: str) -> None:
        slot = self.slot
        if slot["committed_by_asr_key"].get(asr_key(text)) == text:
            del slot["committed_by_asr_key"][asr_key(text)]
        if slot["committed_by_fuzzy_key"].get(fuzzy_key(text)) == text:
            del slot["committed_by_fuzzy_key"][fuzzy_key(text)]

    def _drop_commit_candidate(self, original: str, reason: str) -> bool:
        if self._flag("drop_language_lists") and LANGUAGE_LIST_RE.match(original or ""):
            self._log_drop("language-list", "candidate", reason, original)
            return True
        return super()._drop_commit_candidate(original, reason)

    def _emit_final(self, original: str, reason: str) -> None:
        if self._flag("drop_language_lists") and LANGUAGE_LIST_RE.match(original or ""):
            self._log_drop("language-list", "emit", reason, original)
            return
        before = len(self._out)
        super()._emit_final(original, reason)
        if len(self._out) > before and isinstance(self._out[-1], Transcribed):
            self._recent_finals.append(self._out[-1].original)

    def _offer_partial(self, *, force: bool = False) -> None:
        if not self._flag("clean_partials"):
            super()._offer_partial(force=force)
            return
        if not force and clock.elapsed_since(self._partial_at) < PARTIAL_MIN_INTERVAL_SEC:
            return
        text = clean_partial(self._uncommitted_display())
        if not text and not force:
            return
        previous = self._partial_text
        if not force and previous and text != previous and previous.startswith(text):
            return
        if text == previous:
            self._partial_at = clock.monotonic()
            return
        self._partial_text = text
        self._partial_at = clock.monotonic()
        self._partial_seq += 1
        self._out.append(Partial(
            text=text,
            language=langs.norm_code(self.slot["last_text_lang"]),
            seq=self._partial_seq))

    def _restart_with_audio(self, carry: np.ndarray | None, last_committed: str, *,
                            header_reset: bool = False, dot_switch: bool = False) -> None:
        if self._flag("strict_boundary_dedup") and not dot_switch:
            last_committed = self._last_final_text
        super()._restart_with_audio(carry, last_committed, header_reset=header_reset,
                                    dot_switch=dot_switch)

    def _find_duplicate(self, sentence: str, *, stage: str,
                        trigger: str | None = None,
                        batch_last: str | None = None,
                        batch_repeat: int = 0) -> tuple[str, str] | None:
        strict = self._flag("strict_boundary_dedup")
        dot_any = self._flag("dedup_dot_carry")
        if not (strict or dot_any):
            return super()._find_duplicate(sentence, stage=stage, trigger=trigger,
                                           batch_last=batch_last, batch_repeat=batch_repeat)
        slot = self.slot
        applies = lambda name: stage in self.GUARD_STAGES[name]
        if not sentence:
            return None

        if (applies("rep-dedup") and self.rep_dedup and batch_last is not None
                and sentence == batch_last and batch_repeat >= REP_DEDUP_MAX_REPEATS):
            return "rep-dedup", batch_last

        if (applies("seg-boundary-dedup") and "seg_reset_last_committed" in slot
                and not self.always_commit):
            previous = slot.pop("seg_reset_last_committed")
            if strict:
                if repeats_tail(previous, sentence):
                    return "seg-boundary-dedup", previous
            else:
                words = sentence.split()
                first_word = bare_word(words[0]) if words else ""
                last_word = bare_word(previous.split()[-1]) if previous.split() else ""
                if first_word and last_word and (first_word == last_word
                                                 or last_word.endswith(first_word)):
                    return "seg-boundary-dedup", previous
                if tail_overlaps(previous, sentence):
                    return "seg-boundary-dedup", previous

        if applies("header-reset-tail-dedup") and "header_reset_last_committed" in slot:
            previous = slot.pop("header_reset_last_committed")
            overlaps = repeats_tail(previous, sentence) if strict else tail_overlaps(previous, sentence)
            if overlaps:
                return "header-reset-tail-dedup", previous

        if applies("dot-suffix-dedup"):
            previous = slot.get("dot_switch_prev_committed", "")
            if previous and (trigger == "dot" or stage == "flush" or dot_any):
                slot.pop("dot_switch_prev_committed", None)
                drop = lambda s: DROP_PUNCT_SPACE_RE.sub('', s)
                if drop(previous).endswith(drop(sentence)):
                    return "dot-suffix-dedup", previous

        if applies("committed-suffix-dedup") and not self.always_commit:
            bare = strip_lang_headers(sentence)
            if bare and len(bare.split()) >= 2:
                key = DROP_PUNCT_SPACE_RE.sub('', bare).lower()
                for previous in (slot["committed_display"],
                                 slot.get("dot_switch_prev_committed", ""),
                                 slot.get("seg_reset_last_committed", ""),
                                 slot.get("header_reset_last_committed", ""),
                                 self._last_final_text):
                    stripped = DROP_PUNCT_SPACE_RE.sub(
                        '', strip_lang_headers(previous or "")).lower()
                    if key and stripped and stripped.endswith(key):
                        return "committed-suffix-dedup", previous

        if applies("cross-dedup") and asr_key(sentence) in slot["committed_by_asr_key"]:
            return "cross-dedup", slot["committed_by_asr_key"][asr_key(sentence)]

        if applies("cross-dedup-fuzzy") and not self.always_commit:
            matched = self._cross_dup_match(sentence)
            if matched is not None:
                return "cross-dedup-fuzzy", matched

        return None

    async def _reshape_slot(self, committed_before: str, previous_uncommitted: str,
                            accum_before: int, snapshot: str | None) -> None:
        if not self._flag("carry_uncommitted_audio"):
            await super()._reshape_slot(committed_before, previous_uncommitted, accum_before,
                                        snapshot)
            return
        slot = self.slot
        committed = slot["committed_display"]
        any_commit = committed != committed_before
        decoded = strip_asr_text((slot["state"].text or "").strip())
        header_tail = bool(SEG_HEADER_TAIL_RE.search(decoded))
        if any_commit or not header_tail:
            await super()._reshape_slot(committed_before, previous_uncommitted, accum_before,
                                        snapshot)
            return
        accum = slot["state"].audio_accum
        since = slot.get("seg_chunk_accum")
        if since is None or since >= accum.shape[0]:
            since = 0
        since = max(since, accum.shape[0] - int(MAX_CARRY_SEC * audio_mod.SAMPLING_RATE))
        carry = accum[since:].copy()
        self._restart_with_audio(carry, committed, header_reset=True)
        log.info("[SLOT-RESET] cause=dangling-lang-header carry_sec=%.2f tail=%r "
                 "slot_committed=%r", carry.shape[0] / audio_mod.SAMPLING_RATE,
                 decoded[-40:], committed)
        self._offer_partial(force=True)

    def _trim_tail_silence(self, speech: Speech | None) -> None:
        if speech is None:
            return
        cut = int(max(0.0, speech.silence_waited_out_sec - self.tail_keep_sec)
                  * audio_mod.SAMPLING_RATE)
        if cut <= 0:
            return
        state = self.slot["state"]
        empty = np.zeros((0,), dtype=np.float32)
        buffer = state.buffer if state.buffer is not None else empty
        accum = state.audio_accum if state.audio_accum is not None else empty
        if buffer.shape[0] + accum.shape[0] <= cut:
            return
        from_buffer = min(cut, buffer.shape[0])
        state.buffer = buffer[: buffer.shape[0] - from_buffer]
        if cut > from_buffer:
            state.audio_accum = accum[: accum.shape[0] - (cut - from_buffer)]
        self.slot["tail_trim_samples"] += cut
        log.debug("[AUDIO-TRIM] cause=tail-silence removed_sec=%.2f",
                  cut / audio_mod.SAMPLING_RATE)

    async def _retry_short_utterance(self, speech: Speech) -> None:
        slot = self.slot
        audio = slot["state"].audio_accum
        if audio is None or audio.shape[0] == 0:
            return
        anchor_samples = int(slot["audio_anchor_sec"] * audio_mod.SAMPLING_RATE)
        start = int((speech.started_at - RETRY_LEAD_PAD_SEC)
                    * audio_mod.SAMPLING_RATE) - anchor_samples
        start = max(0, start)
        tail_cut = max(0, int(max(0.0, speech.silence_waited_out_sec
                                  - self.tail_keep_sec) * audio_mod.SAMPLING_RATE)
                       - slot["tail_trim_samples"])
        end = audio.shape[0] - tail_cut
        if end <= start:
            return
        state = self._new_state()
        state.buffer = audio[start:end].copy()
        await self._decode(state)
        retried = (state.text or "").strip()
        log.info("[UTTERANCE-RETRY] cause=empty-decode text=%r", retried)
        if retried:
            slot["state"] = state
            self.state = state

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        slot = self.slot
        self._resync(self._current_text())
        before = (slot["state"].text or "").strip()
        uncommitted_before = uncommitted_from(
            before, slot["committed_display"], slot["committed_seg_count"])
        await self._process_updates(force_reason=reason)
        state = self.slot["state"]
        short_utterance = (self._accum_before_chunk == 0
                           and self._buffered_before_chunk > 0)
        if (self._flag("carry_uncommitted_audio") and not before
                and state.buffer.shape[0] == 0 and state.audio_accum.shape[0] > 0):
            log.info("[CARRY-DECODE] audio_sec=%.2f",
                     state.audio_accum.shape[0] / audio_mod.SAMPLING_RATE)
            state.buffer = state.audio_accum
            state.audio_accum = np.zeros((0,), dtype=np.float32)
        if before or state.buffer.shape[0] > 0 or short_utterance:
            self._trim_tail_silence(speech)
            await self._decode(state)
            if not (state.text or "").strip() and speech is not None:
                await self._retry_short_utterance(speech)
        state = self.slot["state"]
        after = (state.text or "").strip()
        uncommitted_after = uncommitted_from(
            after, self.slot["committed_display"], self.slot["committed_seg_count"])
        if (not strip_lang_headers(display_of(uncommitted_after))
                and strip_lang_headers(display_of(uncommitted_before))):
            state.text = before
        await self._flush_uncommitted(reason=reason)
        self._reset_slot()
        log.info("[SLOT-RESET] cause=%s", reason)
        self._close_utterance(reason)
        if reason == "vad":
            self._vad_reset_sec = self._audio_sec()
        return self._drain()
