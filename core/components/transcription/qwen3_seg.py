import difflib
import re
import time
from functools import cache

import numpy as np

from core.utils import audio as audio_mod
from core.utils import langs
from core.utils import logging
from core.utils import timing

from . import transcribers
from ..registry import Partial, Speech, Transcribed
from .qwen3 import Qwen3Transcription

log = logging.getLogger(__name__)

SEG_TAG = "<SEG>"
SEG_RE = re.compile(SEG_TAG)

PARTIAL_MIN_INTERVAL_SEC = 0.12
KEEP_AFTER_SPEECH_SEC = 0.1
RETRY_LEAD_PAD_SEC = 0.2
MAX_AUDIO_ACCUM_SEC = 90.0
MAX_SEED_COMMITTED_SENTENCES = 1
REP_DEDUP_MAX_REPEATS = 2
FUZZY_DEDUP_MIN_WORDS = 6
FUZZY_DEDUP_RATIO = 0.85
FRAGMENT_MAX_WORDS = 3
SHORT_TAIL_MAX_WORDS = 5
MIN_SPEECH_OVERLAP_SEC = 0.05

PUNCT = '.,!?;:。？！'
QUOTE_COLON_RE = re.compile("[\"“”'‘’：:]+")
PUNCT_CLASS_RE = re.compile(r'[.,!?;:。？！、，]')
TRAILING_PUNCT_RE = re.compile(r'[.,!?;:。？！]+$')
DROP_PUNCT_SPACE_RE = re.compile(r'[.,!?;:。？！\s]+')
FRAGMENT_TAIL_RE = re.compile(r"[,;、，]\s*$")
WORD_CHAR_RE = re.compile(r"[^\W_]")
LONE_COMMA_WORD_RE = re.compile(r"\S+,")

HEADER_ONLY_RE = re.compile(r"^\s*language\s+[A-Z][A-Za-z]*\s*[.!?]?\s*$")
SEG_HEADER_TAIL_RE = re.compile(r"<SEG>\s*language(?:\s+[A-Za-z]*)?[.,!?]?\s*$")
LANG_HEADER_INLINE_RE = re.compile(
    r"^\s*language\s+(?:None|[A-Z][A-Za-z]*)\s*(?:<asr_text>)?[.,!?;:]?\s*")
LANG_HEADER_AFTER_PUNCT_RE = re.compile(
    r"(?<=[.,!?;:])\s+language\s+(?:None|[A-Z][A-Za-z]*)(?=[\s.,!?;:]|$)[.,!?;:]?\s*")
KNOWN_SILENCE_RE = re.compile(
    r"^\s*(?:"
    r"i'?m sorry,? but i can'?t (?:hear|help|understand) you\b.*"
    r"|i'?m sorry,(?:\s*(?:i'?m|i|but))?"
    r"|i want to (?:see the movie|know (?:the weather|how to make))\b.*"
    r"|그리고 그거"
    r"|그러니까"
    r")\s*$",
    re.IGNORECASE,
)
KO_TAIL_FILLER_RE = re.compile(r"(?:그리고|그게|그|어|뭐|자|이|아|음|네|그니까)[.,!?…]?")


@cache
def _boundary():
    from qwen_asr.inference import sentence_boundary

    return sentence_boundary


def squeeze(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def display_of(raw: str) -> str:
    return squeeze(raw.replace(SEG_TAG, ""))


def cut_repeats(text: str, max_repeat: int = 4) -> str:
    tokens = text.split()
    i = 0
    while i < len(tokens):
        j = i + 1
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        if j - i >= max_repeat:
            return ' '.join(tokens[:i]).strip()
        i = j
    return text


def strip_asr_text(text: str) -> str:
    if "<asr_text>" not in text:
        result = text
    else:
        first, rest = text.split("<asr_text>", 1)
        if re.match(r'^\s*language\s+\w', first, re.IGNORECASE):
            text = rest
        result = re.sub(r'\s*<asr_text>\s*', ' ', text).strip()
    return cut_repeats(result)


def strip_lang_headers(text: str) -> str:
    if not text or "language" not in text:
        return text or ""
    text = LANG_HEADER_INLINE_RE.sub("", text)
    text = LANG_HEADER_AFTER_PUNCT_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def boundary_key(sentence: str) -> str:
    return re.sub(r'[.,!?;:。？！、，\'"“”‘’\s]+', '', sentence.replace(SEG_TAG, "")).lower()


def fuzzy_key(text: str) -> str:
    return ' '.join(re.sub(r"[^a-z' ]", ' ', text.lower()).split())


def asr_key(text: str) -> str:
    return ' '.join(text.split()).rstrip(PUNCT).strip().lower()


def bare_word(word: str) -> str:
    return TRAILING_PUNCT_RE.sub('', word).lower()


def is_short_fragment(text: str) -> bool:
    return (bool(FRAGMENT_TAIL_RE.search(text))
            and len(text.split()) <= FRAGMENT_MAX_WORDS)


def tail_overlaps(previous: str, sentence: str) -> bool:
    previous_words = [bare_word(w) for w in previous.split()]
    new_words = [bare_word(w) for w in sentence.split()]
    if not previous_words or not 1 <= len(new_words) <= SHORT_TAIL_MAX_WORDS:
        return False
    tail = previous_words[-len(new_words):]
    hits = sum(1 for a, b in zip(tail, new_words)
               if a and b and (a == b or a.endswith(b) or b.endswith(a)))
    return hits >= max(1, (len(new_words) + 1) // 2)


def committed_cursor(text: str, committed_display: str,
                     committed_seg_count: int = 0) -> int:
    seg_len = len(SEG_TAG)
    text_no_seg = display_of(text)

    def walk(target_len: int, skip_re=None, advance_punct: bool = False) -> int:
        pos, shown, after_space = 0, 0, False
        while pos < len(text) and shown < target_len:
            if text[pos:pos + seg_len] == SEG_TAG:
                pos += seg_len
                after_space = True
            elif text[pos] == ' ' and after_space:
                pos += 1
            elif skip_re and skip_re.match(text[pos]):
                pos += 1
            else:
                after_space = (text[pos] == ' ')
                shown += 1
                pos += 1
        if advance_punct and pos < len(text) and text[pos] in PUNCT:
            pos += 1
        return pos

    if committed_display:
        flatten = lambda s: PUNCT_CLASS_RE.sub('.', s)
        candidates = [
            (committed_display, text_no_seg, False, None),
            (flatten(committed_display), flatten(text_no_seg), False, None),
            (committed_display.rstrip(PUNCT), text_no_seg, True, None),
            (QUOTE_COLON_RE.sub('', committed_display),
             QUOTE_COLON_RE.sub('', text_no_seg), False, QUOTE_COLON_RE),
        ]
        candidates += [
            (committed_norm.lower(), text_norm.lower(), advance_punct, skip_re)
            for committed_norm, text_norm, advance_punct, skip_re in list(candidates)
            if len(committed_norm.lower()) == len(committed_norm)
            and len(text_norm.lower()) == len(text_norm)
        ]
        for committed_norm, text_norm, advance_punct, skip_re in candidates:
            if not committed_norm or not text_norm.startswith(committed_norm):
                continue
            end = len(committed_norm)
            if end < len(text_norm) and text_norm[end].isalpha():
                continue
            return walk(len(committed_norm), skip_re=skip_re, advance_punct=advance_punct)

    if committed_seg_count > 0 or not committed_display:
        pos, found, all_found = 0, 0, True
        while found < committed_seg_count:
            idx = text.find(SEG_TAG, pos)
            if idx == -1:
                all_found = False
                break
            pos = idx + seg_len
            found += 1
        if all_found and pos < len(text):
            candidate = display_of(text[:pos])
            if difflib.SequenceMatcher(None, candidate, committed_display).ratio() < 0.5:
                return -1
            return pos

    return -1


def uncommitted_from(current_text: str, committed_display: str,
                     committed_seg_count: int = 0) -> str:
    pos = committed_cursor(current_text, committed_display, committed_seg_count)
    if pos != -1:
        return current_text[pos:]
    if not (committed_display and current_text):
        return ""
    pos, found = 0, 0
    while found < committed_seg_count:
        idx = current_text.find(SEG_TAG, pos)
        if idx == -1:
            return ""
        pos = idx + len(SEG_TAG)
        found += 1
    return current_text[pos:]


def collapse_repetition(text: str, max_repeats: int = REP_DEDUP_MAX_REPEATS) -> str:
    if not text:
        return text

    def fold(units, keys, max_period):
        out, dropped, i = [], 0, 0
        while i < len(units):
            for n in range(min(max_period, (len(units) - i) // (max_repeats + 1)), 0, -1):
                base = keys[i:i + n]
                if not any(base):
                    continue
                j = i + n
                while keys[j:j + n] == base:
                    j += n
                if (j - i) // n > max_repeats:
                    out.extend(units[i:i + n * max_repeats])
                    dropped += (j - i) // n - max_repeats
                    i = j
                    break
            else:
                out.append(units[i])
                i += 1
        return out, dropped

    sentences = _boundary().split_sentences(text)
    kept, dropped = fold(sentences, [boundary_key(s) for s in sentences], max_period=4)
    result = "".join(kept)

    words = result.split()
    kept_words, dropped_words = fold(
        words, [re.sub(r'[^\w]+', '', w).lower() for w in words], max_period=8)
    if dropped_words:
        dropped += dropped_words
        result = " ".join(kept_words)

    if dropped:
        log.info("[FLUSH-DEDUP] collapsed %d cycles (%d -> %d chars)",
                    dropped, len(text), len(result))
    return result.strip()


@transcribers.register('qwen-seg')
class Qwen3SegTranscription(Qwen3Transcription):
    SETTINGS = {
        **Qwen3Transcription.SETTINGS,
        "dot_commit_confirm": ("dot_commit_confirm", bool),
        "dot_commit_stall_chunks": ("dot_commit_stall_chunks", int),
        "rep_dedup": ("rep_dedup", bool),
    }

    GUARD_STAGES = {
        "rep-dedup": ("extract",),
        "seg-boundary-dedup": ("extract",),
        "dot-suffix-dedup": ("extract",),
        "cross-dedup": ("commit", "flush"),
        "cross-dedup-fuzzy": ("commit", "flush"),
        "header-reset-tail-dedup": ("extract", "commit", "flush"),
        "committed-suffix-dedup": ("extract", "commit", "flush"),
    }

    @classmethod
    def validate(cls, options: dict, *, kind: str = "transcription") -> dict:
        out = super().validate(options, kind=kind)
        out.setdefault("dot_commit_confirm", True)
        out.setdefault("dot_commit_stall_chunks", 1)
        out.setdefault("rep_dedup", True)
        return out

    def start(self, language: str | None = None, vad=None, **_) -> None:
        commit = self.cfg.stity.commit
        self.always_commit = commit.always_commit
        self.enable_dot_commit = commit.enable_dot_commit
        self.hide_seg = commit.hide_seg
        self.dot_commit_confirm = bool(self.settings["dot_commit_confirm"])
        self.dot_commit_stall_chunks = int(self.settings["dot_commit_stall_chunks"])
        self.rep_dedup = bool(self.settings["rep_dedup"])
        self.vad = vad

        self._fed_samples = 0
        self._partial_seq = 0
        self._out: list = []
        self._partial_text = None
        self._partial_at = 0.0
        self._last_final_text = ""
        self._last_final_end_sec = 0.0
        self._deferred_fragment = ""
        self._accum_before_chunk = 0
        self._buffered_before_chunk = 0
        self._reset_slot()

    def _drain(self) -> list:
        out, self._out = self._out, []
        return out

    @timing.measure("decode")
    async def _decode(self, state) -> None:
        await self.model.finish_streaming_transcribe(state)

    @timing.measure("decode")
    async def _decode_stream(self, chunk, state, **callbacks) -> None:
        await self.model.streaming_transcribe(chunk, state, **callbacks)

    def _new_state(self, seed_text: str = ""):
        kw = self.settings
        state = self.model.init_streaming_state(
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

    def _reset_slot(self, seed_text: str = "") -> None:
        self.slot = {
            "state": self._new_state(seed_text),
            "last_text": seed_text,
            "last_text_lang": "",
            "committed_display": "",
            "committed_seg_count": 0,
            "committed_asr_set": set(),
            "committed_fuzzy_keys": [],
            "audio_anchor_sec": self._audio_sec(),
            "tail_trim_samples": 0,
            "real_audio": False,
        }
        self.state = self.slot["state"]

    def _audio_sec(self) -> float:
        return getattr(self, "_fed_samples", 0) / audio_mod.SAMPLING_RATE

    def _next_boundary(self, text: str):
        pattern = (_boundary().DOT_COMMIT_BOUNDARY_RE if self.enable_dot_commit
                   else SEG_RE)
        start = 0
        while True:
            match = pattern.search(text, start)
            if match is None or not (self.hide_seg and match.group() == SEG_TAG):
                return match
            start = match.end()

    @staticmethod
    def _extract_boundary_keys(text: str) -> tuple:
        keys, remaining = [], text
        while True:
            match = _boundary().DOT_COMMIT_BOUNDARY_RE.search(remaining)
            if not match:
                return tuple(keys)
            key = boundary_key(remaining[:match.end()])
            if key:
                keys.append(key)
            remaining = remaining[match.end():]

    def _count_tokens(self, text: str) -> int:
        if not text.strip():
            return 0
        try:
            return len(self.model.processor.tokenizer.encode(text))
        except Exception:  # noqa: BLE001
            return len(text.split())

    async def transcribe(self, audio: bytes) -> list:
        chunk = audio_mod.from_pcm_bytes(audio)
        if chunk.size == 0:
            return self._drain()
        self._fed_samples += chunk.size
        slot = self.slot
        self._accum_before_chunk = slot["state"].audio_accum.shape[0]
        self._buffered_before_chunk = slot["state"].buffer.shape[0] + chunk.size
        if not slot["real_audio"] and not np.any(chunk):
            return self._drain()
        slot["real_audio"] = True
        await self._stream_chunk(chunk)
        return self._drain()

    async def _stream_chunk(self, chunk: np.ndarray) -> None:
        slot = self.slot
        committed_before = slot["committed_display"]
        previous_uncommitted = self._uncommitted_display()
        accum_before = slot["state"].audio_accum.shape[0]

        async def on_seg(_state) -> None:
            await self._process_updates()

        async def on_dot(_state) -> None:
            await self._process_updates()

        async def on_partial(_state) -> None:
            self._offer_partial()

        commits_on_seg = not (self.hide_seg or self.always_commit)
        await self._decode_stream(
            chunk, slot["state"],
            on_seg=on_seg if commits_on_seg else None,
            on_dot=on_dot if self.enable_dot_commit and not self.always_commit else None,
            on_partial=on_partial,
        )

        if slot["committed_display"] != committed_before:
            ids = self.model.processor.tokenizer.encode(slot["state"]._raw_decoded)
            slot["state"].committed_token_len = max(
                slot["state"].committed_token_len,
                max(0, len(ids) - slot["state"].unfixed_token_num),
            )

        if slot["state"].hallucination_detected:
            await self._recover_from_hallucination()
            return

        snapshot = await self._process_updates(chunk_end=True)
        await self._reshape_slot(committed_before, previous_uncommitted, accum_before,
                                 snapshot)
        self._offer_partial(force=True)

    async def _recover_from_hallucination(self) -> None:
        state = self.slot["state"]
        state.hallucination_detected = False
        if not (state.text or "").strip() and getattr(state, "_last_nonempty_text", ""):
            state.text = state._last_nonempty_text
        log.info("[HALLUC-CUT] text=%r", strip_asr_text((state.text or "").strip()))
        await self._flush_uncommitted(reason="vad")
        self._reset_slot()
        self._offer_partial(force=True)

    async def _reshape_slot(self, committed_before: str, previous_uncommitted: str,
                            accum_before: int, snapshot: str | None) -> None:
        slot = self.slot
        committed = slot["committed_display"]
        any_commit = committed != committed_before
        decoded = strip_asr_text((slot["state"].text or "").strip())
        header_tail = bool(SEG_HEADER_TAIL_RE.search(decoded))
        chunk_samples = int(round(float(self.settings.get("chunk_size_sec", 2.0))
                                  * audio_mod.SAMPLING_RATE))
        if any_commit:
            slot["seg_chunk_accum"] = accum_before

        if not any_commit and header_tail:
            accum = slot["state"].audio_accum
            since = slot.get("seg_chunk_accum")
            if (since is None or since >= accum.shape[0]
                    or accum.shape[0] - since > 3 * chunk_samples):
                since = max(0, accum.shape[0] - chunk_samples)
            self._restart_with_audio(accum[since:].copy(), committed,
                                     header_reset=True)
            log.info("[SEG-HEADER-RESET] tail=%r carry=%.2fs", decoded[-40:],
                        (accum.shape[0] - since) / audio_mod.SAMPLING_RATE)
            self._offer_partial(force=True)
            return

        if not any_commit:
            return

        ends_with_seg = decoded.endswith(SEG_TAG) or header_tail
        remaining = "" if ends_with_seg else self._uncommitted_display(snapshot)
        accum = slot["state"].audio_accum
        audio_sec = accum.shape[0] / audio_mod.SAMPLING_RATE

        if audio_sec > MAX_AUDIO_ACCUM_SEC:
            seed_text = self._forced_reset_seed(remaining)
            self._reset_slot(seed_text=seed_text)
            committed_part = seed_text[: len(seed_text) - len(remaining)]
            self.slot.update(
                committed_display=committed_part,
                committed_seg_count=committed_part.count(SEG_TAG),
                last_text=seed_text,
            )
            self.slot["state"].unfixed_token_num = 0
            log.info("[FORCE-SLOT-SWITCH] audio_sec=%.1fs", audio_sec)
            return

        if ends_with_seg or not remaining.strip():
            carry_samples = accum.shape[0] - accum_before if header_tail else 0
            if header_tail:
                carry_samples = min(max(carry_samples, chunk_samples), accum.shape[0])
            carry = accum[-carry_samples:].copy() if carry_samples > 0 else None
            self._restart_with_audio(carry, committed, header_reset=header_tail)
            log.info("[SEG-SLOT-RESET] audio_sec=%.1fs carry=%.2fs", audio_sec,
                        0.0 if carry is None else carry.shape[0] / audio_mod.SAMPLING_RATE)
            return

        if self._period_stayed_in_place(previous_uncommitted, committed_before, committed):
            carry = (accum[-chunk_samples:] if accum.shape[0] >= chunk_samples
                     else accum.copy())
            carry_lang = slot["last_text_lang"]
            carry_boundaries = slot.get("prev_boundary_sentences", ())
            self._restart_with_audio(carry, committed, dot_switch=True)
            if carry_lang:
                self.slot["last_text_lang"] = carry_lang
            if carry_boundaries:
                self.slot["prev_boundary_sentences"] = carry_boundaries
                self.slot["prev_boundary_accum"] = -1
            log.info("[DOT-SLOT-SWITCH] audio_sec=%.1fs prev=%r", audio_sec,
                        previous_uncommitted.strip())

    def _restart_with_audio(self, carry: np.ndarray | None, last_committed: str, *,
                            header_reset: bool = False, dot_switch: bool = False) -> None:
        self._reset_slot()
        slot = self.slot
        if carry is not None:
            slot["state"].audio_accum = carry
        if header_reset:
            slot["real_audio"] = True
        if not last_committed:
            return
        if dot_switch:
            slot["dot_switch_prev_committed"] = last_committed
            return
        slot["seg_reset_last_committed"] = last_committed
        if header_reset:
            slot["header_reset_last_committed"] = last_committed

    @staticmethod
    def _period_stayed_in_place(previous_uncommitted: str, committed_before: str,
                                committed_after: str) -> bool:
        previous = previous_uncommitted.strip()
        if not previous or not re.search(r'[.?!。？！]$', previous):
            return False
        newly = committed_after[len(committed_before):]
        core = re.sub(r'\s+', '', previous.rstrip('.?!。？！'))
        return bool(core and re.match(re.escape(core) + r'[.?!。？！]',
                                      re.sub(r'\s+', '', newly)))

    def _forced_reset_seed(self, remaining: str) -> str:
        committed = self.slot["committed_display"]
        sentences = [s for s in re.split(r'(?<=[。？！?!])', committed) if s.strip()]
        return "".join(sentences[-MAX_SEED_COMMITTED_SENTENCES:]) + remaining

    def _uncommitted_display(self, text_snapshot: str | None = None) -> str:
        slot = self.slot
        current = (text_snapshot if text_snapshot is not None
                   else strip_asr_text((slot["state"].text or "").strip()))
        return display_of(uncommitted_from(
            current, slot["committed_display"], slot["committed_seg_count"]))

    def _offer_partial(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._partial_at < PARTIAL_MIN_INTERVAL_SEC:
            return
        text = self._uncommitted_display()
        if not text and not force:
            return
        previous = self._partial_text
        if not force and previous and text != previous and previous.startswith(text):
            return
        if text == previous:
            self._partial_at = now
            return
        self._partial_text = text
        self._partial_at = now
        self._partial_seq += 1
        self._out.append(Partial(
            text=text,
            language=langs.norm_code(self.slot["last_text_lang"]),
            seq=self._partial_seq))

    def _clear_partial(self) -> None:
        if self._partial_text == "":
            return
        self._partial_text = ""
        self._partial_at = time.perf_counter()
        self._partial_seq += 1
        self._out.append(Partial(text="", language="", seq=self._partial_seq))

    def _language(self) -> str:
        return langs.norm_code(self.slot["state"].language
                               or self.slot["last_text_lang"] or "")

    def _commit_skip_reason(self, sentence: str, *, stage: str,
                            trigger: str | None = None,
                            batch_last: str | None = None,
                            batch_repeat: int = 0) -> str | None:
        slot = self.slot
        applies = lambda name: stage in self.GUARD_STAGES[name]
        if not sentence:
            return None

        if (applies("rep-dedup") and self.rep_dedup and batch_last is not None
                and sentence == batch_last and batch_repeat >= REP_DEDUP_MAX_REPEATS):
            return "rep-dedup"

        if (applies("seg-boundary-dedup") and "seg_reset_last_committed" in slot
                and not self.always_commit):
            previous = slot.pop("seg_reset_last_committed")
            words = sentence.split()
            first_word = bare_word(words[0]) if words else ""
            last_word = bare_word(previous.split()[-1]) if previous.split() else ""
            if first_word and last_word and (first_word == last_word
                                             or last_word.endswith(first_word)):
                return "seg-boundary-dedup"
            if tail_overlaps(previous, sentence):
                return "seg-boundary-dedup"

        if applies("header-reset-tail-dedup") and "header_reset_last_committed" in slot:
            if tail_overlaps(slot.pop("header_reset_last_committed"), sentence):
                return "header-reset-tail-dedup"

        if applies("dot-suffix-dedup"):
            previous = slot.get("dot_switch_prev_committed", "")
            if previous and (trigger == "dot" or stage == "flush"):
                slot.pop("dot_switch_prev_committed", None)
                drop = lambda s: DROP_PUNCT_SPACE_RE.sub('', s)
                if drop(previous).endswith(drop(sentence)):
                    return "dot-suffix-dedup"

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
                        return "committed-suffix-dedup"

        if applies("cross-dedup") and asr_key(sentence) in slot["committed_asr_set"]:
            return "cross-dedup"

        if (applies("cross-dedup-fuzzy") and not self.always_commit
                and self._cross_dup_match(sentence) is not None):
            return "cross-dedup-fuzzy"

        return None

    def _cross_dup_match(self, sentence: str) -> str | None:
        key = fuzzy_key(sentence)
        if not key or len(key.split()) < FUZZY_DEDUP_MIN_WORDS:
            return None
        for previous in self.slot["committed_fuzzy_keys"]:
            if difflib.SequenceMatcher(None, key, previous).ratio() >= FUZZY_DEDUP_RATIO:
                return previous
        return None

    def _strip_committed_prefix(self, sentence: str) -> str:
        if self.always_commit or not sentence:
            return sentence
        words = sentence.split()
        normalized = [re.sub(r"[^a-z']", '', w.lower()) for w in words]
        for previous in self.slot["committed_fuzzy_keys"]:
            previous_words = previous.split()
            if not previous_words or len(normalized) <= len(previous_words):
                continue
            head = ' '.join(normalized[:len(previous_words)])
            if difflib.SequenceMatcher(None, head, previous).ratio() >= FUZZY_DEDUP_RATIO:
                return ' '.join(words[len(previous_words):]).lstrip(' ,.;:!?')
        return sentence

    def _apply_commit_guards(self, text: str) -> str:
        if self.always_commit or not text:
            return text
        kept, batch_last = [], None
        for sentence in _boundary().split_sentences(text):
            shown = sentence.strip()
            if not shown:
                continue
            reason = self._commit_skip_reason(shown, stage="flush", batch_last=batch_last)
            if reason:
                log.info("[COMMIT-SKIP] reason=%s stage=flush text=%r", reason, shown)
                continue
            trimmed = self._strip_committed_prefix(shown)
            if not trimmed:
                continue
            if trimmed != shown:
                log.info("[COMMIT-TRIM] reason=committed-prefix stage=flush kept=%r",
                            trimmed)
                sentence, shown = trimmed + " ", trimmed
            kept.append(sentence)
            batch_last = shown
        return "".join(kept).strip()

    async def _process_updates(self, force_reason: str | None = None,
                               chunk_end: bool = False,
                               final: bool = False) -> str | None:
        slot = self.slot
        state = slot["state"]
        current_text = strip_asr_text((state.text or "").strip())
        current_lang = state.language or ""
        recheck = chunk_end and (self.dot_commit_confirm or final)
        if not current_text or (current_text == slot["last_text"] and not recheck):
            return None

        slot["last_text"] = current_text
        slot["last_text_lang"] = current_lang

        uncommitted = uncommitted_from(
            current_text, slot["committed_display"], slot["committed_seg_count"])
        extracted = self._extract_sentences(uncommitted, chunk_end=chunk_end, final=final)
        if not extracted:
            return None

        latest_text = (state.text or "").strip()
        committed_items = self._advance_cursor(latest_text, extracted, force_reason)
        if not committed_items:
            return None

        committed_items = self._merge_comma_fragments(
            committed_items, closing=bool(force_reason))
        committed_items = [(text, trigger) for text, trigger in committed_items
                           if not self._drop_before_translate(
                               text, force_reason or trigger)]
        for sentence, trigger in committed_items:
            self._emit_final(sentence, force_reason or trigger)
        return strip_asr_text(latest_text)

    def _extract_sentences(self, uncommitted: str, *, chunk_end: bool,
                           final: bool) -> list:
        slot = self.slot
        state = slot["state"]
        extracted: list = []
        remaining = uncommitted
        batch_last, batch_repeat = None, 0
        stalled, previous_boundaries, previous_accum = False, (), -1

        if chunk_end and self.dot_commit_confirm:
            accum = int(state.audio_accum.shape[0]) if state.audio_accum is not None else 0
            previous_boundaries = slot.get("prev_boundary_sentences", ())
            previous_accum = slot.get("prev_boundary_accum", -1)
            slot["prev_boundary_sentences"] = self._extract_boundary_keys(uncommitted)
            slot["prev_boundary_accum"] = accum
            if self.dot_commit_stall_chunks > 0:
                tokens = self._count_tokens(uncommitted)
                committed_len = len(slot["committed_display"])
                same_base = committed_len == slot.get("stall_committed_len")
                if (same_base and tokens == slot.get("stall_tokens")
                        and accum > slot.get("stall_accum", -1)):
                    slot["stall_count"] = slot.get("stall_count", 0) + 1
                else:
                    slot["stall_count"] = 0
                    slot["stall_tokens"] = tokens
                slot["stall_committed_len"] = committed_len
                slot["stall_accum"] = accum
                stalled = slot["stall_count"] >= self.dot_commit_stall_chunks

        def take(sentence: str, trigger: str) -> None:
            nonlocal batch_last, batch_repeat
            shown = sentence.replace(SEG_TAG, "").strip()
            if not shown:
                return
            reason = self._commit_skip_reason(
                shown, stage="extract", trigger=trigger,
                batch_last=batch_last, batch_repeat=batch_repeat)
            if reason:
                log.info("[COMMIT-SKIP] reason=%s stage=extract text=%r", reason, shown)
                extracted.append((sentence, trigger, False))
                return
            extracted.append((sentence, trigger, True))
            slot.pop("pending_dot_text", None)
            slot.pop("pending_dot_accum", None)
            batch_repeat = batch_repeat + 1 if shown == batch_last else 1
            batch_last = shown

        while True:
            if self.always_commit:
                if not remaining.strip():
                    break
                take(remaining.strip(), "always")
                break

            match = self._next_boundary(remaining)
            if not match:
                if not (final and remaining.strip()):
                    break
                trigger = "dot" if self.enable_dot_commit else "seg"
                log.info("[COMMIT-RESIDUAL] rule=final trigger=%s text=%r",
                            trigger, remaining.strip())
                take(remaining.strip(), trigger)
                break

            trigger = "seg" if SEG_TAG in match.group() else "dot"
            sentence = remaining[:match.end()].strip()
            after = remaining[match.end():]

            if trigger == "dot" and self.dot_commit_confirm:
                accum = (int(state.audio_accum.shape[0])
                         if state.audio_accum is not None else 0)
                if self._confirm_dot(sentence, after, accum, chunk_end=chunk_end,
                                     final=final, stalled=stalled,
                                     previous_boundaries=previous_boundaries,
                                     previous_accum=previous_accum) is False:
                    if chunk_end and slot.get("pending_dot_text") != sentence:
                        log.info("[DOT-PENDING] text=%r", sentence)
                        slot["pending_dot_text"] = sentence
                        slot["pending_dot_accum"] = accum
                    break

            take(sentence, trigger)
            remaining = after

        return extracted

    def _confirm_dot(self, sentence: str, after: str, accum: int, *, chunk_end: bool,
                     final: bool, stalled: bool, previous_boundaries: tuple,
                     previous_accum: int) -> bool:
        slot = self.slot
        state = slot["state"]
        tail_tokens = self._count_tokens(after)
        rule = None
        if tail_tokens > state.unfixed_token_num:
            rule = "context"
        elif (chunk_end and boundary_key(sentence) in previous_boundaries
              and accum > previous_accum):
            rule = "stable"
        elif final:
            rule = "final"
        elif stalled and not after.strip():
            rule = "stall"
            slot["stall_count"] = 0
        if rule is None:
            return False
        slot.pop("pending_dot_text", None)
        slot.pop("pending_dot_accum", None)
        log.info("[DOT-CONFIRM] rule=%s text=%r", rule, sentence)
        return True

    def _advance_cursor(self, latest_text: str, extracted: list,
                        force_reason: str | None) -> list:
        slot = self.slot
        committed_items: list = []
        consumed_seg, consumed_any = 0, False

        if force_reason == "vad":
            shown_text = display_of(latest_text)
            committed = slot["committed_display"]
            pos = len(committed) if committed and shown_text.startswith(committed) else 0
            for sentence_raw, trigger, emit in extracted:
                sentence = sentence_raw.replace(SEG_TAG, "").strip()
                if emit and self._skip_at_commit(sentence, trigger):
                    emit = False
                core = sentence.rstrip(PUNCT)
                tail = shown_text[pos:].lstrip()
                lead = len(shown_text[pos:]) - len(tail)
                if not (core and tail.startswith(core)):
                    break
                end = len(core)
                if end < len(tail) and tail[end].isalpha():
                    break
                while end < len(tail) and tail[end] in PUNCT:
                    end += 1
                pos += lead + end
                if not emit:
                    consumed_any = True
                    continue
                sentence = self._strip_committed_prefix(sentence)
                if sentence:
                    committed_items.append((sentence, trigger))
            if committed_items or consumed_any:
                slot["committed_display"] = shown_text[:pos].strip()
                self._remember_committed(committed_items)
            return committed_items

        cursor = committed_cursor(latest_text, slot["committed_display"],
                                  slot["committed_seg_count"])
        if cursor == -1:
            cursor, found = 0, 0
            while found < slot["committed_seg_count"]:
                idx = latest_text.find(SEG_TAG, cursor)
                if idx == -1:
                    return []
                cursor = idx + len(SEG_TAG)
                found += 1
        tail = latest_text[cursor:]
        for sentence_raw, trigger, emit in extracted:
            sentence = sentence_raw.replace(SEG_TAG, "").strip()
            if emit and self._skip_at_commit(sentence, trigger):
                emit = False
            if not emit:
                stripped = tail.lstrip()
                while stripped.startswith(SEG_TAG):
                    stripped = stripped[len(SEG_TAG):].lstrip()
                if stripped.startswith(sentence_raw):
                    cursor += len(tail) - len(stripped) + len(sentence_raw)
                    tail = latest_text[cursor:]
                    consumed_seg += sentence_raw.count(SEG_TAG)
                    consumed_any = True
                continue
            sentence = self._strip_committed_prefix(sentence)
            if not sentence:
                continue
            stripped = tail.lstrip()
            while stripped.startswith(SEG_TAG):
                stripped = stripped[len(SEG_TAG):].lstrip()
            if not stripped.startswith(sentence_raw):
                break
            cursor += len(tail) - len(stripped) + len(sentence_raw)
            tail = latest_text[cursor:]
            committed_items.append((sentence, trigger))
        if committed_items or consumed_any:
            slot["committed_display"] = display_of(latest_text[:cursor])
            slot["committed_seg_count"] += (
                sum(1 for _, trigger in committed_items if trigger == "seg")
                + consumed_seg)
            self._remember_committed(committed_items)
        return committed_items

    def _skip_at_commit(self, sentence: str, trigger: str) -> bool:
        reason = self._commit_skip_reason(sentence, stage="commit", trigger=trigger)
        if reason:
            log.info("[COMMIT-SKIP] reason=%s stage=commit text=%r", reason, sentence)
        return bool(reason)

    def _remember_committed(self, committed_items: list) -> None:
        slot = self.slot
        slot["audio_anchor_sec"] = self._audio_sec()
        slot["committed_asr_set"].update(asr_key(t) for t, _ in committed_items)
        slot["committed_fuzzy_keys"].extend(
            key for key in (fuzzy_key(t) for t, _ in committed_items) if key)

    def _merge_comma_fragments(self, committed_items: list, closing: bool) -> list:
        items = list(committed_items)
        deferred, self._deferred_fragment = self._deferred_fragment, ""
        if deferred:
            if items:
                text, trigger = items[0]
                items[0] = (deferred + " " + text, trigger)
                log.info("[FRAGMENT-JOIN] %r + %r", deferred, text)
            else:
                items = [(deferred, "seg")]
        merged: list = []
        for text, trigger in items:
            if merged and is_short_fragment(merged[-1][0]):
                merged[-1] = (merged[-1][0] + " " + text, trigger)
            else:
                merged.append((text, trigger))
        if not closing and merged and is_short_fragment(merged[-1][0]):
            text, _trigger = merged.pop()
            self._deferred_fragment = text
            log.info("[FRAGMENT-DEFER] text=%r", text)
        return merged

    def _drop_before_translate(self, original: str, reason: str) -> bool:
        text = strip_lang_headers(original or "")
        if not text:
            log.info("[HEADER-ONLY-DROP] reason=%s text=%r", reason, original)
            return True
        if KNOWN_SILENCE_RE.match(text):
            log.info("[HALLUC-DROP] reason=%s text=%r", reason, text)
            return True
        if not WORD_CHAR_RE.search(text):
            log.info("[EMPTY-DROP] reason=%s text=%r", reason, text)
            return True
        if reason in ("vad", "finish"):
            stripped = text.strip()
            if (LONE_COMMA_WORD_RE.fullmatch(stripped)
                    or KO_TAIL_FILLER_RE.fullmatch(stripped)):
                log.info("[TAIL-DROP] reason=%s text=%r", reason, text)
                return True
        return False

    def _is_silence_hallucination(self, original: str, reason: str,
                                  audio_end_sec: float) -> bool:
        spans = None if self.vad is None else self.vad.spans
        if not original.strip() or not spans:
            return False
        start, end = self._last_final_end_sec, audio_end_sec
        if end <= start:
            return False
        now = self._audio_sec()
        for span_start, span_end in spans:
            overlap = (min(end, now if span_end is None else span_end)
                       - max(start, span_start))
            if overlap > MIN_SPEECH_OVERLAP_SEC:
                return False
        log.info("[SILENCE-DROP] reason=%s span=(%.1f~%.1f) text=%r",
                    reason, start, end, original[:40])
        return True

    def _emit_final(self, original: str, reason: str) -> None:
        if HEADER_ONLY_RE.match(original or ""):
            log.info("[HEADER-ONLY-DROP] reason=%s text=%r", reason, original)
            return
        stripped = strip_lang_headers(original)
        if stripped != original:
            log.info("[HEADER-STRIP] reason=%s text=%r -> %r",
                        reason, original, stripped)
            original = stripped
            if not original:
                return
        if KNOWN_SILENCE_RE.match(original):
            log.info("[HALLUC-DROP] reason=%s text=%r", reason, original)
            return
        if not WORD_CHAR_RE.search(original):
            log.info("[EMPTY-DROP] reason=%s text=%r", reason, original)
            return
        if reason in ("vad", "finish") and LONE_COMMA_WORD_RE.fullmatch(original.strip()):
            log.info("[TAIL-DROP] reason=%s text=%r", reason, original)
            return
        audio_end_sec = self._audio_sec()
        if self._is_silence_hallucination(original, reason, audio_end_sec):
            return
        self._last_final_end_sec = audio_end_sec
        self._last_final_text = original
        self._out.append(Transcribed(
            original=original,
            language=self._language(),
            commit_reason=reason,
            decision_audio_sec=round(audio_end_sec, 3),
            recv_elapsed_sec=round(timing.elapsed() or 0.0, 4),
        ))

    async def _flush_uncommitted(self, reason: str) -> None:
        slot = self.slot
        slot.pop("pending_dot_text", None)
        slot.pop("pending_dot_accum", None)
        state = slot["state"]
        current_text = strip_asr_text((state.text or "").strip())
        text = display_of(uncommitted_from(
            current_text, slot["committed_display"], slot["committed_seg_count"]))
        text = collapse_repetition(text)
        deferred, self._deferred_fragment = self._deferred_fragment, ""
        if deferred:
            text = (deferred + " " + text).strip()
            log.info("[FRAGMENT-JOIN] %r + flush", deferred)
        text = self._apply_commit_guards(text)
        if not text:
            return
        text = self._strip_dot_switch_tail(text)
        if not text:
            return
        if self._drop_before_translate(text, reason):
            return

        parts = [text]
        if not self.always_commit:
            split, remaining = [], text
            while True:
                match = _boundary().DOT_COMMIT_BOUNDARY_RE.search(remaining)
                if not match:
                    break
                head = remaining[:match.end()].strip()
                if head:
                    split.append(head)
                remaining = remaining[match.end():]
            if remaining.strip():
                split.append(remaining.strip())
            if len(split) > 1:
                parts = split
                log.info("[FLUSH-SPLIT] %d sentences text=%r", len(parts), text[:80])

        for part in parts:
            self._emit_final(part, reason)
        slot["committed_display"] = display_of(current_text)
        slot["committed_seg_count"] = current_text.count(SEG_TAG)
        slot["audio_anchor_sec"] = self._audio_sec()

    def _strip_dot_switch_tail(self, text: str) -> str:
        previous = self.slot.pop("dot_switch_prev_committed", "")
        if not (previous and text):
            return text
        sentences = re.split(r'(?<=[.!?。！？])\s+', previous.strip())
        last = sentences[-1].strip() if sentences else ""
        if not last:
            return text
        drop = lambda s: DROP_PUNCT_SPACE_RE.sub('', s)
        if not drop(text).startswith(drop(last)):
            return text
        skip, counted = 0, 0
        for char in text:
            if counted >= len(drop(last)):
                break
            skip += 1
            if not DROP_PUNCT_SPACE_RE.match(char):
                counted += 1
        while skip < len(text) and text[skip] in PUNCT + ' ':
            skip += 1
        stripped = text[skip:].strip()
        log.info("[COMMIT-SKIP] reason=dot-suffix-dedup stripped=%r", stripped)
        return stripped

    def _trim_tail_silence(self, speech: Speech | None) -> None:
        if speech is None:
            return
        cut = int(max(0.0, speech.silence_waited_out_sec - KEEP_AFTER_SPEECH_SEC)
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
        log.info("[TAIL-TRIM] %.3fs", cut / audio_mod.SAMPLING_RATE)

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
                                  - KEEP_AFTER_SPEECH_SEC) * audio_mod.SAMPLING_RATE)
                       - slot["tail_trim_samples"])
        end = audio.shape[0] - tail_cut
        if end <= start:
            return
        state = self._new_state()
        state.buffer = audio[start:end].copy()
        await self._decode(state)
        retried = (state.text or "").strip()
        log.info("[VAD-RETRY] %.3fs text=%r",
                 (end - start) / audio_mod.SAMPLING_RATE, retried)
        if retried:
            slot["state"] = state
            self.state = state

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        slot = self.slot
        before = (slot["state"].text or "").strip()
        uncommitted_before = uncommitted_from(
            before, slot["committed_display"], slot["committed_seg_count"])
        await self._process_updates(force_reason=reason)
        state = self.slot["state"]
        short_utterance = (self._accum_before_chunk == 0
                           and self._buffered_before_chunk > 0)
        if before or state.buffer.shape[0] > 0 or short_utterance:
            self._trim_tail_silence(speech)
            await self._decode(state)
            if not (state.text or "").strip() and speech is not None:
                await self._retry_short_utterance(speech)
        state = self.slot["state"]
        after = (state.text or "").strip()
        uncommitted_after = uncommitted_from(
            after, self.slot["committed_display"], self.slot["committed_seg_count"])
        if not uncommitted_after and uncommitted_before:
            state.text = before
        await self._flush_uncommitted(reason=reason)
        self._reset_slot()
        return self._drain()

    async def finish(self, reason: str = "finish",
                     speech: Speech | None = None) -> list:
        self._trim_tail_silence(speech)
        await self._decode(self.slot["state"])
        await self._process_updates(chunk_end=True, final=True)
        await self._flush_uncommitted(reason=reason)
        self._clear_partial()
        return self._drain()
