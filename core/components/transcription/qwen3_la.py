import re
from collections import deque

from core.utils import audio as audio_mod
from core.utils import langs
from core.utils import logging
from core.utils import timing

from . import transcribers
from ..registry import Partial, Speech, Transcribed
from .qwen3 import Qwen3Transcription
from .qwen3_seg import SEG_TAG, strip_asr_text, strip_lang_headers

log = logging.getLogger(__name__)

BOUNDARIES = {
    "sentence": re.compile(r"[.!?。？！]['\"”’)]*$"),
    "clause": re.compile(r"[.!?。？！,;:、，]['\"”’)]*$"),
}


def _boundary(value) -> str:
    if value not in BOUNDARIES:
        raise ValueError(f"expected one of {sorted(BOUNDARIES)}, got {value!r}")
    return value


def words_of(text: str) -> list[str]:
    return strip_lang_headers(text.replace(SEG_TAG, f" {SEG_TAG} ")).split()


def common_prefix(hypotheses) -> list[str]:
    hypotheses = list(hypotheses)
    if not hypotheses:
        return []
    shortest = min(len(h) for h in hypotheses)
    agreed = 0
    while agreed < shortest and all(h[agreed] == hypotheses[0][agreed] for h in hypotheses):
        agreed += 1
    return hypotheses[0][:agreed]


def spoken(words: list[str]) -> str:
    return " ".join(w for w in words if w != SEG_TAG)


@transcribers.register("qwen-la")
class Qwen3LocalAgreement(Qwen3Transcription):
    SETTINGS = {
        **Qwen3Transcription.SETTINGS,
        "agreement": ("agreement", int),
        "boundary": ("boundary", _boundary),
        "min_commit_words": ("min_commit_words", int),
        "max_pending_words": ("max_pending_words", int),
        "max_slot_sec": ("max_slot_sec", float),
    }

    def start(self, languages: list[str] | None = None, **_) -> None:
        self._agreement = max(1, int(self.settings.get("agreement", 2)))
        self._boundary_re = BOUNDARIES[self.settings.get("boundary", "sentence")]
        super().start(languages=languages, **_)

    def _start_stream(self, seed: str = "") -> None:
        super()._start_stream()
        if seed:
            self.state._raw_decoded = seed
            self.state.text = seed
            self.state.chunk_id = self.state.unfixed_chunk_num
        self._history: deque = deque(maxlen=self._agreement)
        self._emitted_words = len(words_of(seed))
        self._shown = None

    async def transcribe(self, audio: bytes) -> list:
        chunk = audio_mod.from_pcm_bytes(audio)
        self._fed_samples += chunk.size
        decodes_before = self.state.chunk_id
        await self._decode_stream(chunk, self.state)
        if self.state.chunk_id != decodes_before:
            self._history.append(words_of(strip_asr_text((self.state.text or "").strip())))
            self._commit_agreed()
            self._trim_if_long()
        self._show_partial()
        return self._drain()

    def _commit_agreed(self) -> None:
        if len(self._history) < self._agreement:
            return
        agreed = common_prefix(self._history)
        if len(agreed) < self._emitted_words:
            log.info("[REVISED] agreed=%d emitted=%d", len(agreed), self._emitted_words)
            return
        pending = agreed[self._emitted_words:]
        cut = self._last_boundary(pending)
        if cut is not None and len([w for w in pending[:cut] if w != SEG_TAG]) >= \
                self.settings.get("min_commit_words", 1):
            self._emit(pending[:cut], "agree")
            return
        waiting = [w for w in pending if w != SEG_TAG]
        if len(waiting) >= self.settings.get("max_pending_words", 12):
            clause = self._last_boundary(pending, clause=True)
            self._emit(pending[:clause or len(pending)], "agree-max")

    def _last_boundary(self, words: list[str], *, clause: bool = False) -> int | None:
        pattern = BOUNDARIES["clause"] if clause else self._boundary_re
        for index in range(len(words) - 1, -1, -1):
            if words[index] == SEG_TAG or pattern.search(words[index]):
                return index + 1
        return None

    def _emit(self, words: list[str], reason: str) -> None:
        self._emitted_words += len(words)
        text = spoken(words)
        if not text:
            return
        self._out.append(Transcribed(
            original=text,
            language=langs.norm_code(self.state.language or ""),
            commit_reason=reason,
            decision_audio_sec=round(self._fed_samples / audio_mod.SAMPLING_RATE, 3),
            committed_elapsed_sec=round(timing.elapsed() or 0.0, 4),
        ))

    def _latest(self) -> list[str]:
        return words_of(strip_asr_text((self.state.text or "").strip()))

    def _trim_if_long(self) -> None:
        limit = self.settings.get("max_slot_sec", 30.0)
        accum = self.state.audio_accum
        if accum is None or accum.shape[0] < limit * audio_mod.SAMPLING_RATE:
            return
        latest = self._latest()
        self._emit(latest[self._emitted_words:], "agree-max")
        seed_words = latest[-self._seed_length(latest):]
        seed = spoken(seed_words) + SEG_TAG
        log.info("[SLOT-RESET] cause=audio-limit audio_sec=%.1f seed=%r",
                 accum.shape[0] / audio_mod.SAMPLING_RATE, seed)
        self._start_stream(seed=seed)

    def _seed_length(self, words: list[str]) -> int:
        for index in range(len(words) - 2, -1, -1):
            if words[index] == SEG_TAG or BOUNDARIES["sentence"].search(words[index]):
                return len(words) - index - 1
        return min(len(words), 12)

    def _show_partial(self) -> None:
        text = spoken(self._latest()[self._emitted_words:])
        if text == self._shown:
            return
        self._shown = text
        self._partial_seq += 1
        self._out.append(Partial(text=text, language=langs.norm_code(self.state.language or ""),
                                 seq=self._partial_seq))

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        out = await self.finish(reason, speech)
        self._start_stream()
        return out

    async def finish(self, reason: str = "finish", speech: Speech | None = None) -> list:
        if speech is not None:
            self._drop_trailing_silence(speech)
        await self._decode(self.state)
        latest = self._latest()
        self._emit(latest[self._emitted_words:], reason)
        self._shown = None
        self._partial_seq += 1
        self._out.append(Partial(text="", language="", seq=self._partial_seq))
        return self._drain()
