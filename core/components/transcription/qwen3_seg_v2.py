import re

from core.utils import logging

from . import transcribers
from .qwen3_seg import MIN_SPEECH_OVERLAP_SEC, Qwen3SegTranscription, strip_lang_headers

log = logging.getLogger(__name__)

KO_RISKY_FILLER_RE = re.compile(r"(?:(?:아니|응|그렇지|그거)[,.]?\s*)+[!?…]?")


@transcribers.register("qwen-seg:v2")
class Qwen3SegTranscriptionV2(Qwen3SegTranscription):

    SETTINGS = {
        **Qwen3SegTranscription.SETTINGS,
        "silence_window_grace_sec": ("silence_window_grace_sec", float),
    }

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

    def _is_silence_hallucination(self, original: str, reason: str,
                                  audio_end_sec: float) -> bool:
        spans = None if self.vad is None else self.vad.spans
        if not original.strip() or not spans:
            return False
        reset = self._last_vad_reset_sec(spans) + self._silence_window_grace_sec()
        start = max(self._last_final_end_sec, reset)
        end = audio_end_sec
        if end <= start:
            return False
        now = self._audio_sec()
        for span_start, span_end in spans:
            overlap = (min(end, now if span_end is None else span_end)
                       - max(start, span_start))
            if overlap > MIN_SPEECH_OVERLAP_SEC:
                return False
        log.info("[DROP] rule=no-speech gate=emit reason=%s span=(%.1f~%.1f) text=%r",
                 reason, start, end, original)
        return True

    def _drop_commit_candidate(self, original: str, reason: str) -> bool:
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
