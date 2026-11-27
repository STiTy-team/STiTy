"""Where a commit's text was spoken: Qwen3-ForcedAligner over the recent audio."""

import re
import threading

import numpy as np

from core.utils import audio as audio_mod
from core.utils import logging

from .decide import ALIGNER_LANGS

log = logging.getLogger(__name__)

SR = audio_mod.SAMPLING_RATE
DEFAULT_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
MIN_SPAN_SEC = 0.2
_HANGUL = re.compile(r"[가-힣]")
_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")


def tokenizer_language(text: str) -> str:
    """The aligner `language` (word tokenization only, not a language decision)."""
    if _HANGUL.search(text):
        return "Korean"
    if _KANA.search(text):
        return "Japanese"
    if _HAN.search(text):
        return "Chinese"
    return "English"


class AlignerSpan:
    def __init__(self, model: str = DEFAULT_MODEL):
        import torch
        from qwen_asr import Qwen3ForcedAligner

        self.aligner = Qwen3ForcedAligner.from_pretrained(
            model, dtype=torch.bfloat16, device_map="cuda"
        )
        self.lock = threading.Lock()

    def tail_words(
        self, audio: np.ndarray, head: str, tail: str, candidates: tuple[str, ...]
    ) -> list[tuple[str, float, float]] | None:
        """Aligned (text, start, end) items of `tail`, in seconds into `audio`, aligning
        `head + tail` over the audio (the head's item count from aligning the head alone)."""
        if not tail.strip() or len(audio) < SR // 5:
            return None
        if any(c not in ALIGNER_LANGS for c in candidates):
            return None
        text = f"{head} {tail}".strip()
        try:
            with self.lock:
                full = self.aligner.align(
                    audio=(audio, SR), text=text, language=tokenizer_language(text)
                )[0].items
                n_head = (
                    len(
                        self.aligner.align(
                            audio=(audio, SR), text=head, language=tokenizer_language(head)
                        )[0].items
                    )
                    if head.strip()
                    else 0
                )
        except Exception as e:  # noqa: BLE001
            log.warning("[ALIGN-FAILED] %s: %r", e, tail[:40])
            return None
        words = [(it.text, float(it.start_time), float(it.end_time)) for it in full[n_head:]]
        return words or None

    def split_at(
        self, audio: np.ndarray, head: str, tail: str, candidates: tuple[str, ...]
    ) -> float | None:
        """Seconds into `audio` where `tail` starts, aligning `head + tail` over the audio (the
        head's item count comes from aligning the head text alone)."""
        if not tail.strip() or len(audio) < SR // 5:
            return None
        if any(c not in ALIGNER_LANGS for c in candidates):
            return None
        text = f"{head} {tail}".strip()
        try:
            with self.lock:
                full = self.aligner.align(
                    audio=(audio, SR), text=text, language=tokenizer_language(text)
                )[0].items
                n_head = (
                    len(
                        self.aligner.align(
                            audio=(audio, SR), text=head, language=tokenizer_language(head)
                        )[0].items
                    )
                    if head.strip()
                    else 0
                )
        except Exception as e:  # noqa: BLE001
            log.warning("[ALIGN-FAILED] %s: %r", e, tail[:40])
            return None
        if n_head >= len(full):
            return None
        return float(full[n_head].start_time)

    def find(
        self, audio: np.ndarray, audio_start_sec: float, text: str, candidates: tuple[str, ...]
    ) -> tuple[float, float] | None:
        """(start, end) in stream seconds of `text` inside `audio`, or None when the aligner
        cannot place it or the session's languages are outside the aligner's."""
        if not text.strip() or len(audio) < SR // 5:
            return None
        if any(c not in ALIGNER_LANGS for c in candidates):
            return None
        try:
            with self.lock:
                result = self.aligner.align(
                    audio=(audio, SR), text=text, language=tokenizer_language(text)
                )[0]
        except Exception as e:  # noqa: BLE001
            log.warning("[ALIGN-FAILED] %s: %r", e, text[:40])
            return None
        items = [it for it in result.items if it.end_time > it.start_time]
        if not items:
            return None
        start, end = items[0].start_time, items[-1].end_time
        if end - start < MIN_SPAN_SEC:
            return None
        return audio_start_sec + start, audio_start_sec + end
