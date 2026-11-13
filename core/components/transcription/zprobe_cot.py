"""zprobe-cot: one adapter writes, every step, the new source words and then the new translation words.

The decoded stream is append-only interleaved deltas (core/research/zprobe p14cot corpus):

    language Interpret<asr_text>{src}<asr_text>{tgt}\\n{src}<asr_text>{tgt}\\n ...

A step that hears nothing new decodes to EOS right away. Only accepted (non-empty) steps enter the history,
re-serialized in the canonical training form, so the prompt always looks like a training `cond`.
The direction comes from the transcript itself: Hangul-heavy transcript -> ko->en, else en->ko.
"""
import re

from core.utils import audio as audio_mod
from core.utils import cache
from core.utils import logging
from core.utils import timing

from . import transcribers
from ..registry import Partial, Speech, Transcribed, Translated
from .base import Transcriber
from .qwen3 import resolve_model

log = logging.getLogger(__name__)

EOS_TOKENS = (151645, 151643)
SPECIAL = re.compile(r"<\|[^<>|]+\|>")
REPEAT_TEMPLATE = r"((.{1,12}?))(?:\2){%d,}"
HEAD = "language Interpret<asr_text>"
SEP = "<asr_text>"
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")


def parse_steps(text: str, final: bool) -> list[tuple[str, str]]:
    """Decoded continuation -> [(src_delta, tgt_delta)].

    Mid-stream only the first finished line counts (no newline yet = the step is not finished = no output);
    on the final decode every finished line plus the unfinished tail. A line without the separator is all
    source. Both parts are stripped; a line with neither is dropped."""
    text = SPECIAL.sub("", text)
    lines = text.split("\n")
    done, tail = lines[:-1], lines[-1]
    if not final:
        done = done[:1]
        tail = ""
    out = []
    for line in done + ([tail] if tail.strip() else []):
        src, _, tgt = line.partition(SEP)
        src, tgt = src.strip(), tgt.strip()
        if src or tgt:
            out.append((src, tgt))
    return out


def step_line(src: str, tgt: str) -> str:
    return f"{src.strip()}{SEP}{tgt.strip()}\n"


def hangul_share(text: str) -> float:
    h, l = len(_HANGUL.findall(text)), len(_LATIN.findall(text))
    return h / (h + l) if h + l else 0.0


def source_lang(transcript: str, tgt: str = "") -> str:
    """ko if the transcript is mostly Hangul, else en; with no transcript yet, the opposite of the translation."""
    if _HANGUL.search(transcript) or _LATIN.search(transcript):
        return "ko" if hangul_share(transcript) > 0.5 else "en"
    return "en" if hangul_share(tgt) > 0.5 else "ko"


def other_lang(lang: str) -> str:
    return "en" if lang == "ko" else "ko"


@transcribers.register("zprobe-cot")
class ZprobeCotTranscription(Transcriber):

    SETTINGS = {
        "model": ("model_path", resolve_model),
        "step_sec": ("step_sec", float),
        "first_sec": ("first_sec", float),
        "max_new_tokens": ("max_new_tokens", int),
        "max_utterance_sec": ("max_utterance_sec", float),
        "repeat_cap": ("repeat_cap", int),
        "head": ("head", str),
        "flush_keep_sec": ("flush_keep_sec", float),
    }

    model = None

    @classmethod
    def validate(cls, options: dict, *, kind: str = "transcription") -> dict:
        out = super().validate(options, kind=kind)
        if not out.get("model_path"):
            raise ValueError("transcription zprobe-cot needs `model` (a merged checkpoint directory)")
        return out

    async def load(self) -> None:
        self.model = await cache.load(("zprobe-cot", self.settings["model_path"]), self._build_model)

    async def _build_model(self):
        import torch
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel.from_pretrained(
            self.settings["model_path"], dtype=torch.bfloat16, device_map="cuda",
            max_inference_batch_size=1)

    def start(self, languages: list[str] | None = None, *, target_lang: str = "",
              vad=None, **_) -> None:
        import numpy as np

        self._vad = vad
        self._np = np
        self._base = self.model._build_text_prompt(context="", force_language=None) \
            + self.settings.get("head", HEAD)
        self._tok = self.model.processor.tokenizer
        self._tok.padding_side = "left"
        self._dev = next(self.model.model.thinker.parameters()).device
        self._fed_samples = 0
        self._seq = 0
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._wav = self._np.empty(0, dtype=self._np.float32)
        self._hist = ""
        self._src: list[str] = []
        self._decoded_samples = 0

    def _in_speech(self) -> bool:
        spans = getattr(self._vad, "spans", None)
        if spans is None:
            return True
        return bool(spans) and spans[-1][1] is None

    def _has_speech(self) -> bool:
        spans = getattr(self._vad, "spans", None)
        return spans is None or bool(spans)

    async def transcribe(self, audio: bytes) -> list:
        self._fed_samples += len(audio) // 2
        self._wav = self._np.concatenate([self._wav, audio_mod.from_pcm_bytes(audio)])
        if not self._hist and not self._in_speech():
            return []
        cap = float(self.settings.get("max_utterance_sec", 14.0))
        if self._wav.shape[0] >= int(cap * audio_mod.SAMPLING_RATE):
            out = self._emit(self._decode(final=True), final=True) + self._close("cap")
            self._reset_utterance()
            return out
        step = int(float(self.settings.get("step_sec", 0.5)) * audio_mod.SAMPLING_RATE)
        first = int(float(self.settings.get("first_sec", 1.0)) * audio_mod.SAMPLING_RATE)
        if self._wav.shape[0] < first or self._wav.shape[0] - self._decoded_samples < step:
            return []
        return self._emit(self._decode(final=False), final=False)

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        return await self.finish(reason, speech)

    async def finish(self, reason: str = "finish",
                     speech: Speech | None = None) -> list:
        if speech is not None:
            keep = float(self.settings.get("flush_keep_sec", 0.1))
            cut = int(max(0.0, speech.silence_waited_out_sec - keep) * audio_mod.SAMPLING_RATE)
            if 0 < cut < self._wav.shape[0]:
                self._wav = self._wav[: self._wav.shape[0] - cut]
        out = []
        if self._wav.shape[0] and (self._hist or self._has_speech()):
            out = self._emit(self._decode(final=True), final=True)
        out += self._close(reason)
        self._reset_utterance()
        return out

    def _transcript(self) -> str:
        return " ".join(self._src)

    def _close(self, reason: str) -> list:
        text = self._transcript().strip()
        if not text:
            return []
        elapsed = round(timing.elapsed() or 0.0, 4)
        return [Transcribed(
            original=text,
            language=source_lang(text),
            commit_reason=reason,
            decision_audio_sec=round(self._fed_samples / audio_mod.SAMPLING_RATE, 3),
            committed_elapsed_sec=elapsed,
        )]

    def _emit(self, text: str, *, final: bool) -> list:
        out = []
        for src, tgt in parse_steps(text, final):
            self._hist += step_line(src, tgt)
            if src:
                self._src.append(src)
            lang = source_lang(self._transcript(), tgt)
            elapsed = round(timing.elapsed() or 0.0, 4)
            if src:
                self._seq += 1
                out.append(Partial(text=self._transcript(), language=lang, seq=self._seq))
            if tgt:
                out.append(Translated(
                    original=src,
                    translation=tgt,
                    language=lang,
                    target_lang=other_lang(lang),
                    commit_reason="cot",
                    decision_audio_sec=round(self._fed_samples / audio_mod.SAMPLING_RATE, 3),
                    committed_elapsed_sec=elapsed,
                    translated_elapsed_sec=elapsed,
                ))
        return out

    @timing.measure("decode")
    def _decode(self, *, final: bool) -> str:
        import torch

        self._decoded_samples = self._wav.shape[0]
        inputs = self.model.processor(
            text=[self._base + self._hist], audio=[self._wav],
            return_tensors="pt", padding=True).to(self._dev).to(torch.bfloat16)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            o = self.model.model.generate(
                **inputs,
                max_new_tokens=int(self.settings.get("max_new_tokens", 96)),
                no_repeat_ngram_size=None,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        seq = o.sequences if hasattr(o, "sequences") else o
        new = seq[0, prompt_len:]
        while len(new) and int(new[-1]) in EOS_TOKENS:
            new = new[:-1]
        txt = self._tok.decode(new, skip_special_tokens=False)
        while "�" in txt and len(new):
            new = new[:-1]
            txt = self._tok.decode(new, skip_special_tokens=False)
        txt = SPECIAL.sub("", txt)
        cap = int(self.settings.get("repeat_cap", 3) or 0)
        if cap:
            txt = re.sub(REPEAT_TEMPLATE % cap, lambda m: m.group(1) * cap, txt, flags=re.S)
        return txt
