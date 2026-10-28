from core.utils import audio as audio_mod
from core.utils import cache
from core.utils import langs
from core.utils import logging
from core.utils import stream
from core.utils import timing

from . import transcribers
from ..registry import Speech
from .base import Transcriber, Transcribed
from .qwen3 import resolve_model

log = logging.getLogger(__name__)

EOS_TOKENS = (151645, 151643)


def _continuation(tok, input_ids, seq) -> str:
    hits = (input_ids == tok.convert_tokens_to_ids("<asr_text>")).nonzero()
    start = int(hits[-1]) + 1 if len(hits) else 0  # auto mode: the model writes <asr_text> itself, the prompt has none
    pre = tok.decode(input_ids[start:], skip_special_tokens=False)
    full = tok.decode(seq[start:], skip_special_tokens=False)
    full = full.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return full[len(pre):]


@transcribers.register("zprobe-prefix")
class ZprobePrefixTranscription(Transcriber):
    """The prefix-trained direct speech-to-translation adapter, append-only: every step_sec it continues the text it
    has already shown from the audio heard so far, and what it writes is never revised. The emitted text is in the
    run's target language, so the cascade's translate() passes it through as its own translation. The last decode of
    an utterance flushes (EOS banned for two tokens, n-gram ban lifted): EOS doubles as "wait" and waiting past the
    end of the utterance is just truncation."""

    SETTINGS = {
        "model": ("model_path", resolve_model),
        "step_sec": ("step_sec", float),
        "first_sec": ("first_sec", float),
        "max_new_tokens": ("max_new_tokens", int),
        "no_repeat": ("no_repeat", int),
        "final_flush": ("final_flush", bool),
        "max_utterance_sec": ("max_utterance_sec", float),
        "target_mode": ("target_mode", str),
        "seg_reset": ("seg_reset", bool),
        "seg_keep_sec": ("seg_keep_sec", float),
        "dual": ("dual", bool),
        "stall_flush_sec": ("stall_flush_sec", float),
        "word_hold": ("word_hold", bool),
    }

    model = None

    @classmethod
    def validate(cls, options: dict, *, kind: str = "transcription") -> dict:
        out = super().validate(options, kind=kind)
        if not out.get("model_path"):
            raise ValueError("transcription zprobe-prefix needs `model` (a merged checkpoint directory)")
        return out

    async def load(self) -> None:
        self.model = await cache.load(("zprobe-prefix", self.settings["model_path"]), self._build_model)

    async def _build_model(self):
        import torch
        from qwen_asr import Qwen3ASRModel

        log.info("[LOAD] model %s", self.settings["model_path"])
        return Qwen3ASRModel.from_pretrained(
            self.settings["model_path"], dtype=torch.bfloat16, device_map="cuda",
            max_inference_batch_size=1)

    def start(self, languages: list[str] | None = None, *, target_lang: str = "",
              vad=None, **_) -> None:
        import numpy as np

        self._vad = vad
        self._target = langs.norm_code(target_lang or "")
        # auto: the prompt does not force the target; the model opens each utterance with its own
        # "language X<asr_text>" header (every training row starts with one, and the data is all
        # cross-direction), so it picks the crossing language per utterance like a built-in router
        self._auto = self.settings.get("target_mode", "forced") == "auto"
        self._base = self.model._build_text_prompt(
            context="", force_language=None if self._auto else langs.CODE_TO_NAME.get(self._target))
        self._tok = self.model.processor.tokenizer
        # dual: every decode carries a second prompt whose header forces the other language, in the same
        # batch. That stream never enters the pipeline (metrics stay comparable); it goes to the event log,
        # where per-utterance alignment against the gold language turns it into a transcription for WER/CER.
        self._dual = bool(self.settings.get("dual"))
        if self._dual:
            others = [langs.norm_code(l) for l in (languages or []) if langs.norm_code(l) != self._target]
            self._dual_lang = others[0] if others else ("en" if self._target == "ko" else "ko")
            self._base2 = self.model._build_text_prompt(
                context="", force_language=langs.CODE_TO_NAME.get(self._dual_lang))
            self._tok.padding_side = "left"
        self._dev = next(self.model.model.thinker.parameters()).device
        self._fed_samples = 0
        self._np = np
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._wav = self._np.empty(0, dtype=self._np.float32)
        self._raw = ""
        self._raw2 = ""
        self._decoded_samples = 0
        self._last_emit_samples = 0
        self._utt_lang = ""

    def _in_speech(self) -> bool:
        spans = getattr(self._vad, "spans", None)
        if spans is None:
            return True
        return bool(spans) and spans[-1][1] is None

    async def transcribe(self, audio: bytes) -> list:
        self._fed_samples += len(audio) // 2
        self._wav = self._np.concatenate([self._wav, audio_mod.from_pcm_bytes(audio)])
        if not self._raw and not self._in_speech():
            return []
        # a VAD segment can run for minutes (crowd noise, no 800 ms gap), far outside the single-sentence
        # training rows; past the cap the decode flushes and the prefix restarts, as if VAD had cut here
        cap = float(self.settings.get("max_utterance_sec", 14.0))
        if self._wav.shape[0] >= int(cap * audio_mod.SAMPLING_RATE):
            out = self._emit(self._decode(final=True), "cap")
            self._reset_utterance()
            return out
        step = int(float(self.settings.get("step_sec", 0.5)) * audio_mod.SAMPLING_RATE)
        first = int(float(self.settings.get("first_sec", 1.0)) * audio_mod.SAMPLING_RATE)
        if self._wav.shape[0] < first or self._wav.shape[0] - self._decoded_samples < step:
            return []
        # EOS doubles as "wait", so a stuck decode shows as seconds of silence while speech keeps arriving;
        # past stall_flush_sec without an emission the decode is forced to say something (same knobs as the
        # final flush), once per stall - the clock restarts at every emission.
        stall = float(self.settings.get("stall_flush_sec", 0) or 0)
        force = bool(stall) and (self._wav.shape[0] - self._last_emit_samples
                                 >= int(stall * audio_mod.SAMPLING_RATE))
        added = self._decode(final=False, force=force)
        # the model marks a finished sentence with its own <SEG> (P ~ 0.93 right after one). Resetting there keeps
        # the text side from ever looking terminal, which is what froze EOS mid-utterance; the audio keeps a short
        # tail so the head of the next sentence is not lost.
        if self.settings.get("seg_reset") and "<SEG>" in added and (self._raw or added.split("<SEG>")[0].strip()):
            out = self._emit(added.split("<SEG>")[0], "seg")
            keep = int(float(self.settings.get("seg_keep_sec", 1.0)) * audio_mod.SAMPLING_RATE)
            tail = self._wav[-keep:] if keep else self._wav[:0]
            self._reset_utterance()
            self._wav = tail
            return out
        # word_hold: a decode may stop mid-word ("예약해" of "예약했습니다") because EOS competes at every token;
        # withholding the text after the last space keeps commits on word boundaries - the held tail is not
        # appended to the prefix, so the next decode regenerates it, completed. A flush emits everything.
        if self.settings.get("word_hold") and not force:
            cut = added.rstrip()
            if not cut or cut[-1] not in ".,!?…%)”’":
                i = added.rfind(" ")
                added = added[: i + 1] if i >= 0 else ""
        return self._emit(added, "prefix")

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        out = await self.finish(reason, speech)
        self._reset_utterance()
        return out

    async def finish(self, reason: str = "finish",
                     speech: Speech | None = None) -> list:
        if speech is not None:
            cut = int(max(0.0, speech.silence_waited_out_sec - 0.1) * audio_mod.SAMPLING_RATE)
            if 0 < cut < self._wav.shape[0]:
                self._wav = self._wav[: self._wav.shape[0] - cut]
        if self._wav.shape[0] == 0 or (not self._raw and not self._has_speech()):
            return []
        return self._emit(self._decode(final=True), reason)

    def _has_speech(self) -> bool:
        spans = getattr(self._vad, "spans", None)
        return spans is None or bool(spans)

    HEADER_RE = None

    def _emit(self, added: str, reason: str) -> list:
        import re
        self._raw += added
        shown = added.replace("<SEG>", " ")
        if self._auto:
            m = re.search(r"language\s+([A-Za-z]+)\s*<asr_text>", self._raw[: 60])
            if m and not self._utt_lang:
                self._utt_lang = langs.norm_code(m.group(1))
            shown = re.sub(r"\s*language\s+[A-Za-z]*\s*(<asr_text>)?\s*", " ", shown)
        if not shown.strip():
            return []
        self._last_emit_samples = self._wav.shape[0]
        return [Transcribed(
            original=shown.strip(),
            language=(self._utt_lang or self._target) if self._auto else self._target,
            commit_reason=reason,
            decision_audio_sec=round(self._fed_samples / audio_mod.SAMPLING_RATE, 3),
            committed_elapsed_sec=round(timing.elapsed() or 0.0, 4),
        )]

    @timing.measure("decode")
    def _decode(self, *, final: bool, force: bool = False) -> str:
        import torch

        self._decoded_samples = self._wav.shape[0]
        flush = (final and bool(self.settings.get("final_flush", True))) or force
        no_repeat = int(self.settings.get("no_repeat", 4))
        texts = [self._base + self._raw]
        if self._dual:
            texts.append(self._base2 + self._raw2)
        inputs = self.model.processor(
            text=texts, audio=[self._wav] * len(texts),
            return_tensors="pt", padding=True).to(self._dev).to(torch.bfloat16)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            o = self.model.model.generate(
                **inputs, max_new_tokens=int(self.settings.get("max_new_tokens", 120)),
                no_repeat_ngram_size=None if flush else (no_repeat or None),
                min_new_tokens=2 if flush else None,
                do_sample=False, pad_token_id=self._tok.eos_token_id)
        seq = o.sequences if hasattr(o, "sequences") else o
        outs = []
        for i in range(len(texts)):
            ids = inputs["input_ids"][i]
            new = seq[i, prompt_len:]
            while len(new) and int(new[-1]) in EOS_TOKENS:
                new = new[:-1]
            txt = _continuation(self._tok, ids, torch.cat([ids, new]))
            while "�" in txt and len(new):
                new = new[:-1]
                txt = _continuation(self._tok, ids, torch.cat([ids, new]))
            outs.append(txt)
        if self._dual:
            self._raw2 += outs[1]
            shown2 = outs[1].replace("<SEG>", " ").strip()
            if shown2:
                stream.record(
                    "dual", original=shown2, language=self._dual_lang,
                    decision_audio_sec=round(self._fed_samples / audio_mod.SAMPLING_RATE, 3),
                    committed_elapsed_sec=round(timing.elapsed() or 0.0, 4))
        txt = outs[0]
        return txt if self.settings.get("seg_reset") else txt.replace("<SEG>", "")
