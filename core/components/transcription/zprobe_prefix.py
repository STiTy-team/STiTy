import re

from core.utils import audio as audio_mod
from core.utils import cache
from core.utils import langs
from core.utils import timing

from . import transcribers
from ..registry import Speech, Translated
from .base import Transcriber
from .qwen3 import resolve_model

EOS_TOKENS = (151645, 151643)
SPECIAL = re.compile(r"<\|[^<>|]+\|>")
REPEAT_TEMPLATE = r"((.{1,12}?))(?:\2){%d,}"
SENTENCE_HEAD = re.compile(r"^.*?[.!?…](?=\s|$)", re.S)


def _continuation(tok, input_ids, seq) -> str:
    hits = (input_ids == tok.convert_tokens_to_ids("<asr_text>")).nonzero()
    start = int(hits[-1]) + 1 if len(hits) else 0
    pre = tok.decode(input_ids[start:], skip_special_tokens=False)
    full = tok.decode(seq[start:], skip_special_tokens=False)
    full = full.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return full[len(pre):]


@transcribers.register("zprobe-prefix")
class ZprobePrefixTranscription(Transcriber):

    SETTINGS = {
        "model": ("model_path", resolve_model),
        "targets": ("targets", list),
        "step_sec": ("step_sec", float),
        "first_sec": ("first_sec", float),
        "max_new_tokens": ("max_new_tokens", int),
        "no_repeat": ("no_repeat", int),
        "max_utterance_sec": ("max_utterance_sec", float),
        "seg_reset": ("seg_reset", bool),
        "seg_keep_sec": ("seg_keep_sec", float),
        "stall_flush_sec": ("stall_flush_sec", float),
        "flush_min_new": ("flush_min_new", int),
        "flush_allow_eos": ("flush_allow_eos", bool),
        "flush_max_new": ("flush_max_new", int),
        "flush_sentence_cut": ("flush_sentence_cut", bool),
        "flush_needs_speech": ("flush_needs_speech", bool),
        "repeat_cap": ("repeat_cap", int),
        "latin_commit_guard": ("latin_commit_guard", bool),
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

        return Qwen3ASRModel.from_pretrained(
            self.settings["model_path"], dtype=torch.bfloat16, device_map="cuda",
            max_inference_batch_size=1)

    def start(self, languages: list[str] | None = None, *, target_lang: str = "",
              vad=None, **_) -> None:
        import numpy as np

        self._vad = vad
        self._np = np
        wanted = [langs.norm_code(t) for t in (self.settings.get("targets") or []) if langs.norm_code(t)]
        primary = langs.norm_code(target_lang or "")
        self._targets = wanted or [code for code in (primary,) if code]
        if not self._targets:
            raise ValueError("transcription zprobe-prefix needs `targets` or a target language")
        self._bases = {
            t: self.model._build_text_prompt(context="", force_language=langs.CODE_TO_NAME.get(t))
            for t in self._targets
        }
        self._tok = self.model.processor.tokenizer
        self._tok.padding_side = "left"
        self._dev = next(self.model.model.thinker.parameters()).device
        self._fed_samples = 0
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._wav = self._np.empty(0, dtype=self._np.float32)
        self._raw = {t: "" for t in self._targets}
        self._decoded_samples = 0
        self._last_emit_samples = 0

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
        if not any(self._raw.values()) and not self._in_speech():
            return []
        cap = float(self.settings.get("max_utterance_sec", 14.0))
        if self._wav.shape[0] >= int(cap * audio_mod.SAMPLING_RATE):
            out = self._emit(self._decode(final=True), "cap")
            self._reset_utterance()
            return out
        step = int(float(self.settings.get("step_sec", 0.5)) * audio_mod.SAMPLING_RATE)
        first = int(float(self.settings.get("first_sec", 1.0)) * audio_mod.SAMPLING_RATE)
        if self._wav.shape[0] < first or self._wav.shape[0] - self._decoded_samples < step:
            return []
        stall = float(self.settings.get("stall_flush_sec", 0) or 0)
        force = bool(stall) and (self._wav.shape[0] - self._last_emit_samples
                                 >= int(stall * audio_mod.SAMPLING_RATE))
        added = self._decode(final=False, force=force)
        driver = added[self._targets[0]]
        if self.settings.get("seg_reset") and "<SEG>" in driver \
                and (self._raw[self._targets[0]] or driver.split("<SEG>")[0].strip()):
            head = {t: text.split("<SEG>")[0] for t, text in added.items()}
            out = self._emit(head, "seg")
            keep = int(float(self.settings.get("seg_keep_sec", 1.0)) * audio_mod.SAMPLING_RATE)
            tail = self._wav[-keep:] if keep else self._wav[:0]
            self._reset_utterance()
            self._wav = tail
            return out
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
        if self._wav.shape[0] == 0 or (not any(self._raw.values()) and not self._has_speech()):
            return []
        allow_eos = bool(self.settings.get("flush_allow_eos")) and reason == "finish"
        return self._emit(self._decode(final=True, allow_eos=allow_eos), reason)

    def _latin_blocked(self, target: str, added: str) -> bool:
        if not self.settings.get("latin_commit_guard") or target != "ko":
            return False
        letters = [c for c in added.replace("<SEG>", "") if c.isalpha()]
        return len(letters) >= 4 and sum(c.isascii() for c in letters) / len(letters) > 0.7

    def _emit(self, added: dict[str, str], reason: str) -> list:
        out = []
        emitted = False
        for target in self._targets:
            text = added.get(target, "")
            if self._latin_blocked(target, text):
                continue
            self._raw[target] += text
            shown = text.replace("<SEG>", " ").strip()
            shown = re.sub(r"\s+", " ", shown)
            if not shown:
                continue
            emitted = True
            elapsed = round(timing.elapsed() or 0.0, 4)
            out.append(Translated(
                original=shown,
                translation=shown,
                language="",
                target_lang=target,
                commit_reason=reason,
                decision_audio_sec=round(self._fed_samples / audio_mod.SAMPLING_RATE, 3),
                committed_elapsed_sec=elapsed,
                translated_elapsed_sec=elapsed,
            ))
        if emitted:
            self._last_emit_samples = self._wav.shape[0]
        return out

    @timing.measure("decode")
    def _decode(self, *, final: bool, force: bool = False, allow_eos: bool = False) -> dict[str, str]:
        import torch

        self._decoded_samples = self._wav.shape[0]
        flush = final or force
        if flush and self.settings.get("flush_needs_speech"):
            tail = self._wav[self._last_emit_samples:]
            if tail.shape[0] == 0 or float(abs(tail).max()) < 0.012:
                flush = False
        no_repeat = int(self.settings.get("no_repeat", 4))
        texts = [self._bases[t] + self._raw[t] for t in self._targets]
        inputs = self.model.processor(
            text=texts, audio=[self._wav] * len(texts),
            return_tensors="pt", padding=True).to(self._dev).to(torch.bfloat16)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            o = self.model.model.generate(
                **inputs, max_new_tokens=(min(int(self.settings.get("max_new_tokens", 120)),
                                              int(self.settings.get("flush_max_new", 0) or 10**9))
                                          if flush else int(self.settings.get("max_new_tokens", 120))),
                no_repeat_ngram_size=None if flush else (no_repeat or None),
                min_new_tokens=(None if allow_eos
                                else (int(self.settings.get("flush_min_new", 2)) or None)) if flush else None,
                do_sample=False, pad_token_id=self._tok.eos_token_id)
        seq = o.sequences if hasattr(o, "sequences") else o
        outs = {}
        for i, target in enumerate(self._targets):
            ids = inputs["input_ids"][i]
            new = seq[i, prompt_len:]
            while len(new) and int(new[-1]) in EOS_TOKENS:
                new = new[:-1]
            txt = _continuation(self._tok, ids, torch.cat([ids, new]))
            while "�" in txt and len(new):
                new = new[:-1]
                txt = _continuation(self._tok, ids, torch.cat([ids, new]))
            txt = SPECIAL.sub("", txt)
            cap = int(self.settings.get("repeat_cap", 0) or 0)
            if cap:
                txt = re.sub(REPEAT_TEMPLATE % cap, lambda m: m.group(1) * cap, txt, flags=re.S)
            if flush and self.settings.get("flush_sentence_cut"):
                m = SENTENCE_HEAD.search(txt)
                if m:
                    txt = m.group(0)
            outs[target] = txt if self.settings.get("seg_reset") else txt.replace("<SEG>", "")
        return outs
