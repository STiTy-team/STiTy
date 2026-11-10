import re

from core.utils import audio as audio_mod
from core.utils import cache
from core.utils import langs
from core.utils import lid
from core.utils import logging
from core.utils import timing

from . import transcribers
from ..registry import Speech, Translated
from .base import Transcriber
from .qwen3 import resolve_model

log = logging.getLogger(__name__)

EOS_TOKENS = (151645, 151643)
SPECIAL = re.compile(r"<\|[^<>|]+\|>")
WAIT = "<WAIT>"
REPEAT_TEMPLATE = r"((.{1,12}?))(?:\2){%d,}"


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
        "seg_reset": ("seg_reset", bool),
        "seg_keep_sec": ("seg_keep_sec", float),
        "repeat_cap": ("repeat_cap", int),
        "latin_commit_guard": ("latin_commit_guard", bool),
        "wait_token": ("wait_token", bool),
        "lid_slot_reset": ("lid_slot_reset", bool),
        "lid_profiles": ("lid_profiles", str),
        "lid_window_sec": ("lid_window_sec", float),
        "lid_keep_sec": ("lid_keep_sec", float),
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
        self._lid = None
        if self.settings.get("lid_slot_reset"):
            profiles = self.settings.get("lid_profiles", "models/zprobe/data/lid_phone_profiles.json")
            self._lid = await cache.load(("fused-lid", profiles),
                                         lambda: lid.FusedLid.build(profiles))

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
        self._slot_lang = None

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
        step = int(float(self.settings.get("step_sec", 0.5)) * audio_mod.SAMPLING_RATE)
        first = int(float(self.settings.get("first_sec", 1.0)) * audio_mod.SAMPLING_RATE)
        if self._wav.shape[0] < first or self._wav.shape[0] - self._decoded_samples < step:
            return []
        if self._lid is not None and self._in_speech():
            window = int(float(self.settings.get("lid_window_sec", 2.5)) * audio_mod.SAMPLING_RATE)
            verdict, agree = self._lid.verdict(self._wav[-window:])
            if verdict and agree and self._slot_lang is None:
                self._slot_lang = verdict
            elif verdict and agree and verdict != self._slot_lang:
                prev = self._slot_lang
                self._slot_lang = verdict
                if any(self._raw.values()):
                    out = self._emit(self._decode(final=False), "lid")
                    keep = int(float(self.settings.get("lid_keep_sec", 1.5)) * audio_mod.SAMPLING_RATE)
                    tail = self._wav[-keep:] if keep else self._wav[:0]
                    self._reset_utterance()
                    self._wav = tail
                    self._slot_lang = verdict
                    log.info("[SLOT-RESET] cause=lid from=%s to=%s keep_sec=%.1f",
                             prev, verdict, keep / audio_mod.SAMPLING_RATE)
                    return out
        added = self._decode(final=False)
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
        return self._emit(self._decode(final=True), reason)

    def _latin_blocked(self, target: str, added: str) -> bool:
        if not self.settings.get("latin_commit_guard") or target != "ko":
            return False
        letters = [c for c in added.replace("<SEG>", "") if c.isalpha()]
        return len(letters) >= 4 and sum(c.isascii() for c in letters) / len(letters) > 0.7

    def _emit(self, added: dict[str, str], reason: str) -> list:
        out = []
        for target in self._targets:
            text = added.get(target, "")
            if self._latin_blocked(target, text):
                continue
            self._raw[target] += text
            shown = text.replace("<SEG>", " ").strip()
            shown = re.sub(r"\s+", " ", shown)
            if not shown:
                continue
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
        return out

    @timing.measure("decode")
    def _decode(self, *, final: bool) -> dict[str, str]:
        import torch

        self._decoded_samples = self._wav.shape[0]
        no_repeat = int(self.settings.get("no_repeat", 4))
        texts = [self._bases[t] + self._raw[t] for t in self._targets]
        inputs = self.model.processor(
            text=texts, audio=[self._wav] * len(texts),
            return_tensors="pt", padding=True).to(self._dev).to(torch.bfloat16)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            o = self.model.model.generate(
                **inputs, max_new_tokens=int(self.settings.get("max_new_tokens", 120)),
                no_repeat_ngram_size=None if final else (no_repeat or None),
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
            if self.settings.get("wait_token"):
                txt = txt.replace(WAIT, "")
                for k in range(len(WAIT) - 1, 0, -1):
                    if txt.endswith(WAIT[:k]):
                        txt = txt[:-k]
                        break
            cap = int(self.settings.get("repeat_cap", 0) or 0)
            if cap:
                txt = re.sub(REPEAT_TEMPLATE % cap, lambda m: m.group(1) * cap, txt, flags=re.S)
            outs[target] = txt if self.settings.get("seg_reset") else txt.replace("<SEG>", "")
        return outs
