"""Acoustic language identification for slot routing, fused from two complementary judges.

whisper-small (decoder language-token argmax restricted to the candidate set) is the stronger
judge on windows that carry a sentence of speech but degrades on very short ones; a phone
recognizer scored against per-language phone-frequency profiles (fitted offline by
core/research/zprobe/lid/build_phone_profiles.py) is weaker overall but strongest exactly
there. Measured on 283 northstar commit windows: whisper 0.910 overall / 0.762 short,
phonotactics 0.876 / 0.912, fused 0.922 overall, 0.901 near language switches. Text-based
identification was measured and rejected for this job: when the transcriber itself writes the
wrong language (English transliterated into Hangul), a script-ratio stage inherits the error
with full confidence - 19 of its 33 errors.

The fused verdict prefers the phonotactic judge below `short_sec` of audio and whisper above
it, falling back to the other when one abstains. `verdict` also reports whether the two judges
agree: agreement ran at 0.975 accuracy, so callers acting irreversibly (slot resets) can
require it.
"""
import json


class FusedLid:

    def __init__(self, whisper, whisper_proc, phone, phone_fe, phone_vocab, profiles,
                 langs=("ko", "en"), short_sec=1.5):
        self._wm = whisper
        self._wproc = whisper_proc
        self._pm = phone
        self._pfe = phone_fe
        self._id2tok = {v: k for k, v in phone_vocab.items()}
        self._special = {v for k, v in phone_vocab.items() if k.startswith("<")}
        self._profiles = profiles
        self._langs = tuple(langs)
        self._short_sec = short_sec
        tok = whisper_proc.tokenizer
        self._allowed = {c: tok.convert_tokens_to_ids(f"<|{c}|>") for c in self._langs}
        self._sot = tok.convert_tokens_to_ids("<|startoftranscript|>")

    @classmethod
    async def build(cls, profiles_path, langs=("ko", "en"), short_sec=1.5):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC,
                                  WhisperForConditionalGeneration, WhisperProcessor)

        blob = json.load(open(profiles_path))
        wproc = WhisperProcessor.from_pretrained("openai/whisper-small")
        wm = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small", dtype=torch.float16).to("cuda").eval()
        pfe = Wav2Vec2FeatureExtractor.from_pretrained(blob["model"])
        pm = Wav2Vec2ForCTC.from_pretrained(blob["model"], dtype=torch.float16).to("cuda").eval()
        vocab = json.load(open(hf_hub_download(blob["model"], "vocab.json")))
        return cls(wm, wproc, pm, pfe, vocab, blob["profiles"], langs=langs, short_sec=short_sec)

    def _whisper(self, wav):
        import torch

        feats = self._wproc(wav, sampling_rate=16000,
                            return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            logits = self._wm(feats, decoder_input_ids=torch.tensor(
                [[self._sot]], device="cuda")).logits[0, -1].float()
        return max(self._allowed, key=lambda c: logits[self._allowed[c]].item())

    def _phones(self, wav):
        import torch

        inp = self._pfe(wav, sampling_rate=16000,
                        return_tensors="pt").input_values.to("cuda", torch.float16)
        with torch.no_grad():
            ids = self._pm(inp).logits.argmax(-1)[0].tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i not in self._special:
                out.append(self._id2tok[i])
            prev = i
        return out

    def _phonotactic(self, wav):
        import math

        phones = self._phones(wav)
        if len(phones) < 4:
            return None
        scores = {lang: sum(math.log(prof.get(ph, 1e-4)) for ph in phones)
                  for lang, prof in self._profiles.items() if lang in self._langs}
        return max(scores, key=scores.get) if scores else None

    def verdict(self, wav) -> tuple[str | None, bool]:
        if wav.shape[0] < int(0.6 * 16000):
            return None, False
        w = self._whisper(wav)
        p = self._phonotactic(wav)
        short = wav.shape[0] < int(self._short_sec * 16000)
        v = (p or w) if short else (w or p)
        return v, bool(w and p and w == p)
