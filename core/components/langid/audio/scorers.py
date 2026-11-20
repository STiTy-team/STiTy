"""Acoustic language scorers. Each returns log-scores for the session's candidate languages
(project two-letter codes); a candidate the model does not know gets -inf."""

import math
import threading

import numpy as np

from core.utils import audio as audio_mod
from core.utils import logging

log = logging.getLogger(__name__)

SR = audio_mod.SAMPLING_RATE
# facebook/mms-lid-* labels are ISO 639-3.
MMS_CODES = {
    "ko": "kor",
    "en": "eng",
    "ja": "jpn",
    "es": "spa",
    "zh": "cmn",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "yue": "yue",
    "ar": "arb",
    "hi": "hin",
    "vi": "vie",
    "th": "tha",
    "id": "ind",
}


class Scorer:
    name = "?"

    def scores(self, audio: np.ndarray, candidates: tuple[str, ...]) -> dict[str, float]:
        raise NotImplementedError

    def scores_batch(
        self, audios: list[np.ndarray], candidates: tuple[str, ...]
    ) -> list[dict[str, float]]:
        """Several clips at once; scorers that can batch on the GPU override this."""
        return [self.scores(a, candidates) for a in audios]


def _device(device: str | None) -> str:
    import torch

    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class WhisperTokenScorer(Scorer):
    """Log-softmax of whisper's language-token logits right after <|startoftranscript|>."""

    name = "whisper-token"

    def __init__(self, model: str, device: str | None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = _device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model, dtype=dtype)
        self.model = self.model.to(self.device).eval()
        tok = self.processor.tokenizer
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.ids = {}
        for t in tok.additional_special_tokens:
            code = t.strip("<|>")
            if t.startswith("<|") and t.endswith("|>") and 2 <= len(code) <= 3 and code.isalpha():
                self.ids[code] = tok.convert_tokens_to_ids(t)
        self.lock = threading.Lock()

    def scores(self, audio, candidates):
        torch = self.torch
        known = [c for c in candidates if c in self.ids]
        out = {c: -math.inf for c in candidates}
        if not known or len(audio) == 0:
            return out
        feats = self.processor.feature_extractor(audio, sampling_rate=SR, return_tensors="pt")
        feats = feats.input_features.to(self.device, self.model.dtype)
        with self.lock, torch.inference_mode():
            logits = (
                self.model(
                    input_features=feats,
                    decoder_input_ids=torch.tensor([[self.sot]], device=self.device),
                )
                .logits[0, -1]
                .float()
            )
        sel = torch.tensor([self.ids[c] for c in known], device=self.device)
        logp = torch.log_softmax(logits[sel], -1).cpu().tolist()
        out.update(dict(zip(known, logp)))
        return out

    def scores_batch(self, audios, candidates):
        """Whisper pads every clip to 30 s, so a batch costs about one encoder pass."""
        torch = self.torch
        known = [c for c in candidates if c in self.ids]
        out = [{c: -math.inf for c in candidates} for _ in audios]
        live = [i for i, a in enumerate(audios) if len(a)]
        if not known or not live:
            return out
        feats = self.processor.feature_extractor(
            [audios[i] for i in live], sampling_rate=SR, return_tensors="pt"
        ).input_features.to(self.device, self.model.dtype)
        start = torch.full((len(live), 1), self.sot, device=self.device)
        with self.lock, torch.inference_mode():
            logits = self.model(input_features=feats, decoder_input_ids=start).logits[:, -1].float()
        sel = torch.tensor([self.ids[c] for c in known], device=self.device)
        logp = torch.log_softmax(logits[:, sel], -1).cpu().tolist()
        for i, row in zip(live, logp):
            out[i].update(dict(zip(known, row)))
        return out


class WhisperDecodeScorer(Scorer):
    """Mean log-prob of a greedy decode forced to each candidate's language token
    (core/components/langid/whisper.py WhisperScorer._score)."""

    name = "whisper-decode"

    def __init__(self, model: str, device: str | None):
        from core.components.langid.whisper import WhisperScorer

        self.inner = WhisperScorer(model, device)

    def scores(self, audio, candidates):
        known = tuple(c for c in candidates if c in self.inner.languages)
        out = {c: -math.inf for c in candidates}
        if not known or len(audio) == 0:
            return out
        with self.inner.lock:
            out.update(self.inner._score(audio, known))
        return out


class EcapaScorer(Scorer):
    """speechbrain/lang-id-voxlingua107-ecapa log-posteriors (107 languages)."""

    name = "ecapa"

    def __init__(self, model: str, device: str | None):
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier

        self.torch = torch
        dev = _device(device)
        self.model = EncoderClassifier.from_hparams(
            source=model,
            savedir=f"/tmp/stity-ecapa/{model.replace('/', '--')}",
            run_opts={"device": "cuda:0" if dev == "cuda" else dev},
        )
        labels = self.model.hparams.label_encoder.ind2lab
        self.index = {str(lab).split(":")[0].strip(): i for i, lab in labels.items()}
        self.lock = threading.Lock()

    def scores(self, audio, candidates):
        torch = self.torch
        out = {c: -math.inf for c in candidates}
        known = [c for c in candidates if c in self.index]
        if not known or len(audio) == 0:
            return out
        with self.lock, torch.inference_mode():
            logp = self.model.classify_batch(torch.from_numpy(audio).float().unsqueeze(0))[0][0]
        logp = logp.float().cpu()
        sel = torch.tensor([self.index[c] for c in known])
        out.update(dict(zip(known, torch.log_softmax(logp[sel], -1).tolist())))
        return out

    def scores_batch(self, audios, candidates):
        torch = self.torch
        known = [c for c in candidates if c in self.index]
        out = [{c: -math.inf for c in candidates} for _ in audios]
        live = [i for i, a in enumerate(audios) if len(a)]
        if not known or not live:
            return out
        longest = max(len(audios[i]) for i in live)
        wavs = torch.zeros(len(live), longest)
        for row, i in enumerate(live):
            wavs[row, : len(audios[i])] = torch.from_numpy(audios[i]).float()
        lens = torch.tensor([len(audios[i]) / longest for i in live])
        with self.lock, torch.inference_mode():
            logp = self.model.classify_batch(wavs, wav_lens=lens)[0].float().cpu()
        sel = torch.tensor([self.index[c] for c in known])
        for row, i in enumerate(live):
            out[i].update(dict(zip(known, torch.log_softmax(logp[row][sel], -1).tolist())))
        return out


class MmsScorer(Scorer):
    """facebook/mms-lid-* (wav2vec2 sequence classifier) log-softmax over the candidates."""

    name = "mms"

    def __init__(self, model: str, device: str | None):
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        self.torch = torch
        self.device = _device(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model).to(self.device).eval()
        self.index = {lab: int(i) for i, lab in self.model.config.id2label.items()}
        self.lock = threading.Lock()

    def scores(self, audio, candidates):
        torch = self.torch
        out = {c: -math.inf for c in candidates}
        known = [c for c in candidates if MMS_CODES.get(c) in self.index]
        if not known or len(audio) < SR // 10:
            return out
        inputs = self.extractor(audio, sampling_rate=SR, return_tensors="pt").to(self.device)
        with self.lock, torch.inference_mode():
            logits = self.model(**inputs).logits[0].float()
        sel = torch.tensor([self.index[MMS_CODES[c]] for c in known], device=self.device)
        out.update(dict(zip(known, torch.log_softmax(logits[sel], -1).cpu().tolist())))
        return out


PHONE_PROFILES = "/home/mobility/STiTy-team/STiTy/models/lidprofiles/phone_profiles.json"
BIGRAM_WEIGHT = 0.7


class PhoneScorer(Scorer):
    """Phonotactic LID: a wav2vec2 phone recognizer turns the clip into a phone sequence, scored
    against per-language phone profiles (core/research/lid_module/build_phone_profiles.py) as a
    unigram or bigram (interpolated with the unigram) log-likelihood. Strong on short clips,
    where whisper is weakest. `model` is the profiles file."""

    def __init__(self, model: str, device: str | None, order: int):
        import json

        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.torch = torch
        self.device = _device(device)
        blob = json.load(open(model))
        self.profiles = blob["profiles"]
        self.order = order
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(blob["model"])
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = (
            Wav2Vec2ForCTC.from_pretrained(blob["model"], dtype=dtype).to(self.device).eval()
        )
        vocab = json.load(open(hf_hub_download(blob["model"], "vocab.json")))
        self.id2tok = {v: k for k, v in vocab.items()}
        self.special = {v for k, v in vocab.items() if k.startswith("<")}
        self.lock = threading.Lock()

    def phones(self, audio: np.ndarray) -> list[str]:
        torch = self.torch
        inp = self.extractor(audio, sampling_rate=SR, return_tensors="pt").input_values
        with self.lock, torch.inference_mode():
            ids = self.model(inp.to(self.device, self.model.dtype)).logits.argmax(-1)[0].tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i not in self.special:
                out.append(self.id2tok[i])
            prev = i
        return out

    def scores(self, audio, candidates):
        out = {c: -math.inf for c in candidates}
        known = [c for c in candidates if c in self.profiles]
        if not known or len(audio) < SR // 10:
            return out
        seq = self.phones(audio)
        if len(seq) < 2:
            return {c: (0.0 if c in known else -math.inf) for c in candidates}
        for c in known:
            uni, bi = self.profiles[c]["uni"], self.profiles[c]["bi"]
            total, prev = 0.0, "<s>"
            for ph in seq:
                p_uni = uni.get(ph, 1e-4)
                if self.order == 2:
                    p = BIGRAM_WEIGHT * bi.get(f"{prev} {ph}", 0.0) + (1 - BIGRAM_WEIGHT) * p_uni
                else:
                    p = p_uni
                total += math.log(max(p, 1e-6))
                prev = ph
            out[c] = total
        return out


LID_HEAD = "/home/mobility/STiTy-team/STiTy/models/lidhead/whisper_small_7lang.pt"


class WhisperHeadScorer(Scorer):
    """A language head (audio/head.py, trained on short crops of many languages by
    core/research/lid_module/short_train.py) over the whisper encoder frames of the clip.
    `model` is the head checkpoint; the encoder named in it is loaded alongside."""

    name = "whisper-head"

    def __init__(self, model: str, device: str | None):
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel

        from .head import LidHead

        self.torch = torch
        self.device = _device(device)
        self.head, self.langs, meta = LidHead.load(model)
        self.head = self.head.to(self.device).eval()
        name = meta.get("encoder", "openai/whisper-small")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.extractor = WhisperFeatureExtractor.from_pretrained(name)
        self.encoder = (
            WhisperModel.from_pretrained(name, dtype=dtype).encoder.to(self.device).eval()
        )
        self.lock = threading.Lock()

    def scores(self, audio, candidates):
        return self.scores_batch([audio], candidates)[0]

    def scores_batch(self, audios, candidates):
        torch = self.torch
        out = [{c: -math.inf for c in candidates} for _ in audios]
        known = [c for c in candidates if c in self.langs]
        live = [i for i, a in enumerate(audios) if len(a) >= 320]
        if not known or not live:
            return out
        clips = [audios[i] for i in live]
        feats = self.extractor(clips, sampling_rate=SR, return_tensors="pt").input_features
        n = [max(1, math.ceil(len(c) / 320)) for c in clips]
        with self.lock, torch.inference_mode():
            frames = self.encoder(feats.to(self.device, self.encoder.dtype)).last_hidden_state
            T = max(n)
            x = frames[:, :T].float()
            mask = (
                torch.arange(T, device=self.device)[None, :]
                < torch.tensor(n, device=self.device)[:, None]
            )
            logits = self.head(x, mask)
        sel = torch.tensor([self.langs.index(c) for c in known], device=self.device)
        logp = torch.log_softmax(logits[:, sel], -1).cpu().tolist()
        for i, row in zip(live, logp):
            out[i].update(dict(zip(known, row)))
        return out


SCORERS = {
    "whisper-token": (WhisperTokenScorer, "openai/whisper-small"),
    "whisper-decode": (WhisperDecodeScorer, "openai/whisper-small"),
    "ecapa": (EcapaScorer, "speechbrain/lang-id-voxlingua107-ecapa"),
    "mms": (MmsScorer, "facebook/mms-lid-256"),
    "whisper-head": (WhisperHeadScorer, LID_HEAD),
    "phone-uni": (lambda model, device: PhoneScorer(model, device, 1), PHONE_PROFILES),
    "phone-bi": (lambda model, device: PhoneScorer(model, device, 2), PHONE_PROFILES),
}


def build_scorer(name: str, model: str | None = None, device: str | None = None) -> Scorer:
    cls, default = SCORERS[name]
    log.info("[LOAD] audio langid scorer %s (%s)", name, model or default)
    return cls(model or default, device)
