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


SCORERS = {
    "whisper-token": (WhisperTokenScorer, "openai/whisper-small"),
    "whisper-decode": (WhisperDecodeScorer, "openai/whisper-small"),
    "ecapa": (EcapaScorer, "speechbrain/lang-id-voxlingua107-ecapa"),
    "mms": (MmsScorer, "facebook/mms-lid-256"),
}


def build_scorer(name: str, model: str | None = None, device: str | None = None) -> Scorer:
    cls, default = SCORERS[name]
    log.info("[LOAD] audio langid scorer %s (%s)", name, model or default)
    return cls(model or default, device)
