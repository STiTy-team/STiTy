import asyncio
import threading

import numpy as np

from core.errors import ConfigError
from core.utils import audio as audio_mod
from core.utils import cache
from core.utils import logging

from . import langids
from .base import LanguageDetector, Transcribed

log = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/whisper-small"
SR = audio_mod.SAMPLING_RATE
MAX_TOKENS = 12
MIN_SPEECH_SEC = 0.2
KEEP_SEC = 30.0
PEAK = 0.5


class WhisperScorer:

    def __init__(self, model_name: str, device: str | None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        try:
            from silero_vad import load_silero_vad
        except ImportError as e:
            raise ConfigError(
                "langid 'whisper' needs the silero-vad package (pip install silero-vad)"
            ) from e

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, dtype=dtype).to(self.device).eval()
        tok = self.processor.tokenizer
        self.tokenizer = tok
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps = tok.convert_tokens_to_ids("<|notimestamps|>")
        self.eot = tok.convert_tokens_to_ids("<|endoftext|>")
        self.languages = {
            t.strip("<|>") for t in tok.additional_special_tokens
            if len(t) == 6 and t.startswith("<|") and t.endswith("|>")
        }
        self.vad = load_silero_vad()
        self.lock = threading.Lock()

    def judge(self, audio: np.ndarray, langs: tuple[str, ...],
              max_sec: float) -> dict[str, float] | None:
        with self.lock:
            speech = self._speech(audio)
            if len(speech) < SR * MIN_SPEECH_SEC:
                return None
            return self._score(speech[: int(SR * max_sec)], langs)

    def _speech(self, audio: np.ndarray) -> np.ndarray:
        from silero_vad import get_speech_timestamps

        spans = get_speech_timestamps(self.torch.from_numpy(audio), self.vad, sampling_rate=SR)
        if not spans:
            return audio[:0]
        return audio[spans[0]["start"]:spans[-1]["end"]]

    def _score(self, audio: np.ndarray, langs: tuple[str, ...]) -> dict[str, float]:
        torch = self.torch
        peak = float(np.abs(audio).max())
        if peak > 1e-4:
            audio = audio / peak * PEAK
        feats = self.processor.feature_extractor(
            audio, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.device, self.model.dtype)
        scores = {}
        with torch.inference_mode():
            encoded = self.model.model.encoder(feats)
            for lang in langs:
                lang_id = self.tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
                dec = torch.tensor([[self.sot, lang_id, self.transcribe, self.notimestamps]],
                                   device=self.device)
                total, n = 0.0, 0
                for _ in range(MAX_TOKENS):
                    logits = self.model(encoder_outputs=encoded,
                                        decoder_input_ids=dec).logits[0, -1].float()
                    logp = torch.log_softmax(logits, -1)
                    nxt = int(torch.argmax(logp))
                    if nxt == self.eot:
                        break
                    total += float(logp[nxt])
                    n += 1
                    dec = torch.cat([dec, torch.tensor([[nxt]], device=self.device)], dim=1)
                scores[lang] = total / n if n else float("-inf")
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return scores


@langids.register("whisper")
class WhisperLanguageDetector(LanguageDetector):

    SETTINGS = {
        "model": ("model", str),
        "device": ("device", str),
        "langs": ("langs", tuple),
        "margin": ("margin", float),
        "max_sec": ("max_sec", float),
    }

    scorer = None

    async def load(self) -> None:
        model = self.settings.get("model", DEFAULT_MODEL)
        device = self.settings.get("device")
        log.info("[LOAD] language id model %s", model)
        self.scorer = await cache.load(
            ("whisper-langid", model, device),
            lambda: asyncio.to_thread(WhisperScorer, model, device))

    def start(self, *, languages: list[str] | None = None, **_) -> None:
        wanted = self.settings.get("langs") or tuple(languages or ())
        unknown = [lang for lang in wanted if lang not in self.scorer.languages]
        if unknown:
            log.warning("[LANGID-UNKNOWN] whisper has no language token for %s", unknown)
        self.langs = tuple(lang for lang in wanted if lang in self.scorer.languages)
        self.audio = np.empty(0, dtype=np.float32)
        self.offset = 0.0
        self.since = 0.0

    def hear(self, audio: bytes) -> None:
        self.audio = np.concatenate([self.audio, audio_mod.from_pcm_bytes(audio)])
        self._forget(len(self.audio) - int(SR * KEEP_SEC))

    async def detect(self, item: Transcribed) -> str | None:
        start = max(0, int((self.since - self.offset) * SR))
        end = max(0, int((item.decision_audio_sec - self.offset) * SR))
        span = self.audio[start:end]
        self.since = item.decision_audio_sec
        self._forget(end)
        if len(self.langs) < 2:
            return None
        scores = await asyncio.to_thread(
            self.scorer.judge, span, self.langs, self.settings.get("max_sec", 6.0))
        if not scores:
            return None
        best, second = sorted(scores, key=scores.get, reverse=True)[:2]
        margin = scores[best] - scores[second]
        log.debug("[LANGID] %s over %s by %.2f", best, second, margin)
        return best if margin >= self.settings.get("margin", 0.25) else None

    def _forget(self, samples: int) -> None:
        samples = min(samples, len(self.audio))
        if samples <= 0:
            return
        self.audio = self.audio[samples:]
        self.offset += samples / SR
