import asyncio
import threading

import numpy as np

from core.components.transcription.qwen3 import resolve_model
from core.errors import ConfigError
from core.utils import audio as audio_mod
from core.utils import cache
from core.utils import langs
from core.utils import logging

from . import langids
from .base import LanguageDetector, Transcribed

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Doo12/Qwen3-ASR-1.7B-koen-seg-mix-merged"
SR = audio_mod.SAMPLING_RATE
MIN_SPEECH_SEC = 0.5
KEEP_SEC = 30.0


class ProbeScorer:
    """ko/en verdict from a Qwen3-ASR audio encoder (ln_post output, mean-pooled) through a linear
    head trained offline by core/research/zprobe/lid/lid_probe.py. Only the audio tower is kept on
    the GPU; the text decoder is dropped after loading."""

    def __init__(self, model_name: str, head_path: str):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.torch = torch
        head = np.load(head_path)
        self.w, self.b = head["w"], float(head["b"][0])
        self.mu, self.sd = head["mu"], head["sd"]
        self.labels = [str(x) for x in head["labels"]]
        trained_on = str(head["model"][0]) if "model" in head else "?"
        log.info("[LOAD] probe langid head %s (trained on %s)", head_path, trained_on)

        asr = Qwen3ASRModel.from_pretrained(
            resolve_model(model_name),
            dtype=torch.bfloat16,
            device_map="cuda",
            max_inference_batch_size=1,
        )
        self.processor = asr.processor
        self.prompt = asr._build_text_prompt(context="", force_language="Korean")
        self.thinker = asr.model.thinker.eval()
        self.thinker.model = None
        self.thinker.lm_head = None
        del asr
        torch.cuda.empty_cache()
        self.device = next(self.thinker.audio_tower.parameters()).device
        self.tap = {}
        self.thinker.audio_tower.ln_post.register_forward_hook(
            lambda mod, i, o: self.tap.update(v=o.detach())
        )
        self.lock = threading.Lock()

    def judge(self, audio: np.ndarray) -> tuple[str, float] | None:
        """(language, confidence in [0.5, 1]) or None when the span is too short to judge."""
        if len(audio) < SR * MIN_SPEECH_SEC:
            return None
        torch = self.torch
        with self.lock, torch.inference_mode():
            inputs = self.processor(
                text=[self.prompt], audio=[audio], return_tensors="pt", padding=True
            )
            feats = inputs["input_features"].to(self.device, torch.bfloat16)
            mask = inputs["feature_attention_mask"].to(self.device)
            self.thinker.get_audio_features(feats, feature_attention_mask=mask)
            e = self.tap["v"]
            e = e.reshape(-1, e.shape[-1]).float().mean(0).cpu().numpy()
        z = float(np.dot((e - self.mu) / self.sd, self.w) + self.b)
        p = 1 / (1 + np.exp(-z))
        return (self.labels[1] if p > 0.5 else self.labels[0]), max(p, 1 - p)


@langids.register("probe")
class ProbeLanguageDetector(LanguageDetector):
    """Judges each committed span by its audio; below `conf` it falls back to the alphabet rule
    (`fallback: alphabet`, the vela default) or leaves the ASR tag alone (`fallback: none`)."""

    SETTINGS = {
        "model": ("model", str),
        "head": ("head", str),
        "conf": ("conf", float),
        "max_sec": ("max_sec", float),
        "lag_sec": ("lag_sec", float),
        "fallback": ("fallback", str),
        "langs": ("langs", tuple),
        "threshold": ("threshold", float),
    }

    scorer = None

    @classmethod
    def validate(cls, options: dict, *, kind: str) -> dict:
        out = super().validate(options, kind=kind)
        if "head" not in out:
            raise ConfigError(f"pipeline.{kind}: 'probe' needs head: <path to the lid_probe npz>")
        if out.get("fallback", "alphabet") not in ("alphabet", "none"):
            raise ConfigError(f"pipeline.{kind}: 'probe' fallback is alphabet or none")
        return out

    async def load(self) -> None:
        model = self.settings.get("model", DEFAULT_MODEL)
        head = self.settings["head"]
        log.info("[LOAD] probe langid encoder %s", model)
        self.scorer = await cache.load(
            ("probe-langid", model, head),
            lambda: asyncio.to_thread(ProbeScorer, model, head),
        )

    def start(self, **_) -> None:
        self.audio = np.empty(0, dtype=np.float32)
        self.offset = 0.0
        self.since = 0.0

    def hear(self, audio: bytes) -> None:
        self.audio = np.concatenate([self.audio, audio_mod.from_pcm_bytes(audio)])
        self._forget(len(self.audio) - int(SR * KEEP_SEC))

    async def detect(self, item: Transcribed) -> str | None:
        # A commit is confirmed after the chunk that follows its speech, so the audio up to the
        # decision ends with the start of the next utterance; lag_sec moves the span back.
        lag = self.settings.get("lag_sec", 0.0)
        span_from, span_to = max(0.0, self.since - lag), max(0.0, item.decision_audio_sec - lag)
        start = max(0, int((span_from - self.offset) * SR))
        end = max(0, int((span_to - self.offset) * SR))
        span = self.audio[start:end][: int(SR * self.settings.get("max_sec", 10.0))]
        self.since = item.decision_audio_sec
        self._forget(end)
        verdict = await asyncio.to_thread(self.scorer.judge, span)
        sure = verdict is not None and verdict[1] >= self.settings.get("conf", 0.99)
        detected = verdict[0] if sure else self._fallback(item)
        log.info(
            "[LANGID] probe=%s conf=%s span=%.1f-%.1fs -> %s: %r",
            verdict[0] if verdict else "-",
            f"{verdict[1]:.3f}" if verdict else "-",
            span_from,
            span_to,
            detected,
            item.original[:40],
        )
        return detected

    def _fallback(self, item: Transcribed) -> str | None:
        if self.settings.get("fallback", "alphabet") != "alphabet":
            return None
        candidates = self.settings.get("langs") or langs.ALPHABET_LANGS
        threshold = self.settings.get("threshold", langs.ALPHABET_THRESHOLD)
        return langs.detect_by_alphabet(item.original, langs=candidates, threshold=threshold)

    def _forget(self, samples: int) -> None:
        samples = min(samples, len(self.audio))
        if samples <= 0:
            return
        self.audio = self.audio[samples:]
        self.offset += samples / SR
