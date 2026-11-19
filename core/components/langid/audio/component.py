import asyncio

import numpy as np

from core.errors import ConfigError
from core.utils import audio as audio_mod
from core.utils import cache
from core.utils import logging

from .. import langids
from ..base import LanguageDetector, Transcribed
from .decide import decide, lag_span, normalise, smooth, with_prior
from .scorers import SCORERS, build_scorer
from .span import AlignerSpan
from .vad import speech_only

log = logging.getLogger(__name__)

SR = audio_mod.SAMPLING_RATE
KEEP_SEC = 30.0
# Forced alignment assumes the audio holds only the text: start 1 s before the previous
# decision (northstar: 91% of the span inside the commit's sentence, vs 40% from 6 s back).
ALIGN_BACK_SEC = 1.0
MIN_SPEECH_SEC = 0.3


@langids.register("audio")
class AudioLanguageDetector(LanguageDetector):
    """Each commit's language from audio only. The commit text is used only to find where it was
    spoken (forced alignment); `mode: segment` scores that span once, `mode: timeline` reads it
    off a continuously scored, HMM-smoothed language timeline."""

    SETTINGS = {
        "mode": ("mode", str),
        "scorer": ("scorer", str),
        "model": ("model", str),
        "conf": ("conf", float),
        "stay_prior": ("stay_prior", float),
        "p_stay": ("p_stay", float),
        "windows": ("windows", tuple),
        "step_sec": ("step_sec", float),
        "lag_sec": ("lag_sec", float),
        "aligner": ("aligner", bool),
        "max_sec": ("max_sec", float),
    }

    scorer = None
    spanner = None

    @classmethod
    def validate(cls, options: dict, *, kind: str) -> dict:
        out = super().validate(options, kind=kind)
        if out.get("mode", "segment") not in ("segment", "timeline"):
            raise ConfigError(f"pipeline.{kind}: 'audio' mode is segment or timeline")
        if out.get("scorer", "whisper-token") not in SCORERS:
            raise ConfigError(f"pipeline.{kind}: 'audio' scorer is one of {sorted(SCORERS)}")
        return out

    async def load(self) -> None:
        name, model = self.settings.get("scorer", "whisper-token"), self.settings.get("model")
        self.scorer = await cache.load(
            ("audio-langid", name, model), lambda: asyncio.to_thread(build_scorer, name, model)
        )
        if self.settings.get("aligner", True):
            self.spanner = await cache.load(
                ("forced-aligner",), lambda: asyncio.to_thread(AlignerSpan)
            )

    def start(self, *, languages: list[str] | None = None, **_) -> None:
        self.candidates = tuple(dict.fromkeys(languages or ()))
        self.audio = np.empty(0, dtype=np.float32)
        self.offset = 0.0
        self.since = 0.0
        self.previous = None
        self.steps: list[tuple[float, dict[str, float] | None]] = []
        self.next_step = self.settings.get("step_sec", 0.5)
        self.trace: list[dict] = []

    @property
    def mode(self) -> str:
        return self.settings.get("mode", "segment")

    def _end_sec(self) -> float:
        return self.offset + len(self.audio) / SR

    def _slice(self, start: float, end: float) -> np.ndarray:
        a = max(0, int((start - self.offset) * SR))
        b = max(0, int((end - self.offset) * SR))
        return self.audio[a:b]

    def hear(self, audio: bytes) -> None:
        self.audio = np.concatenate([self.audio, audio_mod.from_pcm_bytes(audio)])
        if self.mode == "timeline" and len(self.candidates) > 1:
            step = self.settings.get("step_sec", 0.5)
            while self._end_sec() >= self.next_step:
                self.steps.append((self.next_step, self._score_step(self.next_step)))
                self.next_step += step
        keep_from = self._end_sec() - KEEP_SEC
        if keep_from > self.offset:
            self.audio = self.audio[int((keep_from - self.offset) * SR) :]
            self.offset = keep_from
            self.steps = [s for s in self.steps if s[0] >= keep_from]

    def _score_step(self, t: float) -> dict[str, float] | None:
        """Mean log-scores of the trailing windows ending at `t` that hold enough speech."""
        total, n = {c: 0.0 for c in self.candidates}, 0
        for w in self.settings.get("windows", (1.0, 2.0, 4.0)):
            speech = speech_only(self._slice(t - float(w), t))
            if len(speech) < SR * MIN_SPEECH_SEC:
                continue
            scores = self.scorer.scores(speech, self.candidates)
            for c in self.candidates:
                total[c] += scores[c]
            n += 1
        return {c: v / n for c, v in total.items()} if n else None

    def _span(self, item: Transcribed) -> tuple[tuple[float, float], str]:
        decision = item.decision_audio_sec
        if self.spanner is not None:
            start = max(self.offset, self.since - ALIGN_BACK_SEC)
            found = self.spanner.find(
                self._slice(start, decision), start, item.original, self.candidates
            )
            if found is not None:
                return found, "aligner"
        return lag_span(self.since, decision, self.settings.get("lag_sec", 0.5)), "lag"

    def _segment_probs(self, span: tuple[float, float]) -> tuple[dict | None, dict | None]:
        speech = speech_only(self._slice(*span))[: int(SR * self.settings.get("max_sec", 10.0))]
        if len(speech) < SR * MIN_SPEECH_SEC:
            return None, None
        raw = self.scorer.scores(speech, self.candidates)
        prior = with_prior(raw, self.previous, self.settings.get("stay_prior", 0.7))
        return normalise(prior, self.candidates), raw

    def _timeline_probs(self, span: tuple[float, float], decision: float) -> dict | None:
        known = [(t, s) for t, s in self.steps if t <= decision + 1e-6]
        if not known:
            return None
        posts = smooth([s for _, s in known], self.candidates, self.settings.get("p_stay", 0.9))
        # A step's trailing windows end at t; its centre is half the shortest window earlier.
        half = min(float(w) for w in self.settings.get("windows", (1.0, 2.0, 4.0))) / 2
        centres = [t - half for t, _ in known]
        inside = [i for i, c in enumerate(centres) if span[0] <= c <= span[1] and known[i][1]]
        if not inside:
            inside = [min(range(len(centres)), key=lambda i: abs(centres[i] - span[1]))]
        mass = {c: sum(posts[i][c] for i in inside) for c in self.candidates}
        total = sum(mass.values()) or 1.0
        return {c: v / total for c, v in mass.items()}

    async def detect(self, item: Transcribed) -> str | None:
        if len(self.candidates) < 2:
            return None
        span, source = await asyncio.to_thread(self._span, item)
        raw = None
        if self.mode == "segment":
            probs, raw = await asyncio.to_thread(self._segment_probs, span)
        else:
            probs = self._timeline_probs(span, item.decision_audio_sec)
        detected = decide(probs, self.previous, self.settings.get("conf", 0.7))
        self.trace.append(
            {
                "decision": item.decision_audio_sec,
                "span": span,
                "src": source,
                "raw": raw,
                "probs": probs,
                "previous": self.previous,
                "detected": detected,
            }
        )
        self.since = item.decision_audio_sec
        if detected is not None:
            self.previous = detected
        log.info(
            "[LANGID] audio=%s probs=%s span=%.1f-%.1fs src=%s -> %s: %r",
            self.mode,
            {k: round(v, 3) for k, v in probs.items()} if probs else "-",
            span[0],
            span[1],
            source,
            detected,
            item.original[:40],
        )
        return detected
