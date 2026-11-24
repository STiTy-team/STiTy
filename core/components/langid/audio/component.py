import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.errors import ConfigError
from core.utils import audio as audio_mod
from core.utils import cache
from core.utils import logging

from .. import langids
from ..base import LanguageDetector, Transcribed
from .decide import combine, decide, lag_span, normalise, smooth, span_mass, with_prior
from .scorers import SCORERS, build_scorer
from .script import text_language
from .span import AlignerSpan
from .vad import speech_only, speech_spans

log = logging.getLogger(__name__)

SR = audio_mod.SAMPLING_RATE
KEEP_SEC = 30.0
# Forced alignment assumes the audio holds only the text: start 1 s before the previous
# decision (northstar: 91% of the span inside the commit's sentence, vs 40% from 6 s back).
# `align_from: prev_end` starts at the previous commit's aligned end instead (0.92 overall,
# 0.92 vs 0.87 on commits right after a language switch).
ALIGN_BACK_SEC = 1.0
MIN_SPEECH_SEC = 0.3
# A step whose own slot (one step around its time) holds less speech than this is silent.
SILENT_SLOT_SEC = 0.1


def _defaults(mode: str) -> dict:
    """Settings that helped in the 2026-11-20 replays, on unless a config says otherwise:
    aligner window from the previous aligned end and conf 0.5 for both modes; for the timeline,
    centered windows, reading the span's last speech step, p_stay 0.8 at temperature 10. Spans
    shorter than `short_text_sec` take the language of the commit text's script when it singles
    out one candidate (2026-11-21: northstar errors 6 -> 2); 0 turns that off. On longer spans,
    when the acoustic verdict and the text script disagree and the acoustic confidence is under
    `text_override_conf`, the script wins (northstar 2 -> 1); a sure acoustic verdict stands, which
    keeps English transliterated into Hangul English.
    `p_gap` and several scorers did not help and stay off."""
    out = {"align_from": "prev_end", "conf": 0.5, "short_text_sec": 0.8, "text_override_conf": 0.9}
    if mode == "timeline":
        out.update(centered=True, read="end", p_stay=0.8, temperature=10.0)
    return out


def _scorer_specs(settings: dict) -> list[tuple[str, str | None]]:
    """(scorer name, model) pairs: `scorers: [name or name@model, ...]`, else `scorer`/`model`."""
    listed = settings.get("scorers")
    if not listed:
        return [(settings.get("scorer", "whisper-token"), settings.get("model"))]
    specs = []
    for entry in listed:
        name, _, model = str(entry).partition("@")
        specs.append((name, model or None))
    return specs


@langids.register("audio")
class AudioLanguageDetector(LanguageDetector):
    """Each commit's language from audio only. The commit text is used only to find where it was
    spoken (forced alignment); `mode: segment` scores that span once, `mode: timeline` reads it
    off a continuously scored, HMM-smoothed language timeline.

    Timeline options: `centered` windows (around the step instead of ending at it - the commit lag
    pays for the audio after it), `p_gap` (language-stay probability into a step that follows a
    silent one), `read` (`mean`, `tail` = last `read_sec` of the span, `end` = last speech step).
    Several `scorers` run in parallel and are merged per window by `combine` (`mean` with optional
    `weights`, or `max` = the surest scorer). `align_from: prev_end` starts the aligner window at
    the previous commit's aligned end, `decision` 1 s before the previous decision. Defaults: see
    `_defaults` (what helped is on)."""

    SETTINGS = {
        "mode": ("mode", str),
        "scorer": ("scorer", str),
        "model": ("model", str),
        "scorers": ("scorers", tuple),
        "combine": ("combine", str),
        "weights": ("weights", dict),
        "conf": ("conf", float),
        "stay_prior": ("stay_prior", float),
        "p_stay": ("p_stay", float),
        "p_gap": ("p_gap", float),
        "windows": ("windows", tuple),
        "centered": ("centered", bool),
        "read": ("read", str),
        "read_sec": ("read_sec", float),
        "step_sec": ("step_sec", float),
        "lag_sec": ("lag_sec", float),
        "aligner": ("aligner", bool),
        "align_from": ("align_from", str),
        "max_sec": ("max_sec", float),
        "temperature": ("temperature", float),
        "keep_raw": ("keep_raw", bool),
        "short_text_sec": ("short_text_sec", float),
        "text_override_conf": ("text_override_conf", float),
    }

    scorers: dict = {}
    spanner = None

    def __init__(self, settings: dict, *, cfg):
        super().__init__({**_defaults(settings.get("mode", "segment")), **settings}, cfg=cfg)

    @classmethod
    def validate(cls, options: dict, *, kind: str) -> dict:
        out = super().validate(options, kind=kind)
        if out.get("mode", "segment") not in ("segment", "timeline"):
            raise ConfigError(f"pipeline.{kind}: 'audio' mode is segment or timeline")
        for name, _ in _scorer_specs(out):
            if name not in SCORERS:
                raise ConfigError(f"pipeline.{kind}: 'audio' scorer is one of {sorted(SCORERS)}")
        if out.get("combine", "mean") not in ("mean", "max"):
            raise ConfigError(f"pipeline.{kind}: 'audio' combine is mean or max")
        if out.get("read", "mean") not in ("mean", "tail", "end"):
            raise ConfigError(f"pipeline.{kind}: 'audio' read is mean, tail or end")
        if out.get("align_from", "decision") not in ("decision", "prev_end"):
            raise ConfigError(f"pipeline.{kind}: 'audio' align_from is decision or prev_end")
        return out

    @property
    def scorer(self):
        return next(iter(self.scorers.values()), None)

    @scorer.setter
    def scorer(self, value) -> None:
        self.scorers = {"main": value}

    async def load(self) -> None:
        if self.settings.get("mode", "segment") == "timeline":
            # Timeline scoring runs synchronously inside hear(), on the caller's event loop: fine
            # for offline bench, but a live server room would stall on every 0.5 s step.
            log.warning("[LANGID] audio timeline scores inside hear() - bench use only")
        loaded = {}
        for name, model in _scorer_specs(self.settings):
            loaded[f"{name}@{model}" if model else name] = await cache.load(
                ("audio-langid", name, model),
                lambda n=name, m=model: asyncio.to_thread(build_scorer, n, m),
            )
        self.scorers = loaded
        if self.settings.get("aligner", True):
            self.spanner = await cache.load(
                ("forced-aligner",), lambda: asyncio.to_thread(AlignerSpan)
            )

    def start(self, *, languages: list[str] | None = None, **_) -> None:
        self.candidates = tuple(dict.fromkeys(languages or ()))
        self.audio = np.empty(0, dtype=np.float32)
        self.offset = 0.0
        self.since = 0.0
        self.last_end: float | None = None
        self.previous = None
        self.steps: list[tuple[float, dict[str, float] | None]] = []
        self.silent: list[bool] = []
        self.raw: list[dict] = []
        self.next_step = self.settings.get("step_sec", 0.5)
        self.trace: list[dict] = []
        self.pool = ThreadPoolExecutor(max_workers=max(1, len(self.scorers)))

    @property
    def mode(self) -> str:
        return self.settings.get("mode", "segment")

    def _windows(self) -> tuple[float, ...]:
        return tuple(float(w) for w in self.settings.get("windows", (1.0, 2.0, 4.0)))

    def _end_sec(self) -> float:
        return self.offset + len(self.audio) / SR

    def _slice(self, start: float, end: float) -> np.ndarray:
        a = max(0, int((start - self.offset) * SR))
        b = max(0, int((end - self.offset) * SR))
        return self.audio[a:b]

    def _ready(self, t: float) -> bool:
        ahead = max(self._windows()) / 2 if self.settings.get("centered") else 0.0
        return self._end_sec() >= t + ahead - 1e-9

    def hear(self, audio: bytes) -> None:
        self.audio = np.concatenate([self.audio, audio_mod.from_pcm_bytes(audio)])
        if self.mode == "timeline" and len(self.candidates) > 1:
            step = self.settings.get("step_sec", 0.5)
            while self._ready(self.next_step):
                self._add_step(self.next_step, self._end_sec())
                self.next_step += step
        keep_from = self._end_sec() - KEEP_SEC
        if keep_from > self.offset:
            self.audio = self.audio[int((keep_from - self.offset) * SR) :]
            self.offset = keep_from
            kept = [i for i, s in enumerate(self.steps) if s[0] >= keep_from]
            self.steps = [self.steps[i] for i in kept]
            self.silent = [self.silent[i] for i in kept]
            if self.raw:
                self.raw = [self.raw[i] for i in kept]

    def _add_step(self, t: float, limit: float) -> None:
        scores, silent, raw = self._score_step(t, limit)
        self.steps.append((t, scores))
        self.silent.append(silent)
        if self.settings.get("keep_raw"):
            self.raw.append(raw)

    def _score_step(self, t: float, limit: float) -> tuple[dict | None, bool, dict]:
        """Log-scores at step `t` (windows merged across scorers, then averaged over windows),
        whether the step's own slot is silent, and the raw per-scorer, per-window scores. Audio
        after `limit` is not used (a centered window near the decision is cut there)."""
        centered = self.settings.get("centered", False)
        step = self.settings.get("step_sec", 0.5)
        windows = self._windows()
        bounds = {
            w: ((t - w / 2, min(t + w / 2, limit)) if centered else (t - w, t)) for w in windows
        }
        lo = max(min(b[0] for b in bounds.values()), self.offset)
        hi = max(b[1] for b in bounds.values())
        audio = self._slice(lo, hi)
        spans = speech_spans(audio)

        def speech(a: float, b: float) -> np.ndarray:
            x, y = int((a - lo) * SR), int((b - lo) * SR)
            parts = [audio[max(s, x) : min(e, y)] for s, e in spans if min(e, y) > max(s, x)]
            return np.concatenate(parts) if parts else audio[:0]

        slot = (t - step / 2, min(t + step / 2, limit)) if centered else (t - step, t)
        silent = len(speech(*slot)) < SR * SILENT_SLOT_SEC
        clips = {w: speech(*b) for w, b in bounds.items()}
        clips = {w: c for w, c in clips.items() if len(c) >= SR * MIN_SPEECH_SEC}
        if not clips:
            return None, silent, {}
        order = list(clips)
        futures = {
            name: self.pool.submit(s.scores_batch, [clips[w] for w in order], self.candidates)
            for name, s in self.scorers.items()
        }
        raw = {name: dict(zip(map(str, order), f.result())) for name, f in futures.items()}
        merged = []
        for w in map(str, order):
            per = combine(
                {name: raw[name][w] for name in raw},
                self.candidates,
                self.settings.get("combine", "mean"),
                self.settings.get("weights"),
            )
            if per:
                merged.append(per)
        if not merged:
            return None, silent, raw
        mean = {c: sum(m[c] for m in merged) / len(merged) for c in self.candidates}
        return self._scaled(mean), silent, raw

    def _scaled(self, scores: dict[str, float]) -> dict[str, float]:
        """Log-scores times `temperature` (a sharper or flatter scorer scale)."""
        temp = self.settings.get("temperature", 1.0)
        return scores if temp == 1.0 else {k: v * temp for k, v in scores.items()}

    def _span(self, item: Transcribed) -> tuple[tuple[float, float], str]:
        decision = item.decision_audio_sec
        if self.spanner is not None:
            start = self.since - ALIGN_BACK_SEC
            if self.settings.get("align_from") == "prev_end" and self.last_end is not None:
                start = self.last_end
            start = max(self.offset, min(start, decision))
            found = self.spanner.find(
                self._slice(start, decision), start, item.original, self.candidates
            )
            if found is not None:
                self.last_end = found[1]
                return found, "aligner"
        return lag_span(self.since, decision, self.settings.get("lag_sec", 0.5)), "lag"

    def _segment_probs(self, span: tuple[float, float]) -> tuple[dict | None, dict | None]:
        # Keep the end: on a long fallback span the commit's speech is nearest the decision.
        speech = speech_only(self._slice(*span))[-int(SR * self.settings.get("max_sec", 10.0)) :]
        if len(speech) < SR * MIN_SPEECH_SEC:
            return None, None
        raw = {name: s.scores(speech, self.candidates) for name, s in self.scorers.items()}
        merged = combine(
            raw, self.candidates, self.settings.get("combine", "mean"), self.settings.get("weights")
        )
        if merged is None:
            return None, raw
        prior = with_prior(
            self._scaled(merged), self.previous, self.settings.get("stay_prior", 0.7)
        )
        return normalise(prior, self.candidates), raw

    def _timeline_probs(
        self, span: tuple[float, float], decision: float
    ) -> tuple[dict | None, list]:
        """Probabilities over the span, and the provisional steps added for this decision (a
        centered timeline has not yet scored its last steps; they are scored with the audio up to
        the decision and dropped afterwards)."""
        known = [(t, s) for t, s in self.steps if t <= decision + 1e-6]
        silent = (self.silent + [False] * len(known))[: len(known)]
        extra = []
        if self.settings.get("centered"):
            t = self.next_step
            while t <= decision + 1e-6:
                scores, quiet, raw = self._score_step(t, decision)
                extra.append({"t": t, "scores": scores, "silent": quiet, "raw": raw})
                known.append((t, scores))
                silent.append(quiet)
                t += self.settings.get("step_sec", 0.5)
        if not known:
            return None, extra
        gaps = [False] + silent[:-1]
        posts = smooth(
            [s for _, s in known],
            self.candidates,
            self.settings.get("p_stay", 0.9),
            self.settings.get("p_gap"),
            gaps,
        )
        # Trailing windows end at t: a step's centre is half the shortest window earlier.
        half = 0.0 if self.settings.get("centered") else min(self._windows()) / 2
        probs = span_mass(
            known,
            posts,
            span,
            half,
            self.candidates,
            self.settings.get("read", "mean"),
            self.settings.get("read_sec", 1.0),
        )
        return probs, extra

    def tail_language(self, sec: float) -> tuple[str, float] | None:
        """Language of the last `sec` of heard audio and its probability (timeline mode), for a
        transcriber deciding whether its slot crossed a language switch."""
        if self.mode != "timeline" or len(self.candidates) < 2:
            return None
        end = self._end_sec()
        probs, _ = self._timeline_probs((max(self.offset, end - sec), end), end)
        if not probs:
            return None
        lang = max(probs, key=probs.get)
        return lang, probs[lang]

    async def detect(self, item: Transcribed) -> str | None:
        if len(self.candidates) < 2:
            return None
        span, source = await asyncio.to_thread(self._span, item)
        raw, extra = None, []
        by_text = None
        if span[1] - span[0] < self.settings.get("short_text_sec", 0.0):
            by_text = text_language(item.original, self.candidates)
        if by_text is not None:
            probs, source = {c: float(c == by_text) for c in self.candidates}, source + "+text"
        elif self.mode == "segment":
            probs, raw = await asyncio.to_thread(self._segment_probs, span)
        else:
            probs, extra = await asyncio.to_thread(
                self._timeline_probs, span, item.decision_audio_sec
            )
        detected = decide(probs, self.previous, self.settings.get("conf", 0.7))
        if by_text is None and probs and detected is not None:
            written = text_language(item.original, self.candidates)
            if (
                written is not None
                and written != detected
                and max(probs.values()) < self.settings.get("text_override_conf", 0.0)
            ):
                detected, source = written, source + "+text-override"
        self.trace.append(
            {
                "decision": item.decision_audio_sec,
                "span": span,
                "src": source,
                "raw": raw,
                "probs": probs,
                "previous": self.previous,
                "detected": detected,
                "provisional": extra,
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
