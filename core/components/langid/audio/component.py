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
SR_PAD_SEC = 1.0
CUT_BACK_SEC = 3.0
CUT_AHEAD_SEC = 0.5
CUT_GRID_SEC = 0.25
CUT_SIDE_SEC = 2.0
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
    _pump_task = None

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
        self._pump_task = None
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
        """Take audio. In timeline mode the 0.5 s steps are scored off the event loop: inside a
        running loop hear() only schedules `_pump` (a worker thread scores, the loop appends),
        so a live room is not blocked; without a loop (tests, scripts) it scores in place.
        Readers (`detect`, `confirmed_switch`, `tail_language`) await `_settle` first, so they
        see the same steps as the in-place scoring."""
        self.audio = np.concatenate([self.audio, audio_mod.from_pcm_bytes(audio)])
        if self.mode == "timeline" and len(self.candidates) > 1:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is None:
                step = self.settings.get("step_sec", 0.5)
                while self._ready(self.next_step):
                    self._add_step(self.next_step, self._end_sec())
                    self.next_step += step
            elif self._pump_task is None or self._pump_task.done():
                self._pump_task = loop.create_task(self._pump())
        # Keep the audio a step still waiting to be scored needs.
        keep_from = min(
            self._end_sec() - KEEP_SEC, self.next_step - max(self._windows()) - SR_PAD_SEC
        )
        if keep_from > self.offset:
            self.audio = self.audio[int((keep_from - self.offset) * SR) :]
            self.offset = keep_from
            kept = [i for i, s in enumerate(self.steps) if s[0] >= keep_from]
            self.steps = [self.steps[i] for i in kept]
            self.silent = [self.silent[i] for i in kept]
            if self.raw:
                self.raw = [self.raw[i] for i in kept]

    async def _pump(self) -> None:
        step = self.settings.get("step_sec", 0.5)
        while self._ready(self.next_step):
            t, limit, view = self.next_step, self._end_sec(), (self.audio, self.offset)
            scores, silent, raw = await asyncio.to_thread(self._score_step, t, limit, view)
            self.steps.append((t, scores))
            self.silent.append(silent)
            if self.settings.get("keep_raw"):
                self.raw.append(raw)
            self.next_step = t + step

    async def _settle(self) -> None:
        """Wait until every step the heard audio allows is scored."""
        while True:
            task = self._pump_task
            if task is not None and not task.done():
                await task
            elif self.mode == "timeline" and len(self.candidates) > 1 and self._ready(
                self.next_step
            ):
                self._pump_task = asyncio.get_running_loop().create_task(self._pump())
            else:
                return

    def _add_step(self, t: float, limit: float) -> None:
        scores, silent, raw = self._score_step(t, limit)
        self.steps.append((t, scores))
        self.silent.append(silent)
        if self.settings.get("keep_raw"):
            self.raw.append(raw)

    def _score_step(
        self, t: float, limit: float, view: tuple | None = None
    ) -> tuple[dict | None, bool, dict]:
        """Log-scores at step `t` (windows merged across scorers, then averaged over windows),
        whether the step's own slot is silent, and the raw per-scorer, per-window scores. Audio
        after `limit` is not used (a centered window near the decision is cut there)."""
        centered = self.settings.get("centered", False)
        step = self.settings.get("step_sec", 0.5)
        windows = self._windows()
        bounds = {
            w: ((t - w / 2, min(t + w / 2, limit)) if centered else (t - w, t)) for w in windows
        }
        held, held_from = view if view is not None else (self.audio, self.offset)
        lo = max(min(b[0] for b in bounds.values()), held_from)
        hi = max(b[1] for b in bounds.values())
        audio = held[max(0, int((lo - held_from) * SR)) : max(0, int((hi - held_from) * SR))]
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

    async def tail_language(self, sec: float) -> tuple[str, float] | None:
        """Off the event loop: see `_tail_language`."""
        await self._settle()
        return await asyncio.to_thread(self._tail_language, sec)

    def _tail_language(self, sec: float) -> tuple[str, float] | None:
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

    async def confirmed_switch(self, header: str, min_sec: float, conf: float) -> str | None:
        """Off the event loop: see `_confirmed_switch`."""
        await self._settle()
        return await asyncio.to_thread(self._confirmed_switch, header, min_sec, conf)

    async def confirmed_switch_at(
        self, header: str, min_sec: float, conf: float
    ) -> tuple[str, float] | None:
        """Like `confirmed_switch`, with the stream time where the held language starts (half a
        step before its first step)."""
        await self._settle()
        return await asyncio.to_thread(self._confirmed_switch, header, min_sec, conf, True)

    def _confirmed_switch(self, header: str, min_sec: float, conf: float, with_time: bool = False):
        """A language other than `header` that the smoothed timeline has held with p >= `conf`
        for at least `min_sec` up to the newest step, or None."""
        if self.mode != "timeline" or len(self.candidates) < 2:
            return None
        end = self._end_sec()
        known = [(t, sc) for t, sc in self.steps if t <= end + 1e-6]
        silent = (self.silent + [False] * len(known))[: len(known)]
        if self.settings.get("centered"):
            t = self.next_step
            while t <= end + 1e-6:
                scores, quiet, _ = self._score_step(t, end)
                known.append((t, scores))
                silent.append(quiet)
                t += self.settings.get("step_sec", 0.5)
        if not known:
            return None
        posts = smooth(
            [sc for _, sc in known],
            self.candidates,
            self.settings.get("p_stay", 0.9),
            self.settings.get("p_gap"),
            [False] + silent[:-1],
        )
        step = self.settings.get("step_sec", 0.5)
        lang, held = None, 0
        for post in reversed(posts):
            if not post:
                break
            best = max(post, key=post.get)
            if best == header or post[best] < conf or (lang and best != lang):
                break
            lang, held = best, held + 1
        if not lang or held * step < min_sec:
            return None
        if not with_time:
            return lang
        first = known[len(known) - held][0]
        half = step / 2 if self.settings.get("centered") else min(self._windows()) / 2
        return lang, first - half

    def best_cut(
        self, audio: np.ndarray, at: float, old: str, new: str
    ) -> tuple[float, float, float] | None:
        """Where in `audio` (seconds) the speech turns from `old` to `new`, searched in
        [at - CUT_BACK_SEC, at + CUT_AHEAD_SEC] around the timeline's switch time `at`.
        Candidates are the middles of VAD pauses, else a CUT_GRID_SEC grid; each is scored by
        log P(old | up to CUT_SIDE_SEC of speech before) + log P(new | the speech after), with
        the whisper scorer. Returns (cut, p_new after the cut, margin over the best candidate
        elsewhere), or None when no candidate has 0.5 s of speech after it."""
        scorer = next(iter(self.scorers.values()))
        cands = tuple(self.candidates)
        dur = len(audio) / SR
        lo, hi = max(0.0, at - CUT_BACK_SEC), min(dur - 0.5, at + CUT_AHEAD_SEC)
        if hi <= lo:
            return None
        spans = speech_spans(audio)
        edges = [0] + [x for a_, b_ in spans for x in (a_, b_)] + [len(audio)]
        points = [
            (a_ + b_) / 2 / SR
            for a_, b_ in zip(edges[0::2], edges[1::2])
            if (b_ - a_) / SR >= 0.15 and lo <= (a_ + b_) / 2 / SR <= hi
        ]
        if not points:
            points = list(np.arange(lo, hi + 1e-9, CUT_GRID_SEC))
        lefts = [speech_only(audio[int(max(0.0, c - CUT_SIDE_SEC) * SR) : int(c * SR)]) for c in points]
        rights = [speech_only(audio[int(c * SR) :]) for c in points]
        keep = [i for i, r in enumerate(rights) if len(r) >= SR // 2]
        if not keep:
            return None
        points = [points[i] for i in keep]
        lefts = [lefts[i] for i in keep]
        rights = [rights[i] for i in keep]
        rs = scorer.scores_batch(rights, cands)
        ls = scorer.scores_batch(lefts, cands)

        def logp(scores: dict, lang: str) -> float:
            vals = np.array([scores[c] for c in cands], dtype=np.float64)
            top = vals.max()
            if not np.isfinite(top):
                return float(np.log(1.0 / len(cands)))
            return float(scores[lang] - top - np.log(np.exp(vals - top).sum()))

        totals = []
        for left, lsc, rsc in zip(lefts, ls, rs):
            before = logp(lsc, old) if len(left) >= int(0.3 * SR) else float(np.log(0.5))
            totals.append(before + logp(rsc, new))
        order = sorted(range(len(points)), key=lambda i: totals[i], reverse=True)
        best = order[0]
        margin = totals[best] - totals[order[1]] if len(order) > 1 else float("inf")
        p_new = float(np.exp(logp(rs[best], new)))
        log.info(
            "[LID-CUT] at=%.2f cands=%d cut=%.2f p_new=%.3f margin=%.2f",
            at, len(points), points[best], p_new, margin,
        )
        return float(points[best]), p_new, float(margin)

    def split_words(
        self, audio: np.ndarray, words: list, old: str, new: str
    ) -> int | None:
        """Index of the first word of `words` (aligned in `audio`) spoken in `new`: the split
        that maximises log P(old | before) + log P(new | from there). None when the audio from
        the best split is not heard as `new` (p < 0.5) or is shorter than 0.5 s."""
        scorer = next(iter(self.scorers.values()))
        cands = tuple(self.candidates)

        def clip(a: float, b: float) -> np.ndarray:
            return audio[int(a * SR) : int(b * SR)]

        def logp(scores: dict, lang: str) -> float:
            vals = np.array([scores[c] for c in cands], dtype=np.float64)
            top = vals.max()
            if not np.isfinite(top):
                return float("-inf")
            return float(scores[lang] - top - np.log(np.exp(vals - top).sum()))

        end = words[-1][2]
        # A split needs 0.5 s after it, and either nothing before it or 0.3 s to judge.
        splits = [
            k
            for k in range(len(words))
            if end - words[k][1] >= 0.5 and (k == 0 or words[k][1] - words[0][1] >= 0.3)
        ]
        if not splits:
            return None
        sufs = scorer.scores_batch([clip(words[k][1], end) for k in splits], cands)
        pres = scorer.scores_batch(
            [clip(words[0][1], words[k][1]) if k else audio[:0] for k in splits], cands
        )
        best, best_score, best_new = None, float("-inf"), float("-inf")
        for k, suf, pre in zip(splits, sufs, pres):
            total = logp(suf, new) + (logp(pre, old) if k else 0.0)
            if total > best_score:
                best, best_score, best_new = k, total, logp(suf, new)
        log.info("[LID-SPLIT] k=%s/%d p_new=%.3f", best, len(words), float(np.exp(best_new)))
        if best is None or np.exp(best_new) < 0.5:
            return None
        return best

    def tail_words(self, audio: np.ndarray, head: str, tail: str) -> list | None:
        if self.spanner is None:
            return None
        return self.spanner.tail_words(audio, head, tail, self.candidates)

    def tail_start(self, audio: np.ndarray, head: str, tail: str) -> float | None:
        """Seconds into `audio` (a transcriber's slot) where the uncommitted `tail` starts."""
        if self.spanner is None:
            return None
        return self.spanner.split_at(audio, head, tail, self.candidates)

    async def detect(self, item: Transcribed) -> str | None:
        if len(self.candidates) < 2:
            return None
        await self._settle()
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
