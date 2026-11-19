# Audio LID Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A langid component `audio` that decides each commit's language from audio only (A: segment judge, B: language timeline) with swappable scorers, plus an overnight evaluation on northstar r2 and FLEURS mixes.

**Architecture:** `core/components/langid/audio/` holds small pure parts (span finder, VAD trim, scorers, HMM smoother, decision) and one component wiring them by `mode`. A replay script drives the same parts offline from bench `items.jsonl` commits; the best A and B then run inside vela via `make bench`.

**Tech Stack:** Python 3.12, numpy, torch, transformers (whisper), speechbrain (ECAPA), silero-vad, qwen_asr `Qwen3ForcedAligner`.

**Spec:** `docs/superpowers/specs/2026-11-19-audio-lid-module-design.md`

## Global Constraints

- Decide language from audio only; transcript text is used only to align the span. No ASR language tag, no ASR encoder.
- Candidates = the session's `languages`; scores are restricted to them and renormalised.
- Aligner languages: zh, en, yue, fr, de, it, ja, ko, pt, ru, es. Otherwise span = `[prev_decision - lag_sec, decision - lag_sec]`, `lag_sec` 0.5.
- Defaults: `conf` 0.7 (below: keep previous language, none -> `None`), `stay_prior` 0.7 (A), `p_stay` 0.9, `windows` [1, 2, 4] s, `step_sec` 0.5 (B), span cap 10 s, min speech 0.3 s.
- Components are auto-imported by `discover`; no test files inside `core/components/` (tests go in `core/tests/`).
- Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= uv run --project bench --with pytest python -m pytest -q <path>`.
- Long jobs run in tmux with marker files under `/home/mobility/STiTy-team/STiTy/logs/audio-lid/`.

## Review Focus

- Aligner returns no words / garbage for a hallucinated or transliterated commit -> lag fallback, logged `src=lag`.
- A span that is all silence -> no acoustic verdict -> previous language kept.
- Candidate code a scorer does not know (e.g. `yue` for ECAPA) -> that candidate gets -inf, others still decide; all unknown -> `None`.
- First commit of a session (no previous language) with low confidence -> `None`, the ASR tag stays.
- Timeline older than the buffer (commit span before the kept audio) -> nearest step used, no crash.

---

### Task 1: Pure parts — decision, HMM smoother, span fallback, code maps

**Files:**
- Create: `core/components/langid/audio/__init__.py` (empty marker; registry discovers the subpackage)
- Create: `core/components/langid/audio/decide.py`
- Test: `core/tests/test_audio_lid_parts.py`

**Interfaces:**
- Produces:
  - `normalise(scores: dict[str, float], candidates: tuple[str, ...]) -> dict[str, float] | None` — log-probs restricted to candidates, softmax-normalised to probabilities; `None` if no candidate is finite.
  - `decide(probs: dict[str,float] | None, previous: str | None, conf: float) -> str | None`
  - `with_prior(logp: dict[str,float], previous: str | None, stay: float) -> dict[str,float]`
  - `smooth(steps: list[dict[str,float]], candidates, p_stay: float) -> list[dict[str,float]]` — forward–backward posteriors per step from per-step log-likelihoods.
  - `lag_span(prev_decision: float, decision: float, lag: float) -> tuple[float, float]`
  - `ALIGNER_LANGS: dict[str, str]` code -> aligner language name.

- [ ] Step 1: tests — `normalise({"ko":-1,"en":-3,"ja":0},("ko","en"))` sums to 1 with ko>en and no "ja"; all -inf -> None. `decide({"ko":.6,"en":.4},"en",0.7)=="en"`, `decide({"ko":.9,"en":.1},"en",0.7)=="ko"`, `decide(None,None,0.7) is None`. `smooth` with p_stay 0.9 on steps [ko strong, en weak, ko strong] keeps the middle step ko; with p_stay 0.5 the middle is en. `lag_span(10.0, 12.0, 0.5) == (9.5, 11.5)`.
- [ ] Step 2: run, fail. Step 3: implement (log-space forward–backward). Step 4: pass. Step 5: commit.

### Task 2: Scorers

**Files:**
- Create: `core/components/langid/audio/scorers.py`
- Test: `core/tests/test_audio_lid_scorers.py` (GPU-marked smoke: skipped when no CUDA)

**Interfaces:**
- Produces: `build_scorer(name: str, model: str | None, device: str | None) -> Scorer`; `Scorer.scores(audio: np.ndarray, candidates: tuple[str,...]) -> dict[str, float]` (log-scores; unknown candidate -> `-inf`). Names: `whisper-token` (default model openai/whisper-small; logits of `<|xx|>` after `<|startoftranscript|>`), `whisper-decode` (reuse `WhisperScorer._score` from `core/components/langid/whisper.py`), `ecapa` (speechbrain/lang-id-voxlingua107-ecapa, labels like `ko: Korean`), `mms` (facebook/mms-lid-256, ISO-639-3 labels mapped to 2-letter via a dict for ko/kor, en/eng, ja/jpn, es/spa, zh/cmn, fr/fra, de/deu …).
- [ ] Step 1: smoke test loads each scorer and on a 3 s FLEURS ko clip (`/home/mobility/STiTy-team/datasets/fleurs/ko_kr` first manifest item) ranks ko over en. Step 2–4: implement, pass. Step 5: commit.

### Task 3: Span finder and the `audio` component (A and B)

**Files:**
- Create: `core/components/langid/audio/span.py` — `AlignerSpan.find(audio: np.ndarray, audio_start_sec: float, text: str, candidates) -> tuple[float,float] | None` using `Qwen3ForcedAligner` (`Qwen/Qwen3-ForcedAligner-0.6B`); aligner language from text script (Hangul->Korean, kana->Japanese, Han->Chinese, else English); rejects spans < 0.2 s.
- Create: `core/components/langid/audio/component.py` — `@langids.register("audio")` `AudioLanguageDetector` with SETTINGS `mode, scorer, model, conf, stay_prior, p_stay, windows, step_sec, lag_sec, aligner, max_sec`; `hear()` keeps 30 s buffer and in `timeline` mode scores each new `step_sec` (VAD-trimmed trailing windows, averaged log-probs) into a step list; `detect()` finds the span (aligner, else lag), then A: score span with `with_prior`; B: `smooth` the steps of the last 30 s and sum posterior over steps inside the span (nearest step if none); then `decide`; logs `[LANGID] audio=… conf=… span=…-… src=aligner|lag -> …`.
- Create: `core/components/langid/audio/vad.py` — `speech_only(audio) -> np.ndarray` via silero (`get_speech_timestamps`).
- Test: `core/tests/test_audio_lid_component.py` with a fake scorer and a fake span finder: A keeps previous language below conf; B with steps ko,ko,en(weak),ko returns ko for a span over the en step; aligner `None` -> lag span used.
- [ ] Steps: test, fail, implement, pass, commit.

### Task 4: Replay harness

**Files:**
- Create: `core/research/lid_module/replay.py`

**Interfaces:** CLI `--run <items.jsonl> --data-root <dir> --mode segment|timeline --scorer <name> [--model M] --out <json>`; reads commits (`translated` records), scene audio (row `audio`, `offset`), truth from `bench.metrics.asr._routes`; drives `AudioLanguageDetector` exactly as the pipeline would (hear in 0.1 s chunks — 2.0 s for speed is fine — then detect at each commit's `decision_audio_sec`); writes accuracy, confusion, unmatched, span-source share, seconds of compute per audio second; also baseline accuracy of `alphabet` and of the ASR tag (`transcribed` language).
- [ ] Step 1: run on `northstar_ko+en-ko@v3/vela@v2` with A + whisper-token; accuracy printed and JSON written. Step 2: commit.

### Task 5: Data + commit runs for FLEURS mixes

- Datasets (already being built by `logs/audio-lid/data.sh`): `fleurs/{ko_kr+en_us,ja_jp+en_us,es_419+en_us,ko_kr+ja_jp+en_us}.switch0.7-gap0.2-1.0` (exact dir names from `data.log`).
- Dataset configs `fleurs_ko+en_mix`, `fleurs_ja+en_mix`, `fleurs_es+en_mix`, `fleurs_ko+ja+en_mix` (`longform: true`, `target: en` — mock MT so target only drives routing), pipeline `asr.qwen3-1.7b+mt.mock` (copy of `asr.qwen3-0.6b+mt.mock` with `Qwen/Qwen3-ASR-1.7B`). Push configs to S3 one by one.
- Bench each mix in one tmux chain; outputs `bench/runs/<dataset>/asr.qwen3-1.7b+mt.mock/items.jsonl`.

### Task 6: Grid, COMET check, report

- Grid chain in tmux: mode {segment, timeline} x scorer {whisper-token small, whisper-token large-v3-turbo, whisper-decode small, ecapa, mms} x 5 runs; slowest cells last. Summary table `logs/audio-lid/grid.md`.
- Pick best A and best B by mean accuracy over datasets (tie: northstar). Pipelines `vela_audiolid-a`, `vela_audiolid-b` (vela@v2 with `langid: audio …`), bench on `northstar_ko+en-ko`, `northstar_ko+en-en` (v4), plus `vela` on v4 as baseline.
- Report `reports/audio-lid-overnight.md` (Korean, plain): grid, COMET, findings. Update memory.
