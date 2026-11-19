# Audio LID module — design

Date: 2026-11-19. Branch: `exp/audio-lid` (on top of `exp/vela-probelid`, which carries the
`lang_detect_route_*` metric, the probe langid and the r2 scoring on main).

## Why

vela picks each commit's translation direction from its language. The probe LID (a linear head on
the seg-mix ASR's own encoder) lowered COMET on northstar r2 (ko 0.742 -> 0.730, en 0.706 -> 0.696):
14 of 287 commits flipped, 10 of them because the judged audio (previous decision -> this decision)
held the next utterance, not the commit's own speech. The `alphabet` rule (script of the
transcript) is right 99.6% on ko/en, but it depends on the ASR's text and fails on transliteration
("오 스피킹 of 부산") and on same-script language pairs.

The user wants a **general LID module**: it works for any language set and any ASR model, and the
language decision is made **from audio only**.

## Goals

- A langid component that decides each commit's language from the audio of that commit's own
  speech, for any candidate language set given by the session (`start(languages=...)`, i.e. the
  `langMap` source languages).
- Independence from the ASR: it uses only the audio stream, the commit's text and decision time —
  the text only to locate the speech (forced alignment), never to judge the language. No ASR
  language tag, no ASR encoder features.
- Two designs compared on the same data, each with several interchangeable acoustic scorers:
  - **A — segment judge**: score the aligned span of each commit once.
  - **B — language timeline**: score the audio continuously (0.5 s steps, several window sizes),
    smooth with an HMM, and read the aligned span off the timeline at commit time.
- Overnight evaluation (2026-11-19 -> 20) on northstar r2 plus FLEURS-mixed multilingual
  conversations, then a COMET check of the best configurations inside vela.

## Non-goals

- Training a new acoustic LID model (kept as a later upgrade of the scorer slot).
- Pre-commit consumers (brig `lid_slot_reset`, partial preview). B's timeline is built so they can
  read provisional values later, but nothing is wired to them now.
- Replacing `alphabet` in shipped configs. This is an experiment.

## Anti-goals

- Deciding language from the transcript's script or words (that is what `alphabet` already does).
- Anything that only works for ko/en or only with Qwen3-ASR.

## Constraints

- One GPU (RTX 4090, 24 GB), shared with the ASR (vLLM, 0.30) and Hy-MT2 in vela runs.
- gpu2's bench queue window opens 01:00–07:00 KST; the queue is empty now. Runs here go through
  `make bench` in tmux; if a queue job appears it will share the GPU.
- `Qwen3-ForcedAligner-0.6B` (cached) supports 11 languages: zh, en, yue, fr, de, it, ja, ko, pt,
  ru, es. It takes a `language` argument used for word tokenization only.
- Scorers already cached: whisper tiny/base/small/medium/large-v3-turbo, facebook/mms-lid-126/256,
  speechbrain/lang-id-voxlingua107-ecapa.
- Team history (core/research/zprobe/lid): whisper-small was 0.910 on commit windows / 0.762 on
  short ones; comparing per-candidate forced-decode log-probs beat the language-token argmax for
  non-native speech; MMS-LID was 0.890; VAD before LID is essential (0.5 s windows with leading
  silence were near-useless).

## Approaches considered

| | What | Trade-off |
|---|---|---|
| A — segment judge | Align commit text -> span; VAD-trim; score once; prior = previous language | Small, stateless across commits except the prior. No context around short commits |
| **B — language timeline** | Continuous multi-window scoring -> HMM (sticky language) -> read the aligned span | Uses the ~2 s commit lag as free look-ahead; same timeline can serve pre-commit consumers later. More compute, stateful |
| C — train own LID | Simulated code-switching data, new model | Most accurate potentially, too big for tonight; becomes a scorer later |

The user chose **A and B both, each with several scorers**, compared on the same data.

## Design

### Shared parts (`core/components/langid/audio/`)

- **Span finder.** Keeps a rolling audio buffer (30 s). At a commit, aligns the commit text against
  the audio from `max(prev_decision - 6 s, buffer start)` to `decision` with Qwen3-ForcedAligner and
  takes first-word start .. last-word end. The aligner's `language` is chosen from the text's
  script only for tokenization (Hangul -> Korean, kana/kanji -> Japanese, Han -> Chinese, else
  English, which tokenizes on spaces like es/fr/de). If alignment fails, returns an empty or
  implausible span (shorter than 0.2 s, or longer than 4x the text's expected duration), or the
  session language is outside the aligner's 11, it falls back to `[prev_decision - lag, decision - lag]`
  with `lag_sec` (default 0.5, the best value in the replay of 2026-11-19). The span source is
  logged.
- **VAD.** Silero (already a dependency) on the span / windows; non-speech frames are dropped before
  scoring. A span with < 0.3 s of speech gets no acoustic verdict.
- **Scorers** (one interface: `scores(audio, candidates) -> {lang: log-prob}`, restricted to the
  session's candidates and renormalised):
  - `whisper-token`: whisper language-token logits (small, large-v3-turbo).
  - `whisper-decode`: per-candidate forced-decode mean log-prob (small), the team's better method
    for non-native speech.
  - `ecapa-voxlingua107`: SpeechBrain ECAPA, 107 languages.
  - `mms-lid-256`.
  Codes are mapped between each model's label set and the project's 2-letter codes.
- **Decision.** The candidate with the highest posterior. If the top posterior is below `conf`
  (default 0.7) the commit keeps the previous commit's language; with no previous language it returns
  `None` (the ASR tag stays). The confidence and span are logged per commit
  (`[LANGID] audio=… conf=… span=… src=aligner|lag -> …`).

### A — segment judge (`mode: segment`)

Score the VAD-trimmed aligned span once (cap 10 s), add a log prior favouring the previous
language (`stay_prior`, default 0.7 vs uniform rest), decide.

### B — language timeline (`mode: timeline`)

- Every `step_sec` (0.5 s) of audio, score trailing windows of `windows` sizes (default 1, 2, 4 s),
  VAD-trimmed; windows with too little speech are skipped. Average the windows' log-probs at that
  time step.
- Smoothing: a forward–backward HMM over the time steps since the last decision minus 10 s, states =
  candidates, self-transition `p_stay` (default 0.9). At a commit the steps up to the decision time
  are known, so the smoothed posterior over the commit's span uses audio after the span (the commit
  lag) as look-ahead.
- Commit language = the candidate with the most posterior mass over the time steps inside the
  aligned span (speech steps only). If the span has fewer than 1 step, use the nearest step.
- The timeline (step time, smoothed posteriors) is kept so a later pre-commit reader can use it.

### Component wiring

One registered langid `audio` with `mode: segment|timeline`, `scorer: <name>`, `model`, `conf`,
`stay_prior` / `p_stay`, `windows`, `step_sec`, `lag_sec`, `aligner: true|false`. It needs
`hear()` (cascade:v2 calls it since 44828289) and `start(languages=...)`.

## Evaluation

### Data (installed into the worker data root, additive)

- **northstar-260928 r2** (real ko/en dinner conversation, 6 scenes) — existing vela@v2 runs give
  the commits.
- FLEURS test mixes built with `mix.py`, longform, switching often with short gaps (`--switch 0.7
  --gap 0.2 1.0`), limited to about 30 minutes each:
  - `ko_kr+en_us` (different scripts, same pair as northstar)
  - `ja_jp+en_us` (different scripts, another pair)
  - `es_419+en_us` (same script — `alphabet` cannot work here)
  - `ko_kr+ja_jp+en_us` (three languages)
- Commits for FLEURS mixes come from a bench run per mix with base **Qwen3-ASR-1.7B** (`qwen3`
  transcriber, `commit: punct`, `translation: mock`) — a different ASR from vela's, which also
  tests ASR independence. Dataset and pipeline configs get new names per configs/README.md.

### Offline replay (fast, the main grid)

`core/research/lid_module/replay.py` replays each run's commits (text, decision time) through the
module on the scene audio and scores against the `lang_detect_route` truth (the reference sentence
the commit's text matches). Grid: mode {A, B} x scorer {whisper-token small, whisper-token
large-v3-turbo, whisper-decode small, ecapa-voxlingua107, mms-lid-256} x dataset {5}, plus baselines
`alphabet` and the ASR tag. Also for B: time-step accuracy against the manifest timeline. Outputs a
JSON per (mode, scorer, dataset) and one summary table (accuracy, per-language confusion, unmatched,
span source share, mean latency per commit, GPU memory).

### Bench check (COMET)

The best 2 configurations by replay accuracy go into vela (`langid: audio`) on
`northstar_ko+en-ko@v4` and `-en@v4`, compared with the vela@v2 runs already scored on r2 (v3,
same gold and normalization) and rerun on v4 if names differ. COMET + `lang_detect_route_accuracy`.

### Overnight order

1. Install FLEURS ja_jp, es_419 and build the 4 mixes; bench the 4 mixes with Qwen3-ASR-1.7B + mock.
2. Run the replay grid (A, B x scorers) on all 5 datasets.
3. Bench the top 2 configs in vela on northstar v4 ko/en.
4. Write the results into a report.

Long steps run in tmux with marker files (`.claude/rules/long-jobs-in-tmux.md`). No paid API.

## Decisions made after the builder check (coordinator's calls, flagged for the user)

- **Picking the 2 configs for the COMET check:** the best A and the best B (so A and B are both
  checked under COMET), ranked by mean `lang_detect_route_accuracy` over the 5 datasets, ties broken
  by northstar. They go in even if they lose to `alphabet`; the report states the comparison.
- **Real time:** bench is offline, so B scores synchronously inside `hear()` tonight. Latency
  metrics (LAAL/YAAL) of runs with B are not compared; the replay reports scoring time per second of
  audio so a background worker can be sized later.
- **Publishing:** FLEURS ja_jp / es_419 and the mixes are written to disk under the worker data root
  only (untracked in the datasets repo, nothing committed there). New dataset and pipeline configs
  are pushed to S3 one by one (the team's usual practice), the smoke configs stay local.
- **Schedule:** the night of 2026-11-20 (started 00:00 KST). Full grid; if it runs past 07:00 the
  slowest cells (B with whisper-decode, large-v3-turbo) go last so the rest is finished first. gpu2's
  queue is empty; it is not paused (shared setting) — a queue job would share the GPU.
- `whisper-decode` reuses `WhisperScorer` from `core/components/langid/whisper.py`.

## Left to the builder

Exact window/step/prior defaults after a smoke run; code-mapping tables per scorer; how the replay
batches scorer calls; dataset/pipeline config names within the naming rules.
