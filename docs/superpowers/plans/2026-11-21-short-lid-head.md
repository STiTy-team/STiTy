# Short-segment LID head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `whisper-head` scorer (frozen whisper-small encoder + attentive-statistics head, 7 languages) that beats `whisper-token` on clips under 1 s, checked on FLEURS test and YODAS2 held-out, then in the pipeline.

**Architecture:** crops (`short_data.py`, running) → cached encoder frames (`short_features.py`) → head (`audio/head.py`) trained by `short_train.py` → scorer in `scorers.py` → `short_eval.py` (clip buckets) → `replay2.py`/sweep (pipeline) → vela COMET.

**Tech Stack:** torch, transformers whisper-small, numpy.

**Spec:** `docs/superpowers/specs/2026-11-20-short-lid-head-design.md`

## Global Constraints

- Languages, in this order: ko, en, ja, es, zh, fr, de.
- Training/early stopping never use FLEURS test, YODAS2 shard 2, or northstar.
- Success: clips < 1 s, 7-language mean accuracy beats whisper-token by >= 2 points on FLEURS test and YODAS2 held-out; 1.5-4 s loses <= 0.5 points.
- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= uv run --project bench --with pytest python -m pytest -q core/tests/<file>`; no tests under `core/components/`.
- Long steps in tmux with markers under `logs/audio-lid/`.

## Review Focus

- A clip shorter than one encoder frame (20 ms) or empty → no score (all -inf), not a crash.
- A candidate outside the head's 7 languages → -inf; the others still decide.
- Batch with mixed clip lengths → each clip pools only its own frames (mask), same result as one by one.
- Checkpoint language order differs from the request order → scores mapped by name.
- FLEURS test word-boundary crops for zh/ja have no spaces → crops built from aligner word times, not text splitting.

### Task 1: Head model (`core/components/langid/audio/head.py`)
- Produces `LidHead(n_langs, dim=768, hidden=256)`: `forward(frames: Tensor[B,T,768], mask: Tensor[B,T]) -> logits[B,n]` with attentive statistics pooling (masked); `save(path, langs, meta)`, `load(path) -> (head, langs, meta)`.
- Tests (`core/tests/test_audio_lid_head.py`): masked padding does not change logits of a short clip vs the same clip alone; save/load round trip keeps langs and logits.

### Task 2: Feature cache (`core/research/lid_module/short_features.py`)
- whisper-small encoder on batches of crops (padded to 30 s by the feature extractor), keep `ceil(len/320)` frames per crop (encoder frame = 20 ms), write `features/<split>.pt` with `frames` (concatenated fp16), `lengths`, `langs`, `source`, `dur`.

### Task 3: Training (`core/research/lid_module/short_train.py`)
- Balanced sampling over (language, duration bucket); AdamW 1e-3, batch 256, 30 epochs max, label smoothing 0.05; early stop on dev accuracy for clips < 1 s (FLEURS dev + YODAS2 dev); checkpoint `models/lidhead/whisper_small_7lang.pt`.

### Task 4: Scorer `whisper-head` (`scorers.py`)
- `WhisperHeadScorer(model=checkpoint)`: whisper-small encoder + head; `scores`/`scores_batch` → log-softmax over candidates the head knows, others -inf.
- Test: GPU smoke like the other scorers (ko vs en FLEURS clip).

### Task 5: Clip evaluation (`core/research/lid_module/short_eval.py`)
- FLEURS test word-boundary crops (alignment.jsonl, 7 languages) and YODAS2 test crops; buckets 0.3-0.6/0.6-1.0/1.0-1.5/1.5-4.0; 7-way and en+X pairwise accuracy; whisper-token vs whisper-head; table to `logs/audio-lid/eval/short_eval.md`.

### Task 6: Pipeline
- `replay2.py --scorers whisper-token whisper-head@<ckpt>` (centered timeline) on northstar + 4 mixes; compare head-only / joined / short-span-only; if success criteria hold, vela COMET on northstar v4 ko/en; report + Confluence-ready notes.
