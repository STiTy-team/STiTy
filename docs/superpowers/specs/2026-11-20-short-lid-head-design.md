# Short-segment LID head on the whisper encoder — design

Date: 2026-11-20. Branch: `exp/audio-lid`.

## Why

The audio langid (`langid: audio`, timeline mode) now matches the alphabet baseline on vela COMET
for →ko and is 0.004 behind for →en. Of its 8 remaining errors over northstar + 4 FLEURS mixes,
7 are the scorer itself being wrong (smoothing swallowed only 1), almost all on commits of
0.4–0.6 s ("What", "Wait, wait.", "아 그래.", "있다"). Changing windows, switching to ECAPA/MMS, or
adding a phone-profile scorer did not help (phone scorers were far worse: 0.70 on short northstar
spans vs whisper 0.934). The user wants the scorer to tell languages apart on short segments, **for
many languages**, not only ko/en.

## Goals

- A new scorer, `whisper-head`, that decides among 7 languages — ko, en, ja, es, zh, fr, de — from
  a short clip better than `whisper-token`, and is at least as good on longer clips. The timeline
  scores 1, 2 and 4 s windows, so the head must work from 0.3 s up to 4 s.
- Language set is configuration: adding a language = adding its data and retraining.
- Plugs into the existing component as one more scorer (`scorers: [whisper-head@<checkpoint>]`),
  restricted to the session's candidates at inference like the others.

## Non-goals

- Training whisper itself (approach C) or ECAPA (approach B).
- Languages beyond the 7.
- Changing the timeline / span / smoothing logic.

## Anti-goals

- Anything trained or tuned on northstar or on FLEURS **test** (they are the evaluation).
- A scorer that only works for ko/en.
- Text-based features.

## Approaches considered

| | What | Trade-off |
|---|---|---|
| **A — head on frozen whisper-small encoder (picked)** | Pool the encoder frames that cover the clip, small classifier | Keeps whisper's strength, features cached once, minutes to train; cannot learn what the encoder does not hear |
| B — fine-tune ECAPA (VoxLingua107) | Small dedicated LID model | Starts weaker than whisper on northstar (0.91 vs 0.98) |
| C — LoRA on whisper's language token | Same scorer interface | Heaviest; changes whisper's behaviour |

## Design

### Data (`core/research/lid_module/short_data.py`)

- **Training sources** (all audio-only; no transcripts needed):
  - FLEURS **train**, 7 languages (`ko_kr en_us ja_jp es_419 cmn_hans_cn fr_fr de_de`), streamed
    from the HF archive like `build_phone_profiles.py`: up to 1,500 utterances per language.
  - Non-read speech for **all 7 languages**: YODAS2 (`espnet/yodas2` on HF, YouTube audio,
    CC-BY-3.0, not gated), streamed from the first shard of each language (`data/<lang>000/audio/
    00000000.tar.gz`) and read only up to 200 videos per language. Language labels come from the
    channel metadata, so some audio is in another language or is music; this label noise is
    accepted and noted. VoxLingua107 was the first choice but its host is unreachable from this
    machine. Korean/English-only conversational corpora (ksponspeech, aihub71693) are left out so
    the head cannot learn "spontaneous speech = ko or en".
  - Dev (early stopping, model choice): FLEURS **dev** split, 200 utterances per language, and
    YODAS2 videos from a different shard (the last shard of each language), 50 per language.
- **Crops:** silero VAD finds speech; from each FLEURS utterance take 3 random crops, and from each
  YODAS2 video 15, of a duration drawn from {0.3–0.6, 0.6–1.0, 1.0–1.5, 1.5–4.0} s (weights 0.3,
  0.3, 0.2, 0.2), each placed inside the speech of the recording. Only the speech samples of the
  crop are kept (as the component does). FLEURS and YODAS2 crops are about equal per language.
- **Balancing:** per language and duration bucket, by sampling weights at training time.
- **Features cached:** whisper-small encoder output for each crop (padded to 30 s as whisper
  requires), keeping only the frames that cover the crop (`ceil(dur * 50)` frames, 768 dims,
  fp16) in shard files under `models/lidhead/features/<split>/`.

### Model (`core/components/langid/audio/head.py`)

- Input: encoder frames of the clip (T × 768).
- Attentive statistics pooling (attention-weighted mean and standard deviation) → 1,536 →
  Linear 256 → ReLU → dropout 0.2 → Linear 7.
- Loss: cross-entropy with label smoothing 0.05. AdamW lr 1e-3, batch 256, up to 30 epochs, early
  stop on dev accuracy on clips < 1.0 s.
- Checkpoint: `models/lidhead/whisper_small_7lang.pt` with the language order and training
  metadata.

### Scorer `whisper-head` (`scorers.py`)

- Loads whisper-small encoder + the head; `scores(audio, candidates)` returns log-softmax over the
  candidates (logits restricted to them); a candidate outside the head's languages gets -inf.
- `scores_batch` runs the windows of one step in one encoder pass.

### Evaluation (`core/research/lid_module/short_eval.py`)

1. **Short-clip accuracy per language** on two sets, buckets 0.3–0.6 / 0.6–1.0 / 1.0–1.5 /
   1.5–4.0 s, `whisper-token` vs `whisper-head`, restricted to the 7 candidates and to pairs
   (en + one other):
   - FLEURS **test** clips cut at word boundaries with the alignments (`fleurs/<lang>/alignment.jsonl`;
     zh, fr, de test sets are installed and aligned first — a GPU step).
   - YODAS2 held-out videos (a shard used for neither training nor dev), VAD crops — the
     non-read check for all 7 languages.
2. **Pipeline-level:** `replay2.py` with `--scorers whisper-token whisper-head` on northstar + 4
   FLEURS mixes, then a `sweep2.py`-style comparison, including the head replacing or joining
   `whisper-token` on the 1/2/4 s windows and "head on short spans only" (offline first; a component
   option only if it helps).
3. **Success:** on clips under 1 s, the 7-language mean accuracy beats `whisper-token` by at least
   2 points on both FLEURS test and YODAS2 held-out, without losing more than 0.5 points on 1.5–4 s.
   Then, if the pipeline replay keeps the FLEURS mixes at least as good and northstar errors do not
   increase, a vela COMET run on northstar v4 ko/en.

Pipeline settings are picked on the FLEURS mixes as in the earlier grids (those mixes come from
FLEURS test, so their numbers are optimistic); the head itself never sees FLEURS test, YODAS2
held-out videos or northstar in training or early stopping.

## Constraints

- One RTX 4090; bench queue window 01:00–07:00 on gpu2 (no queue jobs now).
- Long steps (data, features, replay) run in tmux with marker files.
- No paid APIs.

## Left to the builder

Exact crop counts after a size check, shard format, attention pooling details, whether to cache
features for the dev set from FLEURS dev or a held-out slice of train.
