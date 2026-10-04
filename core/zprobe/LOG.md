# Handoff: Qwen3-ASR "common z" probe (stage 0)

Goal: find whether the frozen Qwen3-ASR-1.7B holds a language-neutral representation of what
was said, and at which layer (encoder output vs LLM middle layers). Spike — throwaway code.

Everything lives in this directory:

- `probe_z.py` — extract / analyze / gen subcommands. Hooks on `audio_tower.ln_post`
  (pre-projector encoder output) and every LLM decoder layer. Pooled vectors per utterance.
- `run.sh` — smoke (3 utts) -> full en/ko extract (270 FLEURS test sentences, 4-way common
  ids with ja/zh is 248 but ja/zh audio not downloaded yet) -> `analyze.md` -> `gen.log`.
- Runs in tmux session `probe-z`. Markers: `extract.done`, `all.done`. Logs: `smoke.log`,
  `extract.log`.
- Env: `/home/skkai/miniforge3/envs/stity/bin/python`. Data: `~/datasets/fleurs/data/{en_us,ko_kr}`.

What `analyze.md` contains (per layer L0..L28 plus `enc`):

1. cross-lingual retrieval en<->ko acc@1 (raw and mean-centered), same-vs-random cosine gap,
   for audio_mean / asr_pos / text_mean / text_last pooling.
2. language-ID linear probe accuracy on audio_mean.
3. audio vs text convergence: same-language audio->text retrieval, cross-language audio->text.
4. logit lens at teacher-forced transcript positions: top-1 accuracy, script of top-1 token.

`gen.log`: A) does the LLM still translate text? B) language-tag swap on audio
(English audio, forced "Korean" tag). C) translation instruction in system prompt.

Decision rule agreed with the user: retrieval peak at some LLM layer => z exists, go to
stage 1 (leave-pairs-out fine-tune with target-language tag). No peak / zero-shot off-target
=> no z; N+M joint training still fine but unseen pairs need pivot via English.

Next after en/ko: download FLEURS test audio for ja_jp and cmn_hans_cn into
`~/datasets/fleurs/data/<lang>/audio/test.tar.gz` (the HF cache has only TSVs), rerun
extract with `--langs en,ko,ja,zh`.

---

## Progress log 2026-10-03

### en/ko stage 0 result (analyze.md, 270 utts, chance 0.004)

| probe | enc | peak layer | peak raw / centered | L28 |
|---|---|---|---|---|
| en audio <-> ko audio (audio_mean) | 0.02 | L9 | 0.77 / 0.92 | 0.01 |
| en text <-> ko text (text_mean) | - | L12 | 0.98 / 0.99 | 0.05 |
| en audio -> ko text | - | L9 | 0.82 / 1.00 | 0.02 |
| asr_pos (position before first transcript token) | - | L18 | 0.71 / 0.84 | 0.31 |

- Encoder output is NOT language-neutral. z appears in the LLM, L7..L14, and is destroyed
  from ~L19 on where the logit lens starts to read out transcript tokens (en top1 acc L19 0.25
  -> L28 0.94; ko top-1 script flips latin -> hangul at L19).
- Language-ID linear probe = 1.000 at every layer: a language direction coexists with z.
  Per-language mean-centering is enough to get near-perfect retrieval in the middle layers.
- Decision rule says: peak exists -> z exists -> stage 1 (leave-pairs-out fine-tune) is on.

### gen.log (8 utts)

- A: text-only translation works in both directions (prompt format leaks "language None<asr_text>"
  junk but content is fine).
- B: EN audio + "Korean" tag -> 5/8 come out as Korean translations, 3/8 stay English.
  KO audio + "English" tag -> 1/8 English, 7/8 stay Korean. Zero-shot ST signal exists, asymmetric.
- C: system-prompt "Translate the speech into Korean." adds nothing over the tag alone (same 4 outputs).

### Fixes to probe_z.py

- `lang_probe`: StandardScaler + LogisticRegression(C=0.1, max_iter=300); raw lbfgs on 2048-d
  unscaled features hogged 31 cores for minutes. Run analyze with OMP_NUM_THREADS=8 (an.sh).
- logit-lens script labels: hangul/han both printed as "ha" -> now hg/hn (ABBR).
- New subcommand `tagswap --pair SRC-TGT [--sys] --out X.jsonl`: whole-set zero-shot ST probe.
  Reports output-script rate, chrF/BLEU vs target-language FLEURS transcript (parallel corpus),
  chrF vs source transcript (= "still transcribing" score).

### Running now: tmux `probe-z2` (chain2.sh)

1. tagswap en-ko, ko-en, en-zh, en-ja, zh-en, ja-en, ko-ja -> `ts_<pair>.jsonl/.log`; en-ko --sys.
   Marker `tagswap.done`.
2. extract --langs en,ko,ja,zh -> `four.npz` (248 common ids), marker `extract4.done`.
3. analyze -> `analyze4.md`, marker `chain2.done`.

ja_jp / cmn_hans_cn FLEURS test audio downloaded to `~/datasets/fleurs/data/<lang>/audio/test/`
(dl.sh, 650 / 945 files).

### tagswap whole-set results (on-target = output in target script; ja = contains kana, zh = han w/o kana)

| pair | N | on-target rate | chrF on-target vs FLEURS ref | chrF vs source transcript |
|---|---|---|---|---|
| en-ja | 321 | 0.63 | 31.1 | 39.9 |
| en-ko | 270 | 0.40 | 23.7 | 62.3 |
| ja-en | 321 | 0.13 | 22.9 | 77.5 |
| ko-en | 270 | 0.10 | 17.9 | 84.5 |
| en-zh | 349 | 0.014 | 22.8 | 93.0 |
| zh-en | 349 | 0.014 | 19.8 | 82.3 |

en->X works partially, X->en and anything with zh almost never. Tag alone is not a reliable
output-language knob even though z exists.

### analyze: new "common-z tests" section (added 2026-10-03, runs in analyze4.md)

- pairwise mean acc@1 over all language pairs: raw / per-language centered / language subspace
  removed (per-language mean subtracted AND span of (lang mean - global mean) projected out).
- joint gallery: query language vs union of all other languages' utterances (3N candidates for
  4 langs). Correct = same utterance in any language. This is the direct "one common z" test:
  residual language bias would pull same-language-different-sentence above cross-language-same-sentence.
- On en/ko the two tests coincide (only one other language); meaningful from four.npz on.

### tagswap with script purity (fraction of letters in the target script; added after seeing
ko-ja outputs code-switch back to Korean mid-sentence — "contains kana" was too lenient)

| pair | fully switched (purity>=0.9) | stayed source (purity<0.1) | chrF of fully switched |
|---|---|---|---|
| ko-ja | 0.55 | 0.004 | 28.1 |
| en-ja | 0.57 | 0.38 | 32.3 |
| en-ko | 0.34 | 0.63 | 27.4 |
| ja-en | 0.05 | 0.86 | 45.5 |
| ko-en | 0.03 | 0.87 | 58.4 |
| en-zh | 0.011 | 0.99 | 28.8 |
| zh-en | 0.003 | 0.94 | 50.9 |

Reading: X->en almost never switches but the few that do are good translations; en->ja/ko switch
often but half of ko-ja outputs drift back to Korean mid-sentence. zh in either direction is dead.
`tssum --pair P --jsonl ts_P.jsonl` recomputes any of these from the saved outputs.

### 4-language result (analyze4.md, four.npz, 248 utts x en/ko/ja/zh)

All six pairs peak at L9 on audio_mean. Pairwise raw / centered at L9: en-ko 0.78/0.94,
en-ja 0.62/0.87, en-zh 0.77/0.89, ko-ja 0.75/0.81, ko-zh 0.72/0.81, ja-zh 0.56/0.70.
Joint gallery (query language vs union of the other three, 744 candidates), language
subspace removed, L9: en 0.99, ko 0.95, ja 0.89, zh 0.90. => one common z shared by all
four languages, in the same layer band (L7..L14), not pairwise alignments.
Encoder output row stays at 0.01..0.06 for every pair.

### Activation steering (steer subcommand) — the "z + language offset" test

Hook on thinker.model.layers[L-1] output; during prefill add alpha * (mean_tgt - mean_src)
of audio_mean[L] (from four.npz) at the audio-token positions; generate with target tag.
Smoke ko-en L9 alpha=1 (5 utts, enko.npz means): 5/5 pure English, chrF 39.5 (tag alone: 0.03).
Sweep running in tmux `probe-z-steer` (steer1.sh): ko-en layers 6/9/12/15/18, alpha 0.5/1/2,
where=all, multi-layer 6,9,12 (60 utts each) -> steer1.done; then all 8 other pairs at L9
alpha 1 -> steer2.done. Outputs st_<tag>.jsonl/.log.
en-ko with system prompt "Translate..." (ts_en-ko_sys): on-target 0.18, worse than tag alone (0.40).

### steer sweep, ko-en, 60 utts (st_ko-en_*.log)

| setting | fully switched (purity>=0.9) | chrF of switched |
|---|---|---|
| tag only (tagswap, 270) | 0.03 | - |
| L6 a=1 | 0.97 | 43.4 |
| L9 a=1 | 0.97 | 42.3 |
| L12 a=1 | 1.00 | 46.6 |
| L15 a=1 | 0.98 | 48.4 |
| L18 a=1 | 0.93 (5% stayed ko) | 43.5 |
| L9 a=0.5 | 0.12 | - |
| L9 a=2 | 0.98 | 37.9 |
| L9 a=1 where=all | 0.98 | 42.3 |
| L6+L9+L12 a=1 | 1.00 | 33.4 |

Reading: adding the mean language offset once, at a single layer in L6..L15 with alpha=1,
flips the output language on nearly every utterance; the content survives best at L12..L15.
Half the offset does not cross the threshold; double or stacked offsets flip the language but
damage the content. Steering prompt positions too adds nothing.

### steer all pairs, L9 alpha=1, 60 utts (st_<pair>_L9_a1.log)

| pair | tag only | steer fully switched | chrF switched |
|---|---|---|---|
| ko-en | 0.03 | 0.97 | 42.3 |
| ja-en | 0.05 | 1.00 | 38.9 |
| zh-en | 0.003 | 0.83 | 50.2 |
| en-ko | 0.34 | 0.42 | 23.0 |
| en-ja | 0.57 | 0.55 | 28.4 |
| en-zh | 0.011 | 0.38 | 25.3 |
| ko-ja | 0.55 | 0.60 | 28.0 |
| zh-ko | - | 0.28 | 24.7 |
| ja-ko | - | 0.53 | 20.0 |

Target=English: near-perfect flip. Target=non-English: 0.3..0.6, i.e. the mean offset alone is
not enough to leave the English-like default of the residual stream. Next: L15 for all pairs
(steer3.sh, running), then alpha 1.5 / 2 at L15 for non-English targets (steer4.sh).

### steer L15, all pairs + alpha sweep for non-English targets, 60 utts (st_<pair>_L15_a<α>.log)

| pair | L9 a1 | L15 a1 | L15 a1.5 | L15 a2 |
|---|---|---|---|---|
| ko-en | 0.97 / 42 | 0.98 / 48 | | |
| ja-en | 1.00 / 39 | 1.00 / 41 | | |
| zh-en | 0.83 / 50 | 0.90 / 52 | | |
| en-ko | 0.42 / 23 | 0.53 / 31 | 0.52 / 29 | 0.52 / 27 |
| en-zh | 0.38 / 25 | 0.52 / 31 | 0.58 / 29 | 0.52 / 32 |
| en-ja | 0.55 / 28 | 0.75 / 33 | 0.78 / 29 | 0.72 / 29 |
| ko-ja | 0.60 / 28 | 0.58 / 30 | 0.73 / 29 | 0.73 / 29 |
| zh-ko | 0.28 / 25 | 0.38 / 22 | | |
| ja-ko | 0.53 / 20 | 0.57 / 21 | | |
(cells: fully-switched rate / chrF of switched)

L15 > L9 everywhere. For non-English targets more alpha does NOT raise the switch rate
(en-ko/en-zh flat at ~0.52 from a=1 to a=2, chrF drops): the failures are not a magnitude
threshold at the audio positions. Japanese target is the exception (a=1.5 helps, +0.15).
Next (steer5.sh, running): add the offset at every decode step too (`--decode`, where=all)
for en-ko, en-zh, ko-ja at L15, and en-ko at L9.

### decode-step steering (st_*_dec.log; offset added at audio+prompt positions in prefill AND at
every generated token, L15 a=1)

| pair | audio-only L15 a1 | +decode | stayed source | chrF switched |
|---|---|---|---|---|
| en-ko | 0.53 | 0.45 | 0.35 -> 0.02 | 31 -> 24 |
| en-zh | 0.52 | 0.45 | 0.33 -> 0.00 | 31 -> 21 |
| ko-ja | 0.58 | 0.63 | 0 -> 0 | 30 -> 28 |
| en-ko (L9) | 0.42 | 0.25 | 0.47 -> 0.23 | 23 -> 14 |

Decode-step bias kills "stayed in source" but turns it into token-level code-switching
("he did not set a figure for the 쿠츠, saying '그들은 중국의 ..."): script flips, no
translation planning. So the offset is an output-language bias, not a translation trigger.

## Stage 0 conclusion (2026-10-03)

1. z exists: one language-shared content representation for en/ko/ja/zh audio in LLM layers
   L7..L15 (peak L9 for retrieval), absent at the encoder output, destroyed by L19+ where the
   model starts spelling tokens. Language identity is a separate, always-decodable direction.
2. z sits close to English. Removing the language offset (X->en) is enough to make the frozen
   model translate into English (0.9..1.0 switch, chrF 41..52 on FLEURS). Adding an offset
   (en->X, X->Y) only works on half the utterances regardless of magnitude, and forcing it at
   decode time gives code-switched output. The missing piece is "plan the sentence in the target
   language", which the frozen model does not do reliably for non-English targets.
3. Implication for stage 1 (leave-pairs-out fine-tune with a target-language tag): expect
   X->en to come almost free; the training signal must teach the non-English-target planning,
   and the leave-pairs-out test should be designed around non-English targets (e.g. train
   {en,ko,ja,zh}->en and en->{ko,ja,zh}, hold out ko->ja / zh->ko / ja->ko) — those are exactly
   the pairs the offset trick could not solve, so success there is not explainable by z alone.
   Use L12..L15 if a layer has to be picked for any intermediate loss or adapter.

Artifacts: probe_z.py (extract/analyze/gen/tagswap/tssum/steer), enko.npz, four.npz,
analyze.md, analyze4.md, gen.log, ts_*.jsonl/.log, st_*.jsonl/.log.
A copy without the npz files is at ~/probe_z_2026-10-03/.

## Alternatives to the mean offset (2026-10-03, en-ko, 60 utts; alt_*.log)

Tag-only baseline 0.34 / audio-position mean offset L15 a1: 0.53 / chrF 31.

### exp2: shift the <asr_text> position instead (asr_pos vectors, `--pos asr`)

| layer | a | fully switched | chrF |
|---|---|---|---|
| L12 | 1 | 0.35 | 33 |
| L15 | 1 | 0.38 | 31 |
| L18 | 1 | 0.48 | 32 |
| L18 | 2 | 0.53 | 33 |
| L21 | 1 | **0.62** | **34** |
| audio L15 + asr L15 | 1 | 0.48 | 31 |

The output-language decision lives late and at the planning position: shifting the
<asr_text> position at L21 beats anything done on the audio span. L24/L27, a=1.5, L21+L24,
L21+decode, audio L15 + asr L21 are queued (alt3.sh).

### exp3: LDA probe direction instead of mean difference (`--vec probe`)

cos(probe dir, mean diff) = 0.49 at L9, 0.36 at L15, 0.55 at asr L18.
| setting | a=1 | a=2 | a=3 |
|---|---|---|---|
| audio L9 | 0.37 | 0.42 | 0.32 |
| audio L15 | 0.38 | 0.42 | 0.37 |
| asr L18 a=2 | 0.48 | | |
Never better than the mean difference at the same place. Dropped.

### exp1: language-specific FFN neurons (`neurons-extract` / `neurons-steer`)

Stats: per layer x neuron, P(act_fn(gate)>0) and mean activation, per language, over audio
positions and over teacher-forced transcript positions ("trans"). Specific-to-L score =
p_L - max_other p. Intervention: zero the top-k source-specific, set top-k target-specific
to the target mean (x gain), at every position incl. decode. Smoke (3 utts, en/ko stats):
en-specific neurons concentrate in L18..L27, ko-specific in L15..L21.
Full run blocked at 16:58 by another job's vLLM engine (17.7 GB on the shared 4090);
alt3.sh waits for >= 8 GB free and then runs extract + 7 neuron settings + the exp2 follow-ups.

### Paused 2026-10-03 17:05 (user call: GPU busy, resume later)

alt3.sh was stopped before it ran anything. To resume, with >= 8 GB GPU free:

    tmux new-session -d -s probe-z-alt3 -c /home/skkai/STiTy "bash <this dir>/alt3.sh"

It does, in order: neurons-extract (4 langs, ~5 min) -> 7 neuron-steer settings on en-ko ->
asr-position steering L24 / L27 / L21 a=1.5 / L21+L24 / L21+decode / audio L15 + asr L21.
Each step waits by itself until the GPU has 8 GB free (alt3_wait.log shows the waiting).
Needs four.npz (in this scratchpad dir only; rerun `extract --langs en,ko,ja,zh` if gone).

### Resumed 2026-10-04 08:40 (alt3.sh). exp1 neurons + exp2 deeper asr position, en-ko, 60 utts

exp1 language-specific neurons (alt_neur_*.log). en-specific neurons sit in L18..L27, ko-specific
in L0 (embedding-adjacent, 174..215 of top-1000) and L19..L27.

| region | mode | top-k | gain | fully switched | stayed source | chrF |
|---|---|---|---|---|---|---|
| trans | both | 200 | 1 | 0.32 | 0.43 | 29.6 |
| trans | both | 1000 | 1 | 0.53 | 0.17 | 24.3 |
| trans | both | 3000 | 1 | 0.37 | 0.12 | 22.2 |
| trans | off | 1000 | 1 | 0.50 | 0.33 | 31.9 |
| trans | on | 1000 | 1 | 0.42 | 0.43 | 26.8 |
| trans | both | 1000 | 2 | 0.40 | 0.05 | 22.9 |
| audio | both | 1000 | 1 | 0.15 | 0.68 | 28.3 |

Best 0.53 = same switch rate as the audio-position mean offset, with worse content (chrF 24
vs 31). Turning neurons off alone keeps content (chrF 32) but switches only half. More
neurons or more gain removes "stayed English" but produces mixed/broken output. Dropped as a
steering method; useful only as a map of where language identity is written (late layers).

exp2 follow-up (asr position): L24 a=1 -> **0.80 switched / 0.13 stayed / chrF 30.2**, the
best en-ko result of everything tried. Outputs are real translations (literal, some
transliteration). L27 / L21 a1.5 / L21+L24 / L21+decode / audio L15 + asr L21: see below.

exp2 follow-up results (asr position, en-ko, 60 utts):

| setting | fully switched | stayed source | chrF |
|---|---|---|---|
| asr L21 a1 | 0.62 | 0.32 | 34.4 |
| asr L21 a1.5 | 0.55 | 0.38 | 31.1 |
| **asr L24 a1** | **0.80** | 0.13 | 30.2 |
| asr L27 a1 | 0.32 | 0.58 | 30.5 |
| asr L21+L24 a1 | 0.78 | 0.13 | 30.6 |
| asr L21 a1 + decode | 0.75 | 0.02 | 25.7 |
| audio L15 + asr L21 a1 | 0.57 | 0.28 | 29.5 |

The lever is the layer, not the magnitude: L24 at the <asr_text> position is the sweet spot
(L27 is past the point where the language decision is made; a=1.5 hurts). Stacking layers or
positions adds nothing; decode-step bias trades content for script.

Summary of the three alternatives: asr-position shift (exp2) is the only one that beat the
audio-position mean offset (0.80 vs 0.53 on en-ko). Probe direction (exp3) and language-specific
neurons (exp1) did not. alt4.sh (running) checks asr L24 a1 on en-zh, en-ja, ko-ja, zh-ko,
ja-ko, ko-en.

### asr L24 a1 on other pairs (alt4_*.log, 60 utts) vs audio L15 a1

| pair | audio L15 | asr L24 | asr L24 chrF |
|---|---|---|---|
| en-ko | 0.53 | **0.80** | 30 |
| en-ja | 0.75 | **0.88** | 34 |
| zh-ko | 0.38 | **0.62** | 23 |
| ko-ja | 0.58 | 0.60 | 31 |
| ja-ko | 0.57 | 0.42 | 20 |
| en-zh | 0.52 | 0.33 | 33 |
| ko-en | 0.98 | 0.72 | 48 |

No single recipe: asr L24 is the best lever when the SOURCE is English (en-ko, en-ja) and for
zh-ko; audio L15 stays best for X->en and en-zh. Both positions matter and which dominates
depends on the pair. For a trained adapter this argues for letting the model learn where to
inject the target-language signal rather than fixing one layer/position by hand.

## Final state 2026-10-04 09:10

All chains finished, no probe tmux sessions left, GPU released. Copy (with four.npz) at
~/probe_z_2026-10-03/. Open items: none running. Possible next steps, user's call:
- stage 1 fine-tune (leave-pairs-out, non-English-target holdout) — the real test.
- if more probing wanted: asr-position layer sweep per pair (L18..L26), or combined
  audio L15 + asr L24 with per-pair alpha, on the full 248-utt set instead of 60.

## Stage 1 smoke: small-data LoRA fine-tune (2026-10-04, ~/probe_z_2026-10-03/ft_smoke/)

Setup: FLEURS train, 300 sentences per pair, 13 pairs (9 ST + 4 ASR), holdout zh-ko / ja-ko / zh-ja
never seen. Tag = target language (`language Korean<asr_text>…` on any audio). LoRA r=64 on
q/k/v/o of the whole model (35M params, 1.7%), embeddings/lm_head frozen (`--no_seg 1`),
1 epoch, 244 steps, batch 2 x grad_acc 8, lr 1e-4, ~6 min on the 4090 (batch 4 OOMs on long
utterances). eval_loss 0.95 -> 0.94 (flat after step 60: data-limited). Script:
Qwen3-ASR/finetuning/qwen3_asr_sft.py (+ --no_seg, --lora_targets, final/ save; uncommitted),
data builder ft_smoke/build_jsonl.py, chain ft_smoke/ft_run.sh, adapter ft_smoke/out_smoke/final.

Eval: FLEURS test, 60 utts per pair, tag only, no steering (`PROBE_ADAPTER=... probe_z.py tagswap`).

| pair | seen in training? | before (tag only) | after: fully switched | after: chrF |
|---|---|---|---|---|
| ko-en | yes | 0.03 | 1.00 | 53.0 |
| en-ko | yes | 0.34 | 0.95 | 31.0 |
| en-ja | yes | 0.57 | 0.95 | 37.5 |
| zh-ko | **no** | - | 0.92 | 26.7 |
| ja-ko | **no** | - | 0.97 | 28.4 |
| zh-ja | **no** | - | 0.98 | 30.5 |
| ko-ko (ASR) | yes | chrF 89.1 | - | 89.6 |
| en-en (ASR) | yes | chrF 93.7 | - | 94.2 |

Conclusion: with 3.9k samples the tag is re-learned as "output language" and it transfers to
pairs never trained — same switch rate as seen pairs. ASR does not regress. Quality (chrF
27-53) is where the data size shows; ja-ko errors are mostly ASR errors propagating. This is
the leave-pairs-out result the design needed: N languages need ~2N pairs of data, not N^2.
Next: scale data (CoVoST2 / all FLEURS train, all recordings), more epochs, compare LoRA
targets (+MLP) and rank; then streaming (stage 2).

### COMET (Unbabel/wmt22-comet-da, reference-based, all 60 hyps incl. wrong-language ones)
`.venv/bin/python comet_eval.py --n 60 LABEL=file.jsonl ...` (comet lives in .venv, not the stity env)

| pair | tag only | steer audio L15 | steer asr L24 | fine-tuned (tag only) |
|---|---|---|---|---|
| ko-en | 0.675 | 0.811 | - | **0.853** |
| en-ko | 0.649 | 0.731 | 0.790 | **0.869** |
| en-ja | 0.720 | 0.780 | 0.805 | **0.887** |
| zh-ko (holdout) | - | 0.723 | 0.734 | **0.851** |
| ja-ko (holdout) | - | 0.689 | - | **0.829** |
| zh-ja (holdout) | - | - | - | **0.886** |
| ko-ja | 0.824* | - | - | - |

Caveat: COMET gives partial credit to a source-language copy (tag-only rows are mostly
transcriptions yet score 0.65-0.72; *ko-ja tag-only is half code-switched and still 0.82),
so read COMET together with the purity rate. Fine-tuned rows are both pure (0.92-1.00) and
0.83-0.89 COMET, holdout pairs included.

## Minimal-pair experiments (2026-10-04, ft_smoke/ft_min.sh): how little cross-lingual data is enough?

Same recipe as the 13-pair smoke (300 sentences per pair, LoRA r=64, 1 epoch), but only
7 pairs = ASR x4 + 3 translation pairs. B = ASR + en->{ko,ja,zh}. C = ASR + {ko,ja,zh}->en.
Eval FLEURS test, 60 utts, tag only. "seen" = pair in training; everything else unseen.

| pair | A: 13 pairs | B: ASR + en->X | C: ASR + X->en |
|---|---|---|---|
| en-ko | 0.95 / 31.0 / 0.869 (seen) | 0.88 / 31.4 / 0.866 (seen) | 0.82 / 30.2 / 0.829 |
| ko-en | 1.00 / 53.0 / 0.853 (seen) | 1.00 / 53.1 / 0.848 | 1.00 / 53.5 / 0.854 (seen) |
| zh-ko | 0.92 / 26.7 / 0.851 | 0.92 / 26.6 / 0.849 | 0.85 / 26.4 / 0.855 |
| ja-ko | 0.97 / 28.4 / 0.829 | 0.88 / 24.6 / 0.778 | 0.87 / 24.9 / 0.786 |
| zh-ja | 0.98 / 30.5 / 0.886 | 0.98 / 28.5 / 0.881 | 0.93 / 26.8 / 0.873 |
(cells: fully-switched rate / chrF of switched / COMET of all 60)

Reading: three translation pairs in one direction are enough to open every other direction.
B (en->X only) is as good as the full 13-pair set on X->Y and on the never-trained X->en.
C (X->en only) also opens en->X and X->Y, slightly weaker (0.82-0.93 switch) — it never saw a
non-English target, so "write in Korean/Japanese" came entirely from ASR data plus the tag.
Practical consequence: the big public X->en corpora (CoVoST2) alone would be a valid training
set for all directions; adding a little en->X (B) is the cheapest way to tighten non-English
targets. Residual weak spot in every run: ja-ko (ASR errors on Japanese propagate).

## Full-FLEURS run (2026-10-04, ft_smoke/ft_big.sh, ft_big2.sh)

Data: every FLEURS train recording, 13 pairs (holdout zh-ko / ja-ko / zh-ja kept), 31,608
samples (2.1k-3.2k per pair), val 520. Same LoRA (r=64, q/k/v/o).
- v1: lr 1e-4, 2 epochs planned. eval_loss 0.951 -> 0.944 (step 600) -> 0.958 -> 0.972 -> 0.993:
  overfits from epoch ~0.4 — each target sentence recurs ~5x per epoch (3 sources x ~1.7
  recordings). Stopped at step 1500; best checkpoint-600 kept.
- v2: lr 3e-5, dropout 0.1, 1 epoch (1,976 steps, ~50 min). eval_loss 0.936 -> 0.922, flat from
  step 1200. Lowest loss of all runs (smoke 0.938).

Eval on the full FLEURS test common set (249-350 utts per pair), tag only:

| pair | smoke (3.9k, 60 utts) | v1 ckpt-600 | v2 |
|---|---|---|---|
| ko-en | 1.00 / 53.0 | 1.00 / 52.1 | 1.00 / 53.2 |
| en-ko | 0.95 / 31.0 | 0.95 / 30.2 | 0.93 / 31.5 |
| en-ja | 0.95 / 37.5 | 0.98 / 35.6 | 0.98 / 36.7 |
| zh-ko (holdout) | 0.92 / 26.7 | 0.94 / 24.9 | 0.91 / 25.2 |
| ja-ko (holdout) | 0.97 / 28.4 | 0.94 / 24.9 | 0.90 / 25.1 |
| zh-ja (holdout) | 0.98 / 30.5 | 0.98 / 29.5 | 0.97 / 29.9 |
| ko-ko ASR chrF | 89.6 (60) | 88.0 | 87.9 |
| en-en ASR chrF | 94.2 (60) | 93.8 | 94.2 |
(switch rate / chrF of switched; smoke column is on 60 utts, others on the full set)

Reading: 8x more samples changed nothing. The sentence inventory is the same ~1.3k per pair;
extra recordings add speakers, not text. Translation quality is bounded by text diversity,
and FLEURS cannot supply more. Scaling has to come from a corpus with new sentences
(CoVoST2: en->X 100k+, X->en tens of thousands). A single epoch at lr 3e-5 is the safe recipe
for this data; lr 1e-4 memorizes within half an epoch.

Same three adapters on the full test set, with COMET (wmt22-da, all hyps) and the base model's
ASR on the same set (ko-ko 88.1, en-en 93.9 => no ASR regression in any run):

| pair | smoke (3.9k samples) | v1 ckpt-600 (31.6k, lr 1e-4) | v2 (31.6k, lr 3e-5) |
|---|---|---|---|
| ko-en | 1.00 / 52.6 / 0.855 | 1.00 / 52.1 / 0.854 | 1.00 / 53.2 / 0.857 |
| en-ko | 0.94 / 30.8 / 0.877 | 0.95 / 30.2 / 0.875 | 0.93 / 31.5 / 0.882 |
| en-ja | 0.96 / 37.5 / 0.897 | 0.98 / 35.6 / 0.895 | 0.98 / 36.7 / 0.901 |
| zh-ko (holdout) | 0.95 / 25.3 / 0.855 | 0.94 / 24.9 / 0.850 | 0.91 / 25.2 / 0.850 |
| ja-ko (holdout) | 0.96 / 26.2 / 0.824 | 0.94 / 24.9 / 0.814 | 0.90 / 25.1 / 0.806 |
| zh-ja (holdout) | 0.97 / 30.1 / 0.889 | 0.98 / 29.5 / 0.885 | 0.97 / 29.9 / 0.887 |
(switch / chrF / COMET)

Verdict: within noise across all three. 3.9k samples already extract everything FLEURS'
~1.3k sentences per pair can teach. Next scaling step must bring new sentences (CoVoST2).
Adapters: ft_smoke/out_smoke/final, ft_smoke/out_big/checkpoint-600, ft_smoke/out_big2/final.

### Experiment D (2026-10-04): does ja/zh target data transfer to Korean? ASR x4 + en->ja + en->zh,
no Korean target anywhere, 1,800 samples (ft_smoke/ft_D.sh). 60 utts, switch / chrF / COMET.

| pair | B: ASR + en->{ko,ja,zh} | C: ASR + X->en | D: ASR + en->{ja,zh} |
|---|---|---|---|
| en-ko | 0.88 / 31.4 / 0.866 | 0.82 / 30.2 / 0.829 | 0.92 / 32.2 / 0.866 |
| zh-ko | 0.92 / 26.6 / 0.849 | 0.85 / 26.4 / 0.855 | 0.88 / 26.7 / 0.852 |
| ja-ko | 0.88 / 24.6 / 0.778 | 0.87 / 24.9 / 0.786 | 0.90 / 23.5 / 0.775 |
| ko-en | 1.00 / 53.1 / 0.848 | 1.00 / 53.5 / 0.854 | 1.00 / 53.9 / 0.853 |
| en-ja | - | - | 0.98 / 36.6 / 0.885 |

D == B on every Korean-target pair: removing the 300 en->ko samples changed nothing. The
"assemble in language X" skill is shared across targets; Korean wording comes from Korean ASR
data alone. Consequence for scaling: CoVoST2 en->ja/zh (289k) should lift Korean targets as
much as it lifts ja/zh at the current data regime; Korean-specific translation data is a
second-order refinement, not a prerequisite.

v2 adapter, remaining X->en on the full test set: ja-en 1.00 / chrF 50.1 / COMET 0.839,
zh-en 1.00 / 55.4 / 0.857. The zh-en weakness seen with steering (0.83-0.90) is gone after training.

### v2 adapter, full 4x4 matrix on the FLEURS test common set (switch / chrF / COMET)

| src \ tgt | en | ko | ja | zh |
|---|---|---|---|---|
| en | ASR 94.2 | 0.93 / 31.5 / 0.882 | 0.98 / 36.7 / 0.901 | 0.88 / 34.8 / 0.875 |
| ko | 1.00 / 53.2 / 0.857 | ASR 87.9 | 0.99 / 32.4 / 0.878 | 0.90 / 26.2 / 0.847 |
| ja | 1.00 / 50.1 / 0.839 | 0.90 / 25.1 / 0.806 (holdout) | ASR | 0.87 / 24.4 / 0.842 |
| zh | 1.00 / 55.4 / 0.857 | 0.91 / 25.2 / 0.850 (holdout) | 0.97 / 29.9 / 0.887 (holdout) | ASR |

Switch rate clusters by TARGET: en 1.00, ja 0.97-0.99, ko 0.90-0.93, zh 0.87-0.90 — the
source language barely moves it. COMET clusters the same way with a source effect on top:
ja as source is lowest everywhere (ASR errors). So "how well the model assembles language X"
is a per-target property; "how clean z is" is a per-source property.

## SEG model questions (2026-10-04, ft_smoke/seg1-3.sh; SEG model =
models/Qwen3-ASR-1.7B-en-covost2-dailytalk-mix-c200-merged, English <SEG> trained on CoVoST2+DailyTalk)

### Q1: is z the same in the SEG model? Yes. four_seg.npz / analyze_seg.md vs analyze4.md:
pairwise subspace-removed acc@1 L9 0.85 vs 0.87, L15 0.79 vs 0.76, joint gallery L9 en 0.98/ko 0.95/
ja 0.88/zh 0.90 vs 0.99/0.95/0.89/0.90. Every layer within ±0.02. SEG fine-tuning changed output
behaviour only, not the shared content representation.

### Q3 first attempt (seg1.sh): translation LoRA (FLEURS 13 pairs, no <SEG> anywhere in targets)
on top of the SEG model. Result: <SEG> disappears even in English ASR (en-en concatenated
pairs: 0.03 SEG per output vs 3.1 before). Confounded — the adapter unlearned SEG because no
target contained it. Fix in seg3.sh: pseudo-label the en-en rows with the SEG model's own
transcripts (keeps <SEG>), translation targets still without <SEG>, retrain, then test whether
<SEG> shows up in ko/ja outputs (en source) and in en outputs from ko source (SEG never seen
with Korean audio or Korean text).

### Q3 corrected (seg3.sh): SEG supervised only in English ASR targets (pseudo-labelled by the SEG
model, 2.0 <SEG>/utt), translation targets without <SEG>. Test = 2 concatenated FLEURS test
utterances, 40 groups, count <SEG> in the output.

| output | <SEG> per output | groups with >= 1 |
|---|---|---|
| en-en (ASR) | 2.50 | 0.98 |
| en-ko | 0.20 | 0.18 |
| en-ja | 0.00 | 0.00 |
| ko-en | 0.38 | 0.20 |

Zero-shot transfer of the SEG habit into translation output is weak. But every <SEG> that does
appear sits exactly at the utterance boundary (never mid-clause) — the boundary knowledge is in
z; what is missing is the habit of emitting the token outside English-transcription mode.
Translation quality unchanged (en-ko 0.93 / chrF 32).

### Q2 (segprobe): <SEG> is readable by the logit lens only at L28 (P=0.999; L27 0.01; L26 0.001).
Linear probes separate boundary positions from other positions from L3 on, but the simple
labels (sentence-final period vs mostly commas) make that trivial; the --self variant (model's own
clause-level segmentation as labels, negatives = punctuation it did NOT segment at) is running
in seg4.sh -> segprobe2_gap03.md / segprobe2_gap0.md.

### seg5.sh: sentence-level <SEG> added to TRANSLATION targets (150 concatenated pairs x en-ko,
en-ja, ko-en, en-en, ko-ko; "t1 <SEG> t2"), clause-level SEG still only in English ASR.

| input | output | <SEG>/output | groups >= 1 |
|---|---|---|---|
| en x2 | en ASR | 1.75 | 1.00 |
| en x2 | ko | 1.00 (exactly 1 in 40/40) | 1.00 |
| en x2 | ja | 0.95 | 0.95 |
| zh x2 (never trained with SEG or as a pair) | ko | 0.95 | 0.95 |
| en x1 | en ASR | 1.87 | — clause-level alive |
| en x1 | ko | 0.00 (0 in 60/60) | — clause-level does NOT transfer |

The emission habit transfers across pairs once taught in translation mode (zh-ko 0.95), but
only at the granularity taught. English clause-level segmentation stays in the English output
stream; it is not carried by z into Korean. Consistent with segprobe: <SEG> is decided at L28 on
the emitted token stream. To segment inside the target language, the targets need clause-level
<SEG> labels (segment-wise translations joined by <SEG>, as the autoseg pipeline produces).
en-ko translation quality after seg5: 0.85 / chrF 32.4 (purity down from 0.93; concat data
pulls some outputs toward transcription).

### Q2 final (segprobe5_comma.md, 175 concatenated pairs, SEG model's own segmentation as labels).
Earlier probe versions were confounded (period vs comma tokens; then the bare space token that
precedes <SEG>). Clean comparison = comma positions where the model segments (111) vs comma
positions where it does not (302), features at the comma token itself:

| layer | L0 | L1 | L3 | L6 | L9 | L12 | L15 | L18 | L21 | L24 | L27 | L28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comma probe AUC | 0.49 | 0.78 | 0.83 | 0.85 | 0.90 | 0.88 | 0.93 | 0.95 | 0.96 | 0.97 | 0.97 | 0.96 |

The segment-or-not decision is not present in the embedding (chance), builds through the
network, is already 0.90 inside the z band (L9) and saturates ~L21-24; the token is emitted at
L28 (logit lens P(<SEG>) is 0 until L27). So the boundary information lives in the shared
representation; only the emission habit is English-output-specific (Q3).

## SEG summary
1. z is untouched by SEG fine-tuning.
2. Boundary information is in z (comma-probe AUC 0.90 at L9, 0.93 at L15); the <SEG> token is
   written at L28.
3. The emission habit does not transfer zero-shot to non-English output (0.0-0.2 SEG/output),
   and English clause-level granularity does not transfer either (en x1 -> ko: 0 in 60). But
   once SEG is taught in translation mode at some granularity it transfers across pairs
   (zh-ko 0.95 without any zh-ko training). Target-language clause-level SEG therefore needs
   clause-level labels in target text (segment-wise translations joined by <SEG>); the
   information to support it is already in z.
Adapters: ft_smoke/out_segft2 (SEG in en ASR only), out_segft5 (+ sentence-level SEG in translation).

### seg9.sh: clause-level <SEG> in en->ko translation targets ONLY (segment-wise Hy-MT2-1.8B
translations of the SEG model's English segmentation, 300 utts / 609 segments; ft_smoke/segtrans.py).
Everything else as seg3 (SEG only in English ASR). Test: single FLEURS test utterances (k=1), 60 each.

| output | SEG taught for this target? | <SEG>/output | outputs with >= 1 |
|---|---|---|---|
| en-en ASR | yes (clause) | 1.95 | 59/60 |
| en-ko | yes (clause) | 1.28 | 37/60 |
| zh-ko | no (source never seen with SEG) | 1.00 | 30/60 |
| en-ja | no | 0.03 | 1/60 |
| ko-ja | no | 0.00 | 0/60 |
| zh-ja | no | 0.00 | 0/60 |
| ko-en | no (English translation mode) | 0.00 | 0/60 |

Clause-level segmentation transfers across SOURCE languages (zh-ko 1.00 from en-ko training) and
not across TARGET languages (ja: 0). Where it fires, the cut points coincide with the English
segmentation of the same sentence (ids 1661/1662/1663: same clause breaks from en and zh audio);
what varies is whether it fires (37 vs 30 of 60, both 22). ko-en 0 although en-en ASR has SEG:
the habit is attached to (target language x mode), not to English text.
=> labels are needed per TARGET language (one source is enough per target), not per pair, and
they can be generated automatically (segment source with the SEG model, translate each segment
with a local MT, join with <SEG>).
k=2 after seg9: en-ko 2.95 SEG/output (40/40 >= 2), zh-ko 2.75 (40/40 >= 2), en-ja 0.10. On
two-sentence inputs the Korean clause segmentation fires every time, from English or Chinese audio.
en-ko translation quality after seg9 (tagswap, 60): chrF 31.9; the purity>=0.9 rate reads 0.52 only because the
literal "<SEG>" letters count as Latin in the purity metric (29/60 outputs contain <SEG>); the
Korean text itself is clean. Quality unchanged vs seg3 (chrF 32.0).
Adapter: ft_smoke/out_segft9/final. Data: ft_smoke/train_seg9.jsonl (segmt rows carry src_segs).

### seg10.sh: register follows the target-side labels (ft_smoke/segtrans_interp.py, register.py)
Labels: gemma-3-4b-it with a simultaneous-interpreter prompt (sees only preceding interpreted
segments, never the future, must not complete fragments), en->ko clause-SEG targets in two
registers: 반말 (A) and 합니다체 (B). Everything else as seg9. Register metric = sentence-ending
class of each output piece. 60 single utterances per pair.

| adapter | en-ko 반말 / 합니다체 | zh-ko 반말 / 합니다체 | ja-ko 반말 / 합니다체 | SEG/output en-ko, zh-ko, ja-ko |
|---|---|---|---|---|
| seg9 (Hy-MT2 default) | 0.24 / 0.62 | 0.30 / 0.66 | - | 1.28, 1.00, - |
| A 반말 | **0.82 / 0.00** | **0.89 / 0.00** | 0.65 / 0.22 | 1.67, 1.52, 0.57 |
| B 합니다체 | 0.17 / 0.72 | 0.25 / 0.72 | 0.14 / 0.77 | 1.32, 1.07, 0.58 |

The register taught on English-source Korean targets is reproduced for Chinese and Japanese
sources that never had Korean targets of that style: the assembler carries register with the
target language, not with the pair. (chrF against FLEURS references drops for 반말 — 27 vs 33 —
because the references are 합니다체/평서체; COMET is the fair comparison, see line above.)
Interpreter-style prefix-context labels also gave a higher SEG firing rate (1.67 vs 1.28).
Adapters: ft_smoke/out_segft10_banmal/final, out_segft10_hapnida/final.
COMET (wmt22-da, en-ko 60, tag only): seg3 no-SEG targets 0.877 | seg9 Hy-MT2 segment targets 0.820 |
seg10 합니다체 0.815 | seg10 반말 0.761. Segment-wise (future-blind) targets cost ~0.06 COMET vs
whole-sentence targets — the price of streaming-shaped labels; 반말 loses a further 0.05 partly
because the FLEURS references are formal. Quality of segment-wise labels is the next lever
(better interpreter model, or sentence-level targets with projected <SEG>).

## Plan: target-language-specific segmentation via reward (2026-10-04, user go-ahead for A->B->C)

Everything local, API cost 0: SEG model, translation adapters (ft_smoke/out_segft9|10), Hy-MT2 /
gemma-3-4b-it for labels, Qwen3-ForcedAligner-0.6B (cached, `qwen_asr.Qwen3ForcedAligner.align(audio,
text, language)` -> items with .text/.start_time/.end_time), COMET-DA + CometKiwi in .venv, FLEURS.

A. Reward = per-target "is this segmentation good". For utterance u and segmentation s (subset of
   candidate boundaries = the SEG model's English <SEG> positions, timed by the forced aligner):
   segment i translation = translation adapter run on audio[0:t_i] with the already committed
   target text as forced prefix (future-blind, streaming-faithful); R(s) = COMET(concat, ref) -
   lambda * latency (mean segment duration). Evaluate s in {none, all, each single boundary} ->
   per target: which English boundaries survive. Script ft_smoke/segreward.py (+ _score.py in .venv).
   Output answers "fraction of English boundaries invalid for ko/ja/zh" and doubles as option-1 labels.
B. Policy = SEG model in ASR mode (source-side <SEG>), conditioned on a target-language tag;
   GRPO: G=8 sampled segmentations per utterance, reward from A, LoRA update on transcript
   positions. Guard reward hacking with lambda and a max segment length.
C. Distil: run the learned segmentation -> prefix translations -> "t1 <SEG> t2" targets -> SFT the
   translation adapter (as seg9/seg10). End-to-end RL on translation tokens only after that.
Stage A/B code (ft_smoke/): segreward.py (+ segreward_score.py, .venv) — running as srA.sh for ko/ja/zh,
100 test utts each, outputs srA_<tgt>.jsonl/.scores.json. comet_server.py (.venv, HTTP :8777,
models da|kiwi). grpo_seg.py — policy = SEG model + LoRA r32 in ASR mode with context
"Segment for Korean interpretation.", translator = SEG model + out_segft9 adapter (frozen),
reward = COMET-DA(prefix-translated concat) - lam*mean_seg_dur - pen*[max seg > max_seg],
GRPO advantage within G samples, KL to adapter-disabled base. Not yet smoke-tested (needs GPU).
Smoke: PROBE_Z_DIR=<probe_z dir> PYTHONPATH=Qwen3-ASR python grpo_seg.py --seg_model models/…seg-merged
  --trans_adapter ft_smoke/out_segft9/final --utts 4 --G 4 --steps 2 --out ft_smoke/out_grpo_smoke
  (comet_server.py must be running in .venv first).

### Stage A result (srA_<tgt>.scores.json, 100 FLEURS test utts, English boundaries from the SEG
model, prefix-translated with out_segft9, COMET-DA vs FLEURS reference; valid = drop <= 0.02)

| target | boundaries | valid | COMET whole | COMET all boundaries | delta quartiles |
|---|---|---|---|---|---|
| ko | 98 | 0.65 | 0.879 | 0.861 | -0.034 / -0.007 / +0.004 |
| ja | 100 | 0.59 | 0.894 | 0.876 | -0.043 / -0.013 / +0.002 |
| zh | 98 | 0.65 | 0.873 | 0.851 | -0.033 / -0.008 / +0.003 |

Cross-language agreement on the 73 shared boundaries: delta correlation ko-ja 0.13, ko-zh 0.20,
ja-zh 0.34; valid in all three 32%, in none 8%, valid for some targets but not others 60%.
=> English boundaries are not "good or bad", they are good for some targets: a per-target
policy (stage B) is justified. Caveat: single-boundary COMET deltas on 100 utts are noisy.
Stage B status (2026-10-04 19:25): grpo_seg.py fixed (train-split TSV rows, audio-exists filter).
Smoke blocked by another job's vLLM (22 GB) since 18:49; grpo_smoke.sh waits for >= 14 GB free, then
runs; grpo_full.sh (tmux grpo-full) waits for the smoke to print "done" and then runs ko, 200 utts,
G=8, 100 steps -> out_grpo_ko/adapter_step{20,...,100}, log grpo_ko.log (json per step: meanR, meanC,
mean_nseg). Stage C script distill_seg.py written (policy adapter + translator -> en-ko SFT rows),
not run. comet_server.py runs in tmux comet-srv (:8777).

## Stage B+C on the koen-seg-mix base (2026-10-04 evening, machine skkai/mobility)

Base switched to models/Qwen3-ASR-1.7B-koen-seg-mix-merged (en+ko <SEG>, better ASR than the
dailytalk-mix-c200 model). The dailytalk-mix GRPO run was stopped at step 24 (nseg 2.0, COMET 0.86,
nothing wrong with it). Everything below is on the mix base; artefacts in models/zprobe/{adapters,grpo,data,logs}.

Translation adapter on the new base (segft9_mix): train_seg9 rows with the en-en pseudo-labels regenerated by
the mix model (1.93 <SEG>/utt on English), same recipe as seg9 (LoRA r64 q/k/v/o, lr 1e-4, 1 epoch, 243 steps),
eval_loss 0.913. Behaviour matches seg9 on the old base:

| output (ft9mix) | <SEG>/output | fires |
|---|---|---|
| en-ko k=1 | 1.38 | 40/60 |
| en-ko k=2 | 3.08 | 40/40 >= 2 |
| zh-ko k=1 | 1.32 | 35/60 |
| en-ja k=1 | 0.00 | 0/60 |
| en-en ASR | 1.98 | 60/60 |
| ko-ko ASR | **0.33** | 10/60 |

ko-ko: the mix base segments Korean, but train_seg9's 300 ko-ko rows are plain transcripts, so the adapter
unlearns Korean <SEG> (the seg1 confound again, now on the Korean side). For a production adapter on this base
the ko-ko rows need `seglabel --pair ko-ko` too. Irrelevant for stage B (English source).

### Stage B: GRPO, ko target, 200 train utts, G=8, 100 steps, batch 2, lam 0.02, max_seg 6, 33 min on the 4090

| steps | meanR | meanC (COMET of the sampled segmentations) | mean nseg |
|---|---|---|---|
| 1-20 | 0.718 | 0.833 | 2.52 |
| 21-40 | 0.722 | 0.805 | 3.10 |
| 41-60 | 0.720 | 0.799 | 3.16 |
| 61-80 | 0.723 | 0.808 | 3.08 |
| 81-100 | 0.729 | 0.797 | 3.16 |

Within-G reward range 0.16 (there is signal), but the policy moved along the trade-off instead of up:
+0.6 segments per utterance (mean segment 5 s -> 3.3 s, worth +0.034 reward at lam 0.02) paid for by
-0.035 COMET. Reward ends flat. lam 0.02 prices latency high enough that cutting more is as good as cutting
better; nothing pushed the policy toward "drop the boundaries that are invalid for Korean" (stage A's 35%).
Batches are different utterances each step, so meanC per block is noisy (40 utts), but the direction is
consistent over all four later blocks.

### Stage C: distil -> SFT -> compare with ft9mix, plus a control

Three label sets for the same 282 en-ko train utterances (18 of the 300 seg9 ids have no ko row in this
machine's FLEURS ko train.tsv):

| en-ko labels | how | <SEG>/utt | label COMET vs ko ref |
|---|---|---|---|
| Hy-MT2 (seg9 / ft9mix) | mix model's own English segmentation, each segment translated whole by Hy-MT2 | 2.03 | **0.887** |
| control (distillctl) | mix model's own segmentation, prefix translation by segft9_mix (audio prefix + committed Korean) | 1.81 | 0.870 |
| policy (distillmix) | GRPO step100 segmentation, same prefix translation | 2.96 | 0.843 |

7/282 policy rows were dropped as repetition loops ("...그의 캐나다의 상대방과의 대결 기록은..." repeated to
the token limit): trans_gen has no repetition penalty, and the same loops sit inside the stage B reward.
The SFT on the distilled jsonl also failed once with DatasetGenerationError because the distilled rows carried
extra fields (bounds, src_seg) that the other rows lack; build_distill_train.py now writes audio/text/pair only
and skips degenerate rows.

Outputs, FLEURS test, 60 utts, tag only. COMET raw (as in all earlier LOG numbers) and with <SEG> stripped
from the hypothesis (comet_eval.py --strip_seg). The raw score penalises SEG-dense outputs by itself: the
same pair differs by -0.11 raw and -0.04 stripped.

| adapter | en-ko COMET raw / stripped | en-ko k=1 <SEG>/out (fires) | zh-ko COMET raw / stripped | zh-ko <SEG>/out |
|---|---|---|---|---|
| ft9mix | **0.830 / 0.887** | 1.38 (40/60) | 0.805 / 0.861 | 1.32 |
| distillctl (own segmentation, prefix labels) | 0.784 / 0.859 | 1.77 (60/60) | 0.815 / 0.860 | 1.17 |
| distillmix (GRPO policy, prefix labels) | 0.721 / 0.846 | 2.75 (57/60) | 0.789 / 0.857 | 1.73 |

Reading, en-ko stripped: prefix-translation labels cost -0.028 against Hy-MT2 labels at the same segmentation
(control), the GRPO segmentation costs a further -0.013 and doubles the segment count. The output gaps track
the label gaps (-0.017 / -0.027 on the labels), so this is label quality, not a learning problem. zh-ko is
flat across the three (0.857-0.861): the segmentation habit still transfers across source languages and the
Chinese-source quality does not depend on which Korean labels were used. Where the policy adapter fails:
fragments emitted as sentences ("모래. <SEG> 그들은 멈췄습니다.", "타는.", "등등."), one repetition loop in 60,
and a sentence-final period on every piece (inherited from the prefix translator).

Verdict for this round: GRPO at lam 0.02 did not find a Korean-specific boundary policy; it found a denser
one. The distillation path works mechanically end to end (policy -> aligner -> prefix translations -> SFT ->
transfer to zh-ko), but the label generator is the bottleneck before the policy can show anything.
Adapters: models/zprobe/adapters/segft9_mix, models/zprobe/grpo/ko_mix/adapter_step{20..100},
models/zprobe/grpo/sft_distill_komix/final, models/zprobe/grpo/sft_distill_koctl/final.
Data: train_seg9_mix / train_distill_komix(_rows) / train_distill_koctl(_rows). Logs: models/zprobe/logs/
{grpo_ko_mix.log, distill_komix.log, distill_koctl.log, eval_*.log, comet_three_way*.md}.
