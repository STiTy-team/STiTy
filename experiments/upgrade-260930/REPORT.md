# Upgrade 2026-09-30 — what changed, and how to verify it

Written for the agent that runs the experiments on the GPU machine. Everything here can be
checked from this file plus the code. Nothing was run on a GPU yet: the machine this was built
on has a 6 GB laptop GPU, so every model-backed number below is still to be measured.

## 1. Summary

- **Every change is opt-in and switched by one config key.** Fixes live in new component
  variants (`qwen-seg:v2`, `qwen3.5:v2`, `silero:v2`, `cascade:v2`) or in new component kinds
  (`labeler`, `filter`, `enhancement`). The existing components (`qwen-seg`, `qwen3.5`, `silero`,
  `cascade`) are unchanged, so the numbers of the 2026-09-28/29 runs stay the baseline.
- **With every switch off, v2 gives exactly v1's output.** Proven on CPU with scripted decodes
  (8 scenarios, `tests/test_qwen_seg_v2.py::Parity`) and, for VAD, on real audio (northstar, 50
  identical segments). The GPU runs in Tier 0 below must confirm this before any other row is read.
- **Each worklog bug that could be replayed on CPU is reproduced in v1 and gone in v2** with only
  its own switch on: A1, A2, A3, A4, A5, A8, A9, N5, N7 (section 4).
- **VAD is the one change with real-audio evidence already.** On northstar (clean, CPU only),
  the `ramp` tuner cuts the longest speech segment from 89.6 s to 17.5 s and doubles how often a
  segment ends at a sentence end (0.26 → 0.49) without more cuts inside sentences (precision
  0.62 → 0.60). Long segments are the root of the per-window language label bug (B7) — section 6.
- **117 bench runs are planned in 7 tiers** (`plan.tsv`), each against a named base run. Tiers 0–2
  (61 runs, mostly 5–20 minute subsets) answer "does each fix do what it claims"; tiers 3–6 are
  tuning and the combined run. Running cost is $0 (all models local).

## 2. Set up and check the code before any GPU run

```bash
git pull   # or copy the working tree; nothing was committed (user rule)
uv sync --project bench                      # adds onnxruntime (for the smart-turn VAD tuner)
uv run --project bench python -m unittest discover -s tests -t .     # 44 tests, ~1 s, CPU
sha256sum configs/pipelines/asr.qwen-seg-v2-* configs/pipelines/asr.qwen-la-* > /tmp/cfg.sha
uv run --project bench python experiments/upgrade-260930/make_configs.py
sha256sum -c --quiet /tmp/cfg.sha && echo reproducible
#   -> "wrote 89 configs" and "reproducible": the config files are exactly what the
#      generator's VARIANTS table says, so the table is the spec of every experiment
```

CPU checks that need the datasets (`STITY_DATA_ROOT`), no GPU:

```bash
# VAD parity and the VAD table of section 6 (about 10 minutes)
B=asr.qwen-seg-v2-ko+mt.qwen3.5-4b
uv run --project bench python -m bench.vad --dataset northstar_en+ko-en \
  --config ${B}_off --config ${B}_vad2fixed --config ${B}_vad-ramp --config ${B}_vad-turn
#   _off and _vad2fixed must print identical numbers.

# cascade:v2 through the real bench harness with mock parts (real time: about 30 s for 2 items)
#   make a throwaway dataset config with `ids: [en_1660, en_1661]` under fleurs/en_us, then
uv run --project bench python -m bench --config mock-v2 --dataset <that config>
#   -> status ok, the new diagnostics metrics printed; delete the run dir afterwards.
```

The server was also smoke-tested with `mock-v2` (`STITY_STITY__PIPELINE=mock-v2 make server`):
hello → ready → partials → translated finals, clean shutdown.

## 3. What changed

### 3.1 ASR commit logic: `qwen-seg:v2` (`core/components/transcription/qwen3_seg_v2.py`)

Subclass of `qwen-seg`. Each switch is a `transcription:` setting, default `false`.

| Switch | Worklog | What v1 does wrong | What v2 does instead |
|---|---|---|---|
| `no_speech_since_vad_reset` | A1 | The no-speech gate checks VAD speech since the *last commit*; if that commit came before the speech ended, a made-up sentence over later silence passes | Window starts at the later of the last commit and the last VAD flush |
| `keep_held_fragment` | A2 | A held comma fragment (`Needless to say,`) is remembered as committed, so the flush join trims it as a repeat | The held fragment is forgotten until it is actually emitted |
| `strict_boundary_dedup` | A3 | After a SEG reset, a new sentence is dropped if its first word is a *suffix* of the last word (`plata` ends with `a`); dropped commits count as "last commit" | Drop only when the new unit is exactly the tail of the last *emitted* commit |
| `carry_uncommitted_audio` | A4 | Dangling-header reset keeps only the last 2 s of uncommitted audio; a same-chunk VAD flush never decodes the carried audio | Carry all uncommitted audio (cap 30 s); flush decodes carried audio (`[CARRY-DECODE]`) |
| `resync_cursor` | A5, N5 | When a re-decode rewrites or merges committed units, the cursor (a count of SEGs) lands one unit too far — a real unit is never emitted — or finds nothing and wipes the rest | If the committed text is no longer a prefix, re-anchor by text, unit by unit in order (`[CURSOR-RESYNC]`) |
| `loop_guard` | A5 | The model's own loop cut needs 4 identical tokens; `아니, 아니, 아니.` alternates tokens and passes | Loop detection on tokens without punctuation or `<SEG>` (4 repeats of a 1–4 token unit), checked before each SEG/dot callback (`[LOOP-GUARD]`); then the normal repetition reset |
| `carry_on_loop_reset` | N6 | The repetition reset throws the slot's audio away (a real backchannel `응 응 응 응` also triggers it) | Keep the last chunk of audio for the new slot |
| `drop_language_lists` | A8 | `English, Chinese, Japanese, Korean, …` is committed | A commit made only of language names is dropped (`[DROP] rule=language-list`) |
| `clean_partials` | A9 | Partials show `language English …` and lone `l` stubs | Headers and stubs stripped from partials |
| `dedup_dot_carry` | N7 | A dot-switch carry re-commits the sentence just committed when the new commit's trigger is SEG | The dot-suffix dedup applies to any trigger |
| `tail_keep_sec` (float, default 0.1) | N8 | 0.1 s of silence kept before the flush decode | Tunable (the plan tries 0.3) |
| `bias_glossary`, `bias_max_terms`, `bias_recent_commits` | upgrade #3 | No context to Qwen3-ASR | Names/terms from `configs/glossaries/<name>.yml` and/or the last N commits go into Qwen3-ASR's `context` (system prompt) of every new decode window |

### 3.2 New ASR committer: `qwen-la` (`core/components/transcription/qwen3_la.py`) — upgrade #1

LocalAgreement-2 on Qwen3-ASR's streaming hypotheses. A word is committed once two consecutive
decodes agree on it, and only up to the last sentence end (`boundary: clause` also allows
commas) or `<SEG>` inside the agreed words. If `max_pending_words` (12) agreed words wait without
a boundary, they are forced out (`agree-max`). After 30 s of audio in one window
(`max_slot_sec`) everything is emitted and the window restarts with the last sentence as a
prefix. New `commitReason` values: `agree`, `agree-max`. **Not done:** the forced aligner and the
target-side agreement of the MT; see section 7.

### 3.3 Translation: `qwen3.5:v2` (`core/components/translation/local_qwen_3_5_4b/translator_v2.py`)

| Setting | Worklog / report | Effect |
|---|---|---|
| `leak_retry` | model quality (Chinese in Korean output, 18 items) | If >15% of the reply's letters are in a foreign script, ask once more with "write only in <target>"; keep the cleaner reply (`[TRANS-LEAK]`) |
| `meta_guard` | seen in the web demo | If the reply answers the instruction (`Please translate…`), retry with a plain prompt; if it happens again, return empty (`[TRANS-META]`) |
| `style: {ko: "..."}` | N11 register (`김 씨` for `김 부장님`) | Extra instruction per target language |
| `glossary: <name>` | upgrade #3 | Pairs whose source form appears in the line go into the prompt |
| `continuation` | upgrade #2 (partial) | A commit that does not end a sentence is translated as the continuation of the translation already on screen for that sentence |

### 3.4 VAD: `silero:v2` (`core/components/vad/silero_v2.py`, `tuners.py`) — the dynamic VAD module

Same Silero model and the same state machine as `VADIterator`, but a **tuner** decides the
speech threshold and the silence needed to end a segment, window by window:

| Tuner | How it decides | Settings |
|---|---|---|
| `fixed` | As v1 (proven identical on real audio) | — |
| `ramp` | Required silence falls linearly as the segment grows; very short after `max_segment_sec` | `start_ms`, `end_ms`, `ramp_sec`, `max_segment_sec`, `hard_min_ms` |
| `adaptive` | Required silence = a quantile of this speaker's own mid-segment pauses × factor; optionally the threshold rises above the noise floor measured between segments | `pauses`, `pause_quantile`, `pause_factor`, `floor_ms`, `ceiling_ms`, `noise_floor`, `noise_margin`, `max_threshold` |
| `smart-turn` | When a pause reaches `probe_ms`, Smart Turn v3.2 (ONNX, CPU, downloaded from `pipecat-ai/smart-turn-v3`) scores the last 8 s; end if P(end) ≥ `end_probability`, else at `max_silence_ms` | `model`, `probe_ms`, `max_silence_ms`, `end_probability` |

The Speech record's `silence_waited_out_sec` is the silence that actually ended the segment, so
`qwen-seg`'s tail trim follows the tuner.

### 3.5 New component kinds and `cascade:v2` (`core/pipeline/cascade_v2.py`)

`cascade:v2` = enhancement → VAD → ASR → labeler → filter in the room stage, translation per
target language, exactly like the server. Settings:

| Part / setting | Worklog | Effect |
|---|---|---|
| `split_at_vad: true` | A10 | The 200 ms chunk is split at the sample where VAD ended speech: the first part goes into the flushed utterance, the rest into the next one (production does this; v1 bench did not) |
| `labeler: script` | B7, T1, N2, N10 | Re-labels each commit (and partial) by its letters among the conversation's languages: Hangul → ko, Latin → the one Latin-script language, etc. Fixes both "English passed through to Korean readers" and "Korean text sent to the translator as English" (`[RELABEL]`) |
| `filter: commit-rules` | A5, A6, A7, foreign script | Switches: `fillers` (only filler words; list per language, `extra_fillers` to extend), `foreign_script` (script of no conversation language, e.g. `啊。` in Korean audio), `repeats` (same text as one of the last N kept commits within `repeat_window_sec`), `canned_phrases` (exact list). Logged as `[DROP] rule=filler / foreign-script / repeat / canned-phrase gate=filter` |
| `enhancement: spectral` | upgrade #5 | Minimum-statistics spectral subtraction (32 ms delay). `vad_input` / `asr_input` each `raw`, `enhanced` or `mix` (`mix_weight` = share of raw, default 0.2) — the report's "dual path" |
| translation through `TranslationProcessor` | T3 | Bench now translates with the same code as the server (errors caught, same context handling) |

### 3.6 Bench, server and UI fixes (not switchable — they change no measured number)

| Change | Worklog | Files |
|---|---|---|
| Pipeline closed inside the event loop that loaded it; every child process (vLLM translator) stopped and waited for at the end of a run | H1 | `bench/__main__.py`, `core/utils/process.py` (`stop_all`) |
| Any unexpected exception → `status: failed` summary written, then re-raised | H2 | `bench/__main__.py` |
| `item_close` event now has its time | H3 | `bench/__main__.py` |
| New metrics: `passthrough_share`, `wrong_script_share`, `leak_share`, `translation_failures`, `commit_gap_p90/max_sec`, `speech_segment_p50/p90/max_sec`, `commits_per_min`, `term_recall`, `term_translation_recall` | M2, T2, N1, upgrade #1/#3 | `bench/metrics/diagnostics.py` |
| Dataset config keys `ids: [...]` (run named items only) and `terms: <glossary>` (scoring only, not in the config hash) | — | `bench/config.py`, `bench/dataset.py` |
| `python -m bench.vad`: VAD-only sweep on CPU | N1 | `bench/vad.py` |
| Replay page strips `language X` from partials of old runs | A9 (bench UI) | `bench/replay/static/session.html` |
| Mobile app strips `language X` from partials and finals (same rule as the web demo's `cleanAsr`) | A9 (mobile) | `STiTy-Mobile/src/utils/asrText.ts`, `HomeScreen.tsx` |
| Server: translator child stopped on shutdown; malformed/unknown WS messages logged as WARNING (were DEBUG); `settle` polls every 10 ms instead of spinning; a crashed translation worker unsubscribes | H1, V | `server/app/main.py`, `api/websocket.py`, `services/conversation/workers.py` |

The web demo (`show.html`) already cleaned headers; nothing changed there. The mobile app's
hard-coded RunPod fallback URL (V) was left alone — it is deployment config.

## 4. CPU evidence (already run)

`tests/test_qwen_seg_v2.py` feeds v1 and v2 the same scripted decodes through a fake of the
Qwen3-ASR streaming API (`tests/fake_asr.py`). Scripts write `<SEG>` with a space on both sides,
as the real model does; with a space on one side only, v1's cursor miscounts and shows a bug that
does not exist in real runs.

| Scenario | v1 output | v2, that switch only |
|---|---|---|
| A2 held fragment | `the rest is history.` | `Needless to say, the rest is history.` |
| A3 boundary | `It is in La Plata.` (next sentence dropped) | both sentences |
| A1 no speech | `I want to buy a new car.` committed over silence | dropped |
| A4 carry | `General Kenobi, you are a bold one.` lost | committed |
| N5 resync | `C three.` never committed | committed |
| A5 loop | `아니, 아니, 아니.` committed | not committed; `네 알겠습니다.` kept |
| A8 language list | committed | dropped |
| A9 partial header | partial contains `language English` | clean |
| N7 dot carry | `_find_duplicate` misses `괜찮습니다.` on a SEG trigger | caught |
| Parity | — | all 8 scenarios identical with every switch off |

`tests/test_components.py` covers the labeler, filter, enhancer, tuners, `qwen-la`, `qwen3.5:v2`
prompts (glossary, leak retry, meta retry, continuation), `cascade:v2` split point and a mock
end-to-end pipeline, and the diagnostics metrics.

What the scripted tests do **not** show: how often each situation happens in real decodes, and
whether a fix has side effects on real speech. That is what the GPU runs are for.

## 5. Experiments

### 5.1 How to run

```bash
tmux new-session -d -s upgrade -c "$PWD" "bash experiments/upgrade-260930/run.sh 0"
#   one or more tier numbers; rows run one at a time (one GPU job at a time)
#   progress: logs/upgrade-260930/status/chain.log; per run: logs/upgrade-260930/<dataset>__<pipeline>.log
#   finished rows get a .ok marker and are skipped next time; failed rows get .failed

uv run --project bench python experiments/upgrade-260930/compare.py \
  --dataset <dataset> --base <base pipeline> --variant <pipeline> [--items en_1845,en_1946]
#   metrics side by side, which log tags / DROP rules fired more or less, changed transcripts
```

Each row of `plan.tsv` has: tier, id, pipeline, dataset, **base** (the run it is compared with)
and **watch** (what to look at and what should happen). If a base run of the same code and
config already exists on the machine (e.g. the 2026-09-28/29 v1 runs), `run.sh` would still rerun
it; delete that row or touch its `.ok` marker to reuse it.

Before starting, check the restaurant+cafe dataset configs
(`northstar_en+ko-{en,ko}_restaurant+cafe.yml`, written from the worklog's description "MIT
restaurant IRs 0.5–1.0 + DEMAND SCAFE 0.5–1.0"): `$STITY_DATA_ROOT/augment/room/restaurant/` must
exist. If the 2026-09-29 runs used a config under another name, use that one instead so the
numbers stay comparable.

### 5.2 Tiers

Time per run is roughly: model load 3–13 min, plus the dataset's audio in real time (bench feeds
in real time), plus 4 s of silence per item, plus COMET. Worklog subsets 5–20 min of audio,
northstar ~11 min, full FLEURS ~65 min.

| Tier | Rows | What it answers | Rough time |
|---|---|---|---|
| 0 | 24 | v1 base runs on the subsets, and **parity**: `_off` vs v1, `_vad2fixed` vs `_off`, `qwen3.5-v2-4b_off` vs `_off` | 8–9 h |
| 1 | 31 | One ASR fix each, on the subset where the worklog found the bug | 10–11 h |
| 2 | 6 | `labeler` on northstar and the mixed set; `filter` on northstar | 2–3 h |
| 3 | 29 | VAD tuners, chunk size, `qwen-la`, noise handling on northstar clean and restaurant+cafe | 10–11 h |
| 4 | 9 | Translator options | 3 h |
| 5 | 4 | ASR biasing | 1.5 h |
| 6 | 14 | `…_all` on the full sets against the v1 runs in WORKLOG's Results tables | 14–18 h |

Run tier 0 first and stop if parity fails (section 5.3). Tiers 1–2 are the verification of the
claims in this report; tiers 3–6 decide what to switch on.

### 5.3 Pass criteria

- **Parity rows (tier 0):** `compare.py` prints `0 item(s) changed` and no metric differs except
  wall-clock ones (`*_ca_*`, `fsl`, realtime). Transcripts come from greedy decoding and were
  deterministic in the worklog ("same model, same audio gives identical WER/CER"). Translations
  can in principle differ by vLLM batching; if they do, rerun the v1 base once to see whether v1
  differs from itself before blaming v2. **If transcripts differ, stop**: v2 with switches off is
  not v1 on real decodes, and no other row can be attributed to its switch.
- **Fix rows (tiers 1–2):** the `watch` column names the items and log tags. A fix passes when
  (a) the named items change in the stated direction (read them with `--items`), (b) the switch's
  log tag fires (`[CURSOR-RESYNC]`, `[LOOP-GUARD]`, `[CARRY-DECODE]`, `[RELABEL]`,
  `[DROP] rule=…`), and (c) the control items (the last 12 ids of each `_worklog` subset) do not
  get worse. WER on 35-item subsets moves by a few items; read transcripts, not only the average.
- **Label (tier 2):** `passthrough_share` should drop to near 0; `lang_detect_accuracy` should
  rise on northstar (0.71 for the ko model before); BLEU on `en × fleurs_ko+en-en` should rise from
  2.1. COMET can go *down* there while quality goes up (M2: COMET rewarded the untranslated Korean).
- **VAD / LA (tier 3):** better = shorter `speech_segment_*` and `commit_gap_*`, higher
  `lang_detect_accuracy`, with WER/COMET not worse. A setting that cuts inside sentences shows as
  more short commits and a worse COMET.
- **Record every result** as a row in a table like WORKLOG's Results, with the run directory.

### 5.4 What each row can not tell

- `lang_detect_accuracy` on single-language FLEURS rows is forced by `restrict_languages` (M1);
  only the `_open` rows and the mixed/northstar sets measure it.
- The northstar glossary was built from the script's own PN/HON lines. `bias` and `mt-glossary`
  rows measure "what a user who typed the names beforehand gets" — an upper bound, not a blind test.
- `canned_phrases` is a list written from the en finetune's observed outputs; the same sentences
  said for real would be dropped too. It is there to compare with A1's gate, which is the proper fix.
- The `fillers` default list leaves out `아니`, `네`, `응` (they are real answers); `extra_fillers`
  can add them for a test.

## 6. VAD sweep on northstar (clean), CPU, already run

`python -m bench.vad`, tolerance 1.5 s. 121 reference sentences in 6 scenes.
`turn_end_recall` = share of sentence ends followed by a segment end within 1.5 s;
`end_precision` = share of segment ends that follow a sentence end within 1.5 s.

| Config | Segments | p50 / p90 / max length (s) | Recall | Precision | End delay p50 (s) | Missed sentences |
|---|---|---|---|---|---|---|
| `off` (silero 800 ms, = production) | 50 | 9.0 / 33.9 / 89.6 | 0.26 | 0.62 | 0.79 | 0 |
| `vad2fixed` | identical to `off` (all 50 segments equal) | | | | | |
| `vad550` | 76 | 6.0 / 19.2 / 36.2 | 0.40 | 0.65 | 0.60 | 1 |
| `vad400` | 112 | 4.1 / 11.4 / 26.3 | 0.55 | 0.58 | 0.44 | 1 |
| `vad250` | 179 | 2.4 / 6.5 / 18.9 | 0.65 | 0.44 | 0.28 | 1 |
| `vadthr35` | 46 | 7.7 / 36.6 / 89.6 | 0.23 | 0.61 | 0.81 | 0 |
| `vadthr65` | 54 | 7.6 / 30.8 / 89.4 | 0.27 | 0.61 | 0.78 | 1 |
| `vad-ramp` | 99 | 5.4 / 12.3 / 17.5 | 0.49 | 0.60 | 0.51 | 1 |
| `vad-ramp-fast` | 118 | 5.0 / 9.1 / 17.5 | 0.55 | 0.56 | 0.35 | 1 |
| `vad-adapt` | 73 | 6.7 / 18.8 / 41.5 | 0.36 | 0.59 | 0.59 | 1 |
| `vad-adapt-q75` | 136 | 3.0 / 11.0 / 24.1 | 0.57 | 0.50 | 0.33 | 1 |
| `vad-noise` | identical to `off` on clean audio (the noise floor stays below 0.5) | | | | | |
| `vad-turn` (Smart Turn v3.2) | 114 | 3.6 / 12.8 / 21.0 | 0.50 | 0.52 | 0.28 | 0 |
| `denoise-vad` | 50 | 9.1 / 33.9 / 89.6 | 0.26 | 0.62 | 0.83 | 0 |

Reading it: the production setting's segments match WORKLOG N1 (median 9.0 s, max 89.6 s). Only a
shorter required silence helps; the threshold does not. `ramp` keeps precision at the baseline while
nearly doubling recall, which is why it is in `…_all`. Smart Turn ends turns fastest (0.28 s) but
cuts inside sentences more often; it was never fine-tuned on Korean (report proposal #4 step 4).
The noisy set was not swept here (the augmentation files are only on the GPU machine); run
`python -m bench.vad` on `northstar_en+ko-en_restaurant+cafe` there first — it is CPU-only and
decides which VAD rows of tier 3 are worth the GPU time.

## 7. Not done, and why

| Item | Why not |
|---|---|
| Fixes in the production server (`Qwen3-ASR/examples/streaming_websocket_server.py`) | Per the user, production bugs are reported, not patched; v2 lives in `core/` (bench and `server/app`). A production↔`qwen-seg` parity test still does not exist (A10) |
| N4 (ko model writes English in Hangul), N3 (en model invents English), N11, invented text | Model quality; needs training |
| N9 (two speakers in one commit), N10 root cause | Needs diarization (proposal #7) or more tracing |
| M3, M5, M6, B6 | Metric definitions; changing them breaks comparison with the existing runs |
| Forced aligner and target-side LocalAgreement / AlignAtt for MT (proposals #1 step 4, #2 steps 2, 4, 5) | Needs model integration and a streaming MT protocol (append-only deltas) the clients do not speak yet. `qwen-la` + `continuation` are the parts that fit the current protocol |
| QE gate (#6), diarization (#7), TTS (#8), echo cancellation (#9), speaker extraction (#10), training (#11) | New models or client work; out of the "easy fixes" scope |
| Smart Turn Korean fine-tune | Training |
| Server `healthz` pinging the translator (T2), HTTP conversation create, tickets, late-join backlog | Server features, not fixes |

## 8. Files

New: `core/components/{transcription/qwen3_seg_v2.py, transcription/qwen3_la.py,
translation/local_qwen_3_5_4b/translator_v2.py, vad/silero_v2.py, vad/tuners.py, labeler/,
filter/, enhancement/}`, `core/pipeline/cascade_v2.py`, `core/utils/{scripts.py, glossary.py}`,
`bench/{vad.py, metrics/diagnostics.py}`, `configs/glossaries/northstar-260928.yml`,
`configs/pipelines/mock-v2.yml` and 89 generated `asr.qwen-seg-v2-*` / `asr.qwen-la-*` configs,
dataset configs `*_worklog.yml` (4) and `northstar_*_restaurant+cafe.yml` (2),
`experiments/upgrade-260930/{make_configs.py, plan.tsv, run.sh, compare.py, REPORT.md}`, `tests/`,
`STiTy-Mobile/src/utils/asrText.ts`.

Changed: `core/components/registry.py` (the new per-chunk methods `enhance`/`filter`/`label` are
not timed), `core/utils/process.py`, `bench/{__main__.py, config.py, dataset.py, pyproject.toml,
uv.lock, README.md, replay/static/session.html}`, `server/app/{main.py, api/websocket.py,
services/conversation/workers.py}`, `STiTy-Mobile/src/screens/HomeScreen.tsx`,
`configs/README.md`, `core/CLAUDE.md`, `CLAUDE.md`, `docs/WEBSOCKET_PROTOCOL.md`, `.gitignore`,
and `terms: northstar-260928` appended to the two untracked `northstar_en+ko-*.yml` configs.
