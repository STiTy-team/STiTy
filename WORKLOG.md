# WORKLOG — server demo + fresh bench (started 2026-09-28)

Scope notes from the user:
- No commits or pushes. Fixes stay as uncommitted changes in the working tree; each issue below
  lists the files it touched instead of a commit hash. Removing files (`git rm`) is allowed.
- No parallel jobs: one GPU job at a time.
- Server: HTTP conversation creation, the ticket flow and late-join backlog are **not implemented**
  in `server/app`. Test only what exists (join by shared `conversationId` in `start`) and see how
  far it goes as a minimal proof of concept.

Handoff review (2026-09-29): the campaign notes below were checked against the code and the 11
FLEURS runs. Wrong statements are corrected where they stand; everything still open is in
"Open issues" (ids A/T/M/H/V/D) near the end.

## Checklist

### Part 1 — server
- [x] 1. Read server/app
- [x] 2a. Start with `mock` (starts; env override was broken, see S1)
- [x] 2b. Start with `asr.qwen-seg-en+mt.qwen3.5-4b` (after S3, S4)
- [x] 3. Scripted client `server/scripts/client.py` — all cases pass on mock (35/35 before the
  translator-count check), en (37/37, `logs/server/client-en.3.log`) and ko (37/37,
  `logs/server/client-ko.1.log`). Not implemented in the server, so not testable: HTTP create,
  tickets, late-join backlog (the client prints how many finals a late joiner missed: 4 en, 3 ko).
- [x] 3b. Web demo against this server — `demo_proxy.py` in plain relay mode, `show.html?auto=en`
  and `?auto=ko` driven by headless Chrome (Playwright on /usr/bin/google-chrome; the Chrome
  extension was not connected). en: hello, ready, 23 partials, 6 finals with Korean
  translations; ko: 8 finals with English. The default lane layout leaves other-language lanes
  empty (one `targetLang` per client); `layout=tag&pick=<target>` renders cleanly.
- [x] 4. `server/README.md` written (launch, GPU split, translator, env, mobile app, web demo, demo script); `make serve` removed (S2)

### Part 2 — bench
- [x] 1. `git rm -r bench/runs/` (12 tracked files staged for deletion) and removed the untracked leftovers (items/events of 4 runs, 44 MB)
- [x] 2. Data: fleurs sets present (see B1 for 268 vs 270); cafe noise = DEMAND SCAFE, hall IR =
  Arvedi Auditorium (OpenAIR is offline, B2); `$STITY_DATA_ROOT/augment/README` written;
  checks passed (16 kHz mono via `load_window`, noise RMS ≈ −40 dBFS and never silent, IR direct
  peak 36–40 dB above the pre-onset floor at 10–21 ms, RT60 1.3–1.4 s)
- [x] 3. 11 runs — all 8 qwen runs finished 14:01, 3 mock runs' bench part 11:23 (COMET after). Started 2026-09-28 03:26 in tmux `bench-full`. Smoke first (3 items each,
  en / ko+cafe / ko+hall / mock): all `status: ok`, COMET present, `augment` on every row.
  Smoke configs and run dirs deleted afterwards. API cost: none (local ASR, local Qwen3.5, local
  COMET), so no cost tracking was needed.
- [x] 4. Check every run — per-run notes below, full output `logs/bench/check/<run>.txt` (`logs/bench/check_run.py`)
- [x] 5. Replay — `logs/bench/replay_check_http.py` + `logs/bench/replay_check_browser.py` against
  `python -m bench.replay --port 9140` (same code as `make replay`; the user's own `make replay` on
  :9130 was left alone). Dashboard: 11 runs in 5 datasets (fleurs-en-ko 3, fleurs-ko-en 3,
  fleurs-ko+en-en 3, .cafe 1, .hall 1); for each dataset the leaderboard has the right rows, the
  distribution chart and the accuracy-vs-latency scatter render (one dot per run). Session replay
  opens for all 11 runs (item loaded, events and timing playhead drawn, no page errors; the only
  console error is the browser's `/favicon.ico` 404). cafe/hall: the Data line reads
  `augmented: noise → volume` / `room → volume`; **all 268 items of each** return data without a
  mismatch error and serve audio that differs from the clean file (relative RMS difference median
  12–14, min 0.14 cafe / 0.96 hall); no clipping (at most 2 samples at full scale per item — the
  volume step's peak cap). Screenshots in `logs/bench/replay-shots/`.
- [x] 6. 11 `bench/runs/*/summary.json` written; not committed (user: no commits). The old tracked summaries are staged for deletion (`git rm`).

## Issues

(each: symptom, evidence, root cause, fix + files, how checked)

### S1. `make server PIPELINE=<name>` always ran the `mock` pipeline
- Symptom: the pipeline named on the command line was ignored.
- Evidence: `STITY_STITY__PIPELINE=nonexistent ... AppSettings().stity.pipeline` printed `mock`.
- Root cause: `AppSettings.settings_customise_sources` returned only `init_settings` and the YAML
  source, and the class had no `env_prefix`, so the `STITY_STITY__PIPELINE` variable the Makefile
  sets was never read. There was also no way to bind anything but 127.0.0.1, so a phone on the
  LAN could not reach the server.
- Fix: `server/app/config.py` — `env_prefix="STITY_"`, `env_nested_delimiter="__"`, env source
  between init and YAML; new `server.host` (default 127.0.0.1). `server/app/__main__.py` passes
  `host` to uvicorn. `Makefile` — `server` target takes `HOST=` and `PORT=`.
- Checked: `AppSettings()` with the variables set prints the pipeline/host; `make server PORT=8799`
  listened on 8799 and answered `/api/v1/healthz`.

### S2. `make serve` target was dead
- Symptom/evidence: it ran `python -m server --pipeline ...`; there is no `server/__main__.py`
  (the package is `server/app`) and nothing parses `--pipeline`. `PROFILE`/`STITY_PROFILE` in the
  `server` target was read by nothing (grep finds no reader).
- Fix: removed the `serve` target and the `PROFILE` variable from `Makefile`. Nothing in the docs
  referenced either.

### S3. Real pipelines could not load in the server environment
- Symptom: `make server PIPELINE=asr.qwen-seg-en+mt.qwen3.5-4b` died at startup.
- Evidence: `logs/server/srv-en.attempt1.log`: `ModuleNotFoundError: No module named 'torch'`
  from `core/components/vad/silero.py`.
- Root cause: `server/pyproject.toml` listed only web dependencies. `core/` imports torch,
  silero_vad, qwen_asr, vllm, httpx, soundfile, librosa lazily, so `mock` started and every real
  pipeline died at model load. `numpy>=2.5.3` also conflicted with vllm 0.14 (numba needs <2.3).
- Fix: `server/pyproject.toml` — added `qwen-asr[vllm]` (path source `../Qwen3-ASR`, same as
  bench), `silero-vad>=6.2`, `httpx`, `soundfile`, `librosa`; `requires-python <3.13` (vllm
  wheels); unpinned numpy. `server/uv.lock` relocked (vllm 0.14.0, torch 2.9.1, same as bench).
- Checked: `uv sync` succeeded; server start with the en pipeline — see S4.

### S4. Qwen3.5 translator vLLM ran out of GPU memory at startup
- Symptom: with the ASR engine loaded, the translator's `vllm serve` exited with code 1 and the
  server startup failed (`ProcessError: qwen3.5-vllm exited with code 1`).
- Evidence: `/tmp/qwen3.5-vllm.log`: `Capturing CUDA graphs (PIECEWISE): 0/51` then
  `torch.OutOfMemoryError ... this process has 14.35 GiB memory in use` while its budget was
  0.52 × 23.5 = 12.2 GiB. The 2026-09-27 runs that worked were launched with
  `limit_mm_per_prompt={image:0,video:0}, max_num_batched_tokens=2048, max_num_seqs=8`
  (`non-default args` lines in the same log) and captured only 5 graph sizes.
- Root cause: `core/components/translation/local_qwen_3_5_4b/translator.py::_start_server` did not
  pass those three limits; the working copy that had them was never committed. Without them vLLM
  sizes the engine for 256 sequences plus Qwen3.5's vision encoder, and CUDA-graph capture
  goes over its share. This hit bench as well as the server (same code).
- Fix: pass the three limits in `_start_server` (same file).
- Checked: `logs/server/srv-en.log` — `Loaded models in 780.3s` (cold HDD), `Application startup
  complete`, 22.0 GB on the card; the scripted client then passed (below).

### S5. Ctrl-C on the server printed a traceback from the translator's exit hook
- Evidence: end of `logs/server/srv-en.log`: `Exception ignored in atexit callback ...
  ManagedProcess._terminate ... ProcessLookupError`.
- Root cause: Ctrl-C reaches the whole process group, so the `vllm serve` child is already dead
  when `core/utils/process.py::ManagedProcess._terminate` calls `terminate()`.
- Fix: ignore `ProcessLookupError` there. The remaining `AsyncLLM.__del__ ... 'NoneType' object is
  not callable` line is vLLM's own destructor during interpreter teardown after the server has
  logged `Stopped STiTy server`; the card is freed (nvidia-smi back to baseline), left as is.

### Client-side findings while writing `server/scripts/client.py` (not server bugs)
- Mock translator answers `[ko] <text>`; language checks accept that when the backend is `mock`.
- Pacing: a microphone delivers a 100 ms piece once that time has passed; the first version sent
  it at the start of its span (negative delays) and, after a pause, burst the next clip to catch
  up on one fixed clock. That burst plus a `quiet()` that counted idle time from the stream
  start made "room survives a client dropping mid-sentence" fail with 0 new finals. Fixed in the
  client (clock re-anchors per `stream()` call; delay measured at receipt; idle counted from the
  last piece sent). Server behaviour was right: `logs/server/srv-en.log` shows the room kept
  running, and variants (drop / stop / silence / none) in fresh rooms all delivered the fragment
  and the full sentence.
- Japanese check counted kana only; kanji-only translations (`構造物への落書き。`) failed it.
- Malformed JSON text frames are ignored (same path as unknown message types, validation errors
  without a field location), not refused. Kept as designed; the client checks the socket stays up.

### Server test results (key numbers)
| server | checks | final delay (median, max) | translator calls vs distinct (segment, target) | finals delivered |
|---|---|---|---|---|
| en (`asr.qwen-seg-en+mt.qwen3.5-4b`) | 37/37 | 0.39 s, 0.61 s | 23 = 23 | 37 translated finals |
| ko (`asr.qwen-seg-ko+mt.qwen3.5-4b`) | 37/37 | 0.35 s, 0.38 s | 20 = 20 | 34 translated finals |

Stages confirmed from the server log: one `Started transcription worker` per room opening
that had audio, one `Started <lang> translation worker` per language per opening, every
`Opened conversation` has its `Closed conversation` (5/5), no tracebacks, healthz ok after the
bad-input and drop cases.

### ASR finding: made-up sentence after a `seg` reset (en finetune, not the server)
- `server/scripts/client.py` single case: after `Lions are the most social cats ...` the 0.4 s tail
  decodes to `I want to buy a new car.` Server log: `REPEAT-COLLAPSE before='I want to buy a new
  car. ×3' after='... ×2'`, then two `vad` commits of it. Deterministic (same with and without
  pauses between clips), same audio through `core/components/transcription/qwen3_seg.py` as bench.
  The `silence-phrase` rule only knows `I'm sorry, but I can't help you.`-style phrases. Matches
  the known en-finetune behaviour (A1, A6) (hallucinated phrases appended on 25/270 FLEURS items in the
  2026-09-27 run). Left for the bench check (Part 2 §4) rather than patched blind.

### B1. FLEURS sets have 268 items, bench/README.md says 270
- Evidence: `fleurs/en_us/manifest.jsonl` and `ko_kr` have 268 lines, identical id sets; the
  2026-09-27 build log (`logs/datasets.log`) wrote 270. Both sets were rebuilt 2026-09-28 00:39
  (no log here) and their `dataset.yml` now lists `translations: [de, ko]` / `[de, en]`.
- Root cause: that rebuild also took German as a target; `fleurs/convert.py` skips sentences
  without a reference in every target language, and ids 1681 and 1854 have no German one.
- Decision: kept as is, not rebuilt. The rebuild is an hour old and sits next to
  `fleurs/ko_kr+en_us.level-26`, which suggests another session is using this data; rebuilding
  in place would change their inputs under them. The 11 runs are all on the same 268 (en/ko) and
  the mixed set built from them, so they are comparable with each other. Not comparable item-for-
  item with the removed 2026-09-27 runs (270).

### B2. OpenAIR (suggested hall IR source) is offline
- Evidence: every `openair.hosted.york.ac.uk/?page_id=…` returns `<title>Account Suspended`.
- Replacement: the Arvedi Auditorium RIR dataset (Zenodo 20098848, CC BY 4.0, a real concert
  hall). Only the four needed files were read out of the 904 MB zip with HTTP range requests.
  Details and checks in `$STITY_DATA_ROOT/augment/README`.

### B3. Augmented configs raise quiet clips, the clean config does not (confound, not a bug)
- FLEURS ko is very quiet: RMS median −46.1 dBFS, quietest −65.5 (ko_1770). `fleurs-ko-en.cafe`
  and `.hall` end with `volume: {min_level_db: -23}`, so their speech reaches the model up to
  ~30 dB louder than in clean `fleurs-ko-en` (smoke: `gain_db` 11–31). Noise and reverb make
  things harder while the volume floor makes them easier, so "cafe/hall are worse" is not
  guaranteed. Checked against the per-item gains: not seen — per-item WER vs `gain_db` Spearman
  0.02 (cafe), 0.01 (hall); raised clips got slightly worse.

### B4. qwen-seg ASR dropped the rest of a sentence after a stray `language English` header
- Symptom: first full en→ko run, en_1895: reference "as knowledge of greek declined the west found
  itself cut off from its greek philosophical and scientific roots", transcript only
  `As knowledge of Greek declined,`. 15 of 268 items logged `[DROP] rule=header-only
  text='language English'` (12 of them at the VAD flush).
- Evidence: events of en_1895: after the `seg` commit the reassembled decode was
  `…declined, <SEG>language English the West found itself cut off from its Greek philosophical
  and scientific` (the partial showed those words); at the VAD flush the candidate was
  `language English` alone and was dropped as header-only.
- Root cause: `core/components/transcription/qwen3_seg.py::flush` re-decodes the tail; that
  decode stopped right after a fresh `language English` header. The existing guard ("if the
  re-decode leaves nothing uncommitted, keep the text from before") compared the raw string
  with empty, and `<SEG>language English` is not empty, so the words were replaced by a bare
  header. Reproduced offline with `parse_asr_output` + `uncommitted_from` on the two strings.
- Fix: the guard compares after `strip_lang_headers(display_of(...))` (same file).
- Checked: 30-item en smoke against the pre-fix run (copy in `logs/bench/before-B4/`): en_1687
  0.20 → 0.00 (recovers "with each generation"), the other 29 items byte-identical; the one
  remaining header-only drop (en_1683) is a finish leftover with reason `seg`, harmless.
- Consequence: the campaign was stopped (run 1 finished, run 2 1.5 min in) and restarted from
  scratch at 04:44 so all 11 runs use the same code. Pre-fix run 1: WER 0.1305, BLEU 19.84,
  COMET 0.8435 (`logs/bench/before-B4/summary.json`).

### B5. The mixed ko+en set is not level-matched (user question, 2026-09-28 08:15)
- Measured with the datasets repo's `_contract.active_power` per turn (offset/duration from the
  manifest):

  | set | ko turns (median active level) | en turns | jump at a language switch |
  |---|---|---|---|
  | `fleurs/ko_kr+en_us` (used by `fleurs-ko+en-en`) | −45.3 dBFS (p10 −51.8, p90 −27.4) | −58.4 dBFS (p10 −63.6, p90 −22.3) | median 16.5 dB, p90 32.8, max 51.3 |
  | `fleurs/ko_kr+en_us.level-26` | −26.0 | −26.0 | median 0.0, max 0.8 (1 turn held back to avoid clipping) |

- Cause: `mix.py` only levels turns with `--level`; the set in the brief was built without it,
  so each turn keeps its FLEURS recording level, and English is ~13 dB quieter.
- Cause: the levelled set was built (`mix.py fleurs --src ko_kr en_us --level -26`, 00:42), but
  `mix.py` writes it to `fleurs/ko_kr+en_us.level-26/` (the suffix comes from `--level`), and an
  un-levelled rebuild of `fleurs/ko_kr+en_us/` ran at 00:40. `configs/datasets/fleurs-ko+en-en.yml`
  still named the un-levelled directory.
- Fix (user's choice, 08:30): `configs/datasets/fleurs-ko+en-en.yml` now names
  `fleurs/ko_kr+en_us.level-26`. No mixed run had started (qwen ones were still queued, mock
  ko+en is third in its queue), so all three `fleurs-ko+en-en` runs use the levelled set. Same 44
  conversations and references; only turn gains differ. The briefly queued extra
  `.level-26` config and runs were cancelled and removed.
- Note: the single-language sets (`fleurs/en_us`, `fleurs/ko_kr`) are raw FLEURS levels
  (en turns median −58.4 dBFS, ko −45.3), as the brief specifies; cafe/hall add `volume` on top (B3).

- Reorder (user, 08:23): the chain was stopped 3 items into `ko × fleurs-ko-en.hall` and
  restarted with the two levelled-mix qwen runs first. bench has no resume (`reset_run_dir`
  wipes a run's files on start), so hall restarts from item 1 later; finished runs are skipped
  by their `status/*.ok` markers.

### R1. Replay "compare with" offered runs that did not hear the same datapoint (user report)
- Symptom: the session page's compare picker listed any run with the same manifest hash
  (`session.html`: `r.dataset_sha === DATA.run.dataset_sha`), so clean `fleurs-ko-en`, `.cafe` and
  `.hall` (same `fleurs/ko_kr` manifest, different audio) were offered against each other; the
  item on screen was never checked, and a running run (no summary, no hash) was never offered.
- Fix: `bench/replay/runs.py::run_catalog` adds `data_key` (the run's `dataset`/`target`/`augment`
  config) and `item_ids` (ids in its `items.jsonl`); `bench/replay/static/session.html` offers a run
  only with the same `data_key` (and hash when both have one), a different pipeline config, and a
  row for the item on screen, rebuilt on every item change (`buildComparePicker`).
  `bench/README.md` describes the new rule.
- Checked (headless Chrome, replay on :9140): qwen `fleurs-en-ko` ↔ `mock-fleurs-en-ko` offered
  both ways; clean `fleurs-ko-en` and `.cafe` offer nothing; on the qwen page item `en_1660`
  (mock has it) offers mock, `en_1776` (mock not there yet) offers nothing; no page errors.

### R2. Dashboard dataset selection as a dropdown (user request)
- Before: a corpus `<select>` (only `fleurs`) plus one pill button per dataset config; the user
  wanted the datasets themselves in a dropdown.
- Change: `bench/replay/static/dashboard.html`/`.js` — the corpus select and pills are replaced by
  one `<select id="dataset">` listing every dataset config grouped by corpus (`<optgroup>`), each
  with its run count; `state.corpus` is gone. `bench/README.md` updated.
- Checked (headless Chrome): options `fleurs-en-ko (2 runs)`, `fleurs-ko+en-en (1 run)`,
  `fleurs-ko-en (1 run)`, `fleurs-ko-en.cafe (1 run)`, `fleurs-ko-en.hall (1 run)`; selecting
  `.cafe` switches the leaderboard, writes `#dataset=fleurs-ko-en.cafe`, and survives a reload;
  no page errors.

### B6. Layer timing playhead is not linear; bench feeds audio serially (observation, user question)
- The session page plays on the audio clock; the Layer timing x-axis is wall clock, and the
  playhead is `wallAt(now)` interpolated from `chunk` events. bench awaits
  `pipeline.listen(chunk)` before feeding the next chunk (`bench/__main__.py:73`), so the 2 s ASR
  decode makes one chunk take ~0.26–0.44 s and the next ~0.01 s (en_1895: audio 4.0 at wall 4.278,
  audio 4.2 at 4.284). Chunk lag over the en→ko run: median 0.04 s, p99 0.43 s, max 2.7 s.
- The chart is accurate. Caveat for the numbers: in bench, translation runs inside `listen()`
  (`CascadePipeline._translated`), so audio feeding also waits for the translator, while the
  server runs transcription and translation in separate workers. Wall-clock (`_ca`) latencies in
  bench include that wait. `fsl` (`committed_elapsed_sec − decision_audio_sec`) is stamped before
  the commit's own translation, so it only includes waits caused by earlier chunks (M6);
  the audio-clock ones (`laal_ms`, `yaal_ms`, `longyaal_ms`, `token_emission_ms`) do not.
- Replay fix (user asked, 2026-09-28): `bench/replay/static/session.html` — the playhead now
  sweeps the wall axis at playback speed from the item's median chunk lag (`feedLag` in
  `timingModel`) instead of following `wallAt(now)`; commit lines and speech spans still use
  `wallAt`, so their positions are unchanged. `bench/README.md` says so. Checked headless on
  en_1895 over audio 0.2–6.2 s (through the 4.0 s stall): a constant 108.1 px per audio second
  between 200 ms samples, no page errors.
- Not fixed (needs a bench change and a full rerun): the serial feeding itself. Accuracy and
  audio-clock latencies are unaffected — the pipeline's decisions depend only on the audio
  sequence (server pause test and the B4 smoke gave byte-identical output), so WER/CER/BLEU/
  COMET/LAAL/YAAL/LongYAAL/language detection hold. The `_ca` metrics include the wait, `fsl`
  only the part from earlier chunks; `token_emission_ms` may shift slightly (partials are throttled on wall time).
- Measured (`logs/bench/feed_wait.py`): how long the chunk carrying each commit's decision audio
  waited before its processing could start (previous chunk still running). 90% of commits: 0 ms;
  mean 1–10 ms per run; max 0.19–0.92 s (en→ko 590 commits: mean 5 ms, max 345 ms; ko on en:
  mean 10 ms, max 921 ms). The commit's own decode and translation (median 150–245 ms) are in the
  server's path too. So the serial feeding moves `fsl` and the `_ca` metrics by a few ms on
  average; the playhead jumps come from the decode chunk itself, not from lost accuracy.

### B7. Language label is set once per decode window, so turns after a language switch keep the old label (northstar-260928, 2026-09-29)

User report: "the translation isn't working at all". It is working; the language label that
decides whether and how to translate is wrong for a large share of commits.

**What happens.** Qwen3-ASR writes `language Korean<asr_text>` once at the start of a decode
window (slot) and then keeps appending sentences separated by `<SEG>`. When the speaker switches
to English inside that window, the model does not write a new header, so `state.language` stays
`Korean`, and `qwen3_seg.py::_language()` stamps every commit of the window with it. Raw decode
from `northstar-260928_ko+en-ko`, scene2 at 42 s:

```
language Korean<asr_text>하 4일 4일? <SEG> 3일 맞네요. <SEG> 3일 화요일? <SEG> 아 그래. <SEG> Another day, change. <SEG> My favorite thing in the whole
```

`Another day, change.` is committed as `ko`. Then `core/pipeline/stages.py::translate`:

- target `ko`: label equals target, so the English text is passed through untranslated.
- target `en`: the translator is told the English text is Korean (`Translate ONLY the line below,
  which is in Korean, into English`); sometimes it answers in Korean (`And it's the Bluetooth
  pairing at that.` -> `그건 블루투스 페어링 문제입니다.`).

**Size.** Commits whose text is mostly Latin letters but labelled not-`en`, and mostly Hangul but
labelled not-`ko` (script count over `transcribed` events):

| run | Latin text | labelled not-en | Hangul text | labelled not-ko |
|---|---|---|---|---|
| northstar ko+en-en, ko model | 79 | 34 | 228 | 7 |
| northstar ko+en-ko, ko model | 79 | 34 | 228 | 7 |
| northstar ko+en-en, en model | 116 | 2 | 97 | 90 |
| fleurs ko+en-en lvl26, ko model | 208 | 0 | 280 | 0 |
| fleurs ko+en-en lvl26, en model | 379 | 0 | 182 | 175 |

The FLEURS mix has a pause between turns, so windows reset and the ko model labels right (0
wrong); northstar has fast turn-taking inside one window, so 43% of the ko model's English
commits are labelled `ko`. The en model's Korean-labelled-`en` rows are mostly the model's own
header (it says `language English` for Korean, see T1), plus the same
window effect.

**Is it our bench?** No. Checked the experiment side: translator calls are correct when the label
is right (`Wow. Okay.` -> `와, 알겠습니다.`), both targets got identical transcripts (WER 0.375 /
CER 0.281 on both), the COMET set has the 49 English script lines expected for target ko, and
BLEU uses `ko-mecab` for Korean. The scores are low because of the label.

**Is it in the original server?** Yes, the root cause is. `streaming_websocket_server.py` reads
the same window-level `state.language` (`current_lang = state.language or ""`, line 2107). The
server hides it only on the Google Translate path: Google returns a detected source language and
`_maybe_fix_direction` re-translates when it equals the target. With the local translator
(`--local-translation`, `core/translator/local_translator.py`), the detected language is the ASR
label echoed back, so the server fails the same way. Bench runs `qwen3.5` locally, so it shows the
failure. Per the user (2026-09-29): the problem exists in the original server, so it is reported
here and not fixed.

**Effect on the northstar numbers.** Every northstar run carries it (clean and restaurant+cafe, both models, both targets). BLEU/COMET for the ko model
drop most on target ko (BLEU 5.2, COMET 0.567 vs 11.0 / 0.699 on target en), because English lines
pass through untranslated. `lang_detect_accuracy` (0.71 for the ko model) measures this label, so
it is the number to watch when a fix lands.

**Follow-up (user, 2026-09-29): English target still shows Korean.** Every Korean-looking
translation in the four English-target runs was classified by cause:

| run | commits | Korean in English output | Hangul text labelled `en`, passed through | English labelled `ko`, translator answered in Korean |
|---|---|---|---|---|
| en model, clean | 298 | 125 | 125 | 0 |
| en model, restaurant+cafe | 275 | 90 | 90 | 0 |
| ko model, clean | 307 | 7 | 6 | 1 |
| ko model, restaurant+cafe | 294 | 5 | 4 | 1 |

So it is almost entirely one path: the label says `en`, the target is `en`, and the text is passed
through untranslated. Two places do that: `core/pipeline/stages.py::translate`
(`item.language == target_lang`) and `core/components/translation/local_qwen_3_5_4b/translator.py`
(`source_lang == target_lang`). The translator itself is never called on those lines.

For the en model the label is mostly the model's own header, not the window effect above: its raw
decode windows open with `language English` over Hangul text 51 times against `language Korean`
7 times (clean run). The ko model's few cases are the window effect.

Existed before: yes. The original server's local translator has the same skip —
`core/translator/local_translator.py` lines 156 and 399, `if source_code == target_code: return
text` (since `9e90c1d7`, 2026-09-06) — and it gets the same ASR label as `source_code`. Only the
server's default Google path avoids it, because Google detects the source itself and ignores the
label. There is no Google translator component in `core/components/translation/` (only `mock`,
`local`, `qwen3.5`), so bench cannot show the Google behaviour. Reported, not fixed, per the user.

Same mechanism as T1 below, which covers the FLEURS mix.

**Possible fixes (not done).** Label each committed sentence rather than the window (e.g. by
script, as the web demo proxy already does (`demo_proxy.py::script_lang`), or by re-reading the header for the sentence's decode), or
never pass through on label alone and let a detector decide, as the Google path effectively does.

## Run checks (per run; full output in `logs/bench/check/<run>.txt`, made by `logs/bench/check_run.py`)

**asr.qwen-seg-en+mt.qwen3.5-4b × fleurs-en-ko** — status ok, 268 items, 0 errored, 0 empty,
COMET 0.8454 (not in `unavailable`; only `longyaal_*` are, correctly, since the run is not long-form).
WER 0.120 / CER 0.0715 / BLEU 19.9 / YAAL 3635 ms / FSL 0.15 s; language detection 590/590 en.
Worst items are made-up tails from the en finetune (see the ASR finding in Part 1): 36 items
commit at least one made-up phrase (51 commits; 14–26 items are >30% longer than the reference
depending on the measure), 9 of them with `I want to buy a new car.` twice, others `I'm sorry,
sir. I'm afraid I can't help you.` (A1, A6); the translator translates them, so they also cost
BLEU/COMET. Translation problems: 18 items get Chinese in the Korean output, e.g. en_1930
(`6 명의 인질,其中包括儿童和老人…`), en_1695 also Arabic (model quality, under Open issues); en_1703 fragment `Of nine members yesterday.` → `구원 일요일.`
(wrong meaning, attached to the right sentence). No empty translations, no misplaced ones.
Language detection 590/590 is forced by the harness (M1).

**asr.qwen-seg-ko+mt.qwen3.5-4b × fleurs-ko-en** — status ok, 268 items, 0 errored, 0 empty,
COMET 0.8376. WER 0.156 / CER 0.0805 / BLEU 21.9 / YAAL 3831 ms / FSL 0.23 s; 520/520 ko.
No repeated sentences. Made-up continuations instead of the heard words in 3 items: ko_1833
(Aristotle's four elements → "사람은 태어날 때부터 타고난 성격이 있다…"), ko_1709, ko_1813.
Proper nouns suffer most (ko_1817 셰갠 존 for 솅겐 존 → "Sean John"; ko_1885 drops the time).

**asr.qwen-seg-ko+mt.qwen3.5-4b × fleurs-ko+en-en (levelled)** — status ok, 44 conversations,
0 errored, 0 empty, COMET 0.8325, LongYAAL 3724 ms (LAAL/YAAL correctly unavailable: long-form).
WER 0.125 (ko+en 0.119 / en+ko 0.130 — per talk order, not per language, M5), CER 0.077, BLEU
21.8. Language detection 0.941: the 25 "English commits labelled `ko`" are Hangul transcriptions
(fillers, English words spelled in Hangul) whose label matches the text — the transcription is
what is wrong; the 4 "Korean labelled `en`" are English sentences the metric placed in a Korean
turn by time window. Label-vs-script agreement in this run is 100% (M1). Worst: a lost English turn in
conv0024 and filler loops (`그러니까.`, `아니, 아니.`) in 4 conversations — the loop bug (A5).
conv0043 `prides` → `Klubs`. No empty or wrong-script translations.

**asr.qwen-seg-en+mt.qwen3.5-4b × fleurs-ko+en-en (levelled)** — status ok, 44 conversations,
0 errored, 0 empty, COMET 0.6045, LongYAAL 4595 ms. WER 0.268, CER 0.197, BLEU 2.1, language
detection 0.504. Wrong-language model, not a bug: the English finetune writes Korean speech in
Hangul fairly well (WER still 0.27 overall) but labels it `en` (279 of 286 Korean-turn commits:
177 are Hangul, 102 are English — canned phrases, or Korean speech the ASR itself translated,
48 of them); with target `en`, text labelled `en` is passed through untranslated, so the Korean
turns reach the English reader in Korean, hence BLEU 2 (T1). COMET 0.60 flatters it: rows left
in Hangul average 0.637, above the English rows (M2). Only the web demo proxy re-tags finals by
script (`demo_proxy.py::fix_direction`/`script_lang`, commit `dc9110f6`); the mobile app,
core/ and bench/ do not. Made-up tails again (`I'm sorry,
sir. I'm afraid I can't help you.` opening conv0000/0034, `I want to buy a new car.`); no
repetition loops (0 items with ≥5 `DEDUP-SKIP`).

**asr.qwen-seg-ko+mt.qwen3.5-4b × fleurs-ko-en.cafe** — status ok, 268 items, 0 errored, 0 empty,
COMET 0.8261. WER 0.176 / CER 0.089 / BLEU 21.5 / YAAL 3905 ms. Somewhat worse than clean
(WER 0.156, COMET 0.838), not a collapse. Every row has an `augment` record: `noise.level`
0.109–0.497 (config [0.1, 0.5]), all four SCAFE files used (61–76 items each), `volume.gain_db`
0–41.7 (mean 19.8) at `min_level_db −23`. One loop item (ko_1848).

**asr.qwen-seg-ko+mt.qwen3.5-4b × fleurs-ko-en.hall** — status ok, 268 items, 0 errored, 0 empty,
COMET 0.8198. WER 0.201 / CER 0.108 / BLEU 20.9 / YAAL 4157 ms — worse than clean and than
cafe. Every row has an `augment` record: `room.level` 1.0 (config 1.0), all four Arvedi IRs used
(64–71 each), `volume.gain_db` 0–42.5. Reverb triggers the loop bug (A5) in 5 items
(ko_1848, 1887, 1902, 1931, 1946); ko_1902 and ko_1931 are only `아니, 아니, 아니.`. ko_1917 has
a made-up clause (로마 → "로봇 몸구조에 부착되어…").

**asr.qwen-seg-en+mt.qwen3.5-4b × fleurs-ko-en (English model, Korean audio)** — status ok, 268
items, 0 errored, 0 empty, COMET 0.8444. WER 0.166 / CER 0.086 / BLEU 22.4 / YAAL 3987 ms;
language detection 0.978 — the 10 "commits with no language" (`ko->?`) are Chinese text (`啊。`
×8, `如果他`, `对吗？`) translated into spurious English; the rest of the score is forced by the
harness (M1). Only mildly worse in WER than the
Korean model (0.156) — the English finetune still writes Korean well; the gap is mostly numbers
spoken as words (ko_1883 `15미터` → `십오 미터`, `2011년` → `이천십일년`) and spacing. COMET is
even slightly higher (0.844 vs 0.838), within noise. Looks like a wrong-language model, not a bug.

Augment files used by the cafe/hall runs (md5, made 03:15; `install.sh`/`convert.py` were added
to `datasets/augment/` at 12:44 without touching them — regenerating them could make replay
refuse those runs): ch01 28574aa6…, ch06 bc2622e6…, ch11 1b6a4f00…, ch16 fd68ed13…;
A105 18a3446c…, A306 9f14b886…, A406 b0a7acf8…, A506 e82d46f1….

**asr.qwen-seg-ko+mt.qwen3.5-4b × fleurs-en-ko (Korean model, English audio)** — status ok, 268
items, 0 errored, 0 empty, COMET 0.8600. WER 0.069 / CER 0.048 / BLEU 21.9 / YAAL 3825 ms;
language detection 0.996. **Better than the English model on the same audio** (WER 0.120,
COMET 0.845), against the brief's expectation. Why: the English finetune's made-up tails
(16 items >30% longer than the reference, `I want to buy a new car.`…) cost it ~0.05 WER, and
the Korean finetune has none on this set; its own errors are lost second halves (next item).
So for English audio the Korean finetune is the better model here; worth knowing before choosing
which server the app points English speakers at.

- Lost second halves (pre-existing, left as is): en_1738 ends "…entered the water." (drops
  "even a giant dinosaur such as t rex would be no match for it"), en_1989 drops everything after
  its first clause. A `dangling-lang-header` reset carries 2–4 s of audio into a new slot's
  `audio_accum`; when VAD fires on the same chunk, `flush()` only re-decodes if there is previous
  text, `buffer` audio or a "short utterance", none of which holds, so the carried audio is
  thrown away with the slot. The production server has the same condition
  (`streaming_websocket_server.py`: `if _pre_text or _has_buffered_audio`, `_cur_buf` = `buffer`)
  and also carries in `audio_accum`, so this is not a regression of the port. Frequency (items
  where a header reset and a VAD reset land on the same chunk): en→ko 2, ko model on en 4,
  ko→en 0, en model on ko 1, hall 0. A second path loses words even without VAD firing — see A4.

## Long jobs

(command, tmux session, log path)

| what | command | tmux | log |
|---|---|---|---|
| en server | `bash logs/server/run_server.sh asr.qwen-seg-en+mt.qwen3.5-4b srv-en` (wraps `make server PIPELINE=... HOST=0.0.0.0`, exports `CUDA_HOME`) | `srv-en` | `logs/server/srv-en.log` (attempt1: S3, attempt2: S4) |
| ko server | `bash logs/server/run_server.sh asr.qwen-seg-ko+mt.qwen3.5-4b srv-ko` | `srv-ko` | `logs/server/srv-ko.log` |
| bench smoke (3 items each: en, ko+cafe, ko+hall, mock) | `bash logs/bench/chain.sh logs/bench/smoke.list smoke` | `bench-smoke` | `logs/bench/<run>.log`, markers `logs/bench/status/` |
| bench full campaign (11 runs, sequential; order in `logs/bench/full.list`) | `bash logs/bench/chain.sh logs/bench/full.list full` (each line: `make bench CONFIG=<p> DATASET=<d>`, which also runs COMET) | `bench-full` | `logs/bench/<pipeline>-<dataset>.log`; progress `logs/bench/status/full.log`; markers `logs/bench/status/<run>.ok/.failed`, `full.done` |
| mock runs in parallel (user asked, 2026-09-28 08:09) | `bash logs/bench/mock_parallel.sh`: `python -m bench --config mock` for the 3 datasets, no GPU; claims `status/mock-<d>.ok` first so the main chain skips them; COMET for them runs after `status/full.done` (the qwen runs leave ~2.5 GB free, too little for COMET) | `bench-mock` | `logs/bench/mock-<d>.log`, `logs/bench/status/mock.log`, `mock.done` |
| web demo proxy | `uv run --project server python STiTy-Mobile/demo-web/partial_demo/demo_proxy.py 8080 --route en=ws://127.0.0.1:8765/api/v1/ws,ko=... --default ws://127.0.0.1:8765/api/v1/ws` | `webdemo` | `logs/server/webdemo.log` |

## Results

All 11 runs, 2026-09-28, same code (after B4), levelled mix (B5). `en`/`ko` = the
`asr.qwen-seg-{en,ko}+mt.qwen3.5-4b` pipelines. LAAL/YAAL are on the audio clock; LAAL CA on the
wall clock (includes translation). Long-form runs report LongYAAL; their LAAL/YAAL are
unavailable by design. Mock output is `word1 word2 …`, so its scores only show the harness runs.

| pipeline | dataset | status | items (err/empty) | WER | CER | BLEU | COMET | FSL | LAAL | YAAL / LongYAAL | LAAL CA | lang acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | fleurs-en-ko | ok | 268 (0/0) | 0.120 | 0.071 | 19.9 | 0.845 | 0.15 s | 3.92 s | 3.64 s | 4.71 s | 1.000 |
| ko | fleurs-en-ko | ok | 268 (0/0) | 0.069 | 0.048 | 21.9 | 0.860 | 0.22 s | 4.55 s | 3.83 s | 6.03 s | 0.996 |
| mock | fleurs-en-ko | ok | 268 (0/0) | 6.319 | 3.683 | 0.0 | 0.239 | 0.00 s | 1.92 s | 1.89 s | 1.93 s | 1.000 |
| ko | fleurs-ko-en | ok | 268 (0/0) | 0.156 | 0.080 | 21.9 | 0.838 | 0.23 s | 3.94 s | 3.83 s | 4.60 s | 1.000 |
| en | fleurs-ko-en | ok | 268 (0/0) | 0.166 | 0.086 | 22.4 | 0.844 | 0.20 s | 4.09 s | 3.99 s | 4.72 s | 0.978 |
| mock | fleurs-ko-en | ok | 268 (0/0) | 5.584 | 10.511 | 0.0 | 0.269 | 0.00 s | 2.06 s | 2.04 s | 2.07 s | 1.000 |
| ko | fleurs-ko-en.cafe | ok | 268 (0/0) | 0.176 | 0.089 | 21.5 | 0.826 | 0.22 s | 4.02 s | 3.90 s | 4.62 s | 1.000 |
| ko | fleurs-ko-en.hall | ok | 268 (0/0) | 0.201 | 0.108 | 20.9 | 0.820 | 0.20 s | 4.26 s | 4.16 s | 4.86 s | 1.000 |
| ko | fleurs-ko+en-en | ok | 44 (0/0) | 0.125 | 0.077 | 21.8 | 0.832 | 0.19 s | — | 3.72 s (long) | — | 0.941 |
| en | fleurs-ko+en-en | ok | 44 (0/0) | 0.268 | 0.197 | 2.1 | 0.604 | 0.17 s | — | 4.59 s (long) | — | 0.504 |
| mock | fleurs-ko+en-en | ok | 44 (0/0) | 3.459 | 5.447 | 0.0 | 0.268 | 0.00 s | — | 1.20 s (long) | — | 0.544 |

Reading it:
- Clean → cafe → hall on the same 268 Korean sentences: WER 0.156 → 0.176 → 0.201, COMET 0.838 →
  0.826 → 0.820, YAAL 3.83 → 3.90 → 4.16 s. Worse, not collapsed; augmentation applied (B3 caveat:
  the augmented runs are also louder).
- Wrong-language models: the English model on Korean audio is barely worse (WER 0.166 vs 0.156)
  and the Korean model on English audio is *better* than the English model (WER 0.069 vs 0.120,
  COMET 0.860 vs 0.845) because the English finetune appends made-up sentences. On the mix the
  English model fails the way a wrong-language model should: it writes Korean but labels it `en`,
  so Korean turns go out untranslated (BLEU 2.1, COMET 0.60, language accuracy 0.50).
- Every run: status ok, 0 errored, 0 empty transcriptions, COMET present (not in `unavailable`).
- Caveats from the handoff review: `lang acc` on single-language runs is forced by the harness
  (M1); COMET on the en × mix row rewards untranslated Korean (M2), BLEU 2.1 is the honest number;
  YAAL averages over different item sets per run, compare LAAL instead (M3). Checked and fine: this
  table matches every `summary.json`; COMET input/score counts match (268 per sentence run, 132 per
  mix run); no empty translations; no markup in committed text; B1, B5 and S1–S5 hold; all 11
  runs used the post-B4 code.

### northstar-260928 (2026-09-29)

Six acted ko/en business-dinner scenes, each streamed whole (`longform`). 8 runs: both finetunes x
target en/ko x clean / restaurant+cafe (MIT restaurant IRs 0.5-1.0 + DEMAND SCAFE 0.5-1.0).
Chains: `logs/bench/northstar*.list`, markers `logs/bench/status/northstar*.log`. All numbers carry
the label problem in B7. Translation scores count only lines whose script language differs from the
target (49 English lines for target ko, 72 Korean lines for target en). `slot resets` = count of
`SLOT-RESET` log lines.

| model | target | audio | status | WER | CER | BLEU | COMET | LongYAAL | word delay p50 | lang acc | seg/scene | slot resets |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ko | en | clean | ok (0/0) | 0.375 | 0.281 | 11.0 | 0.699 | 2.56 s | 1.03 s | 0.713 | 51 | 143 |
| ko | en | restaurant+cafe | ok (0/0) | 0.562 | 0.455 | 9.9 | 0.591 | 2.44 s | 1.11 s | 0.776 | 49 | 164 |
| ko | ko | clean | ok (0/0) | 0.375 | 0.281 | 5.2 | 0.567 | 2.60 s | 1.03 s | 0.713 | 51 | 143 |
| ko | ko | restaurant+cafe | ok (0/0) | 0.562 | 0.455 | 4.7 | 0.555 | 2.60 s | 1.11 s | 0.776 | 49 | 164 |
| en | en | clean | ok (0/0) | 0.517 | 0.379 | 4.5 | 0.532 | 3.41 s | 1.04 s | 0.383 | 50 | 138 |
| en | en | restaurant+cafe | ok (0/0) | 0.696 | 0.600 | 1.7 | 0.467 | 3.55 s | 1.19 s | 0.389 | 46 | 159 |
| en | ko | clean | ok (0/0) | 0.517 | 0.379 | 9.3 | 0.753 | 2.89 s | 1.04 s | 0.383 | 50 | 138 |
| en | ko | restaurant+cafe | ok (0/0) | 0.696 | 0.600 | 10.6 | 0.720 | 2.72 s | 1.19 s | 0.389 | 46 | 159 |

- Same model, same audio gives identical WER/CER on both targets (deterministic ASR).
- Restaurant+cafe costs 18-19 WER points on both models.
- Each model is best on the target whose source it hears well: ko model for target en (COMET
  0.699), en model for target ko (0.753). The other pairing passes through the mislabelled lines.
- Language accuracy rises slightly under noise for the ko model (0.713 -> 0.776) along with more
  slot resets (143 -> 164): more resets mean more windows start on the new speaker's language.

## Open issues

From the handoff review (2026-09-29), checked against the code and the 11 FLEURS runs in
`bench/runs/<dataset>/<pipeline>/`; B7 (northstar) is above. **en** / **ko** = the
`asr.qwen-seg-{en,ko}+mt.qwen3.5-4b` pipelines. "Production too" means the same logic is in
`Qwen3-ASR/examples/streaming_websocket_server.py` (`core/components/transcription/qwen3_seg.py`
is a port of it). Per the user, bugs that exist in production are reported here, not fixed.

Most important first:
1. Four ASR guards throw away real speech or let fake speech through (A1–A4), all in production too.
   They explain most "lost words" and "made-up sentence" cases.
2. Two headline numbers do not mean what they seem: language accuracy on single-language runs is
   forced by the harness (M1), and COMET rewards Korean passed through untranslated (M2).
3. Mislabelled text is never translated (T1, and B7 for fast turn-taking).
4. Bench and production have drifted apart and nothing checks them (A10).
5. On real conversation (northstar, N below) speech segments rarely end, so one decode window spans
   several speakers; committed text is silently lost when a re-decode rewrites it (N5); the en
   model invents fluent English for Korean (N3).

### A. ASR commit logic (`core/components/transcription/qwen3_seg.py`)

**A1. The no-speech gate looks at the wrong time window** — high, production too.
- Made-up sentences over pure silence get committed: en_1755 commits `I'm sorry, sir. I'm afraid I
  can't help you.` at 6.8 s; the speech ended at 4.67 s.
- `_is_silence_hallucination` checks for VAD speech between the *last final* and now
  (`qwen3_seg.py:1030`, production `:2986`). If the last commit came before the speech ended, that
  window still contains the speech, so the gate passes.
- Commits with no VAD speech since the last VAD reset: en × en-ko 15 (all made-up phrases),
  en × mix 50, cafe 9, hall 4, the rest ≤ 1.
- Fix idea: measure from the later of the last final and the last VAD reset.

**A2. Held comma fragments are deleted when joined at a flush** — high for English, production too.
- A short fragment like `Needless to say,` is held to join the next sentence; at a VAD flush the
  joined text loses the fragment.
- `_advance_cursor` records the fragment as committed (`_remember_committed`, `qwen3_seg.py:962`)
  before `_merge_comma_fragments` holds it (`:996-999`). At flush the joined text goes through
  `_apply_commit_guards`, and `committed-prefix` / `cross-dedup` trims the fragment as a repeat
  (`:1087-1091`). Production has the same order (`:1463-1476`, `:1905-1925`).
- All 5 English flush-joins lost their fragment: en_1845 `In a carriage,`, en_1946 and conv0024
  `Cancellation policies vary,`, conv0004 `Needless to say,`, conv0026 `However,`. Korean survives
  only because `fuzzy_key` strips Hangul.
- Fix idea: remember a fragment only when it is actually emitted.

**A3. `seg-boundary-dedup` drops real sentences** — medium, production too.
- en_1877 ends at `…in La Plata.`; the next sentence `A city fifty kilometers … Buenos Aires` is
  dropped (WER 0.34, in both runs on that audio).
- The rule drops the whole next unit when the previous last word *ends with* its first word, with
  no minimum length (`qwen3_seg.py:646-648`, production `:1824`); `plata` ends with `a`. A
  silence-phrase commit that was itself dropped is still stored as the reset's last commit, so
  conv0016 lost `At the other end of the spectrum…` (`that` ends with `at`). en_1682 lost a unit
  because only its first word repeated.
- Fix idea: require an exact match (or a real overlap of several words); don't record dropped
  commits as the last commit.

**A4. Words already shown in partials never get committed ("lost second halves")** — medium–high,
production too.
- Path 1 (the "Lost second halves" note under Run checks): a dangling-language-header reset carries
  audio only into `audio_accum` with an empty buffer (`qwen3_seg.py:492-499`, `555-560`). If VAD
  fires on the same chunk, `flush()` re-decodes only when there is earlier text, buffered audio or
  a short utterance (`:1200-1204`), none of which holds, and `_decode()` decodes only the buffer
  (`qwen3.py:155-157`). Items: en × en-ko en_1844, en_2001; ko × en-ko en_1673, en_1738, en_1844,
  en_1989; en × ko-en ko_1944; en × mix conv0011.
- Path 2: when the audio since the last commit is longer than 3 chunks, the carry keeps only the
  last 2 s (`qwen3_seg.py:495-497`, production `:1255`), so words the partial showed are lost even
  without VAD firing. en_1813 loses `about half prefer to speak Catalan, a vast majority
  understands it and virtually`; en_1960 loses `In a statement, Bieber said … tragic accident`;
  also en_1718, ko_1709.
- Header resets whose shown words never reached a commit: en × en-ko 3, ko × en-ko 7, en × mix 3,
  ko × ko-en 1.
- Fix idea: on a reset, carry the full uncommitted audio (or commit the shown partial), and make
  `flush()` decode `audio_accum` too.

**A5. Repetition loops wipe out the following turn** — medium–high, production too.
- Detector: `_cut_repeat_hallucination` (`Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:66-77`, added
  2026-05-30, `6d0d2839`) only cuts when one whitespace token repeats 4× in a row.
  `아니, 아니, 아니. <SEG> 아니, …` alternates tokens and never trips it.
- What loses the next turn: the skipped loop still moves `committed_display` /
  `committed_seg_count` forward over the junk (`qwen3_seg.py:936-961`). Later decodes start fresh
  and don't begin with that stored text, so `committed_cursor` returns -1 and `uncommitted_from`
  returns `""` (`:187-202`, `:210-219`). Nothing is shown or committed until VAD, and at the VAD
  flush `_trim_tail_silence` has emptied the buffer, so nothing is decoded then either.
- Kinds of loop: filler (`아니, 아니.`: conv0024, conv0043, ko_1887, ko_1902, ko_1931), sentence
  repeats (conv0001, conv0038, ko_1848, en_1863, ko_1787), header loops (`<SEG>language None.`
  en_1718, `<SEG>language Korean` ko_1946 — the worst: it loses `3월 말 현재 코로나 … 연장될` that
  the partial already showed). conv0024's loop started on the silence at 9.2 s and wiped the
  English turn at 11.2–19.9 s. ko_1902 and ko_1931 lose the whole sentence.
- Items with ≥ 5 `DEDUP-SKIP`: cafe 1 (ko_1848), hall 5 (ko_1848, 1887, 1902, 1931, 1946),
  ko × mix 5 (conv0001, 0024, 0025, 0038, 0043), ko × en-ko 2 (en_1718, en_1863), en × ko-en 1
  (ko_1787), others 0.
- Fix idea: compare repeats after stripping punctuation and `<SEG>`; don't advance the committed
  cursor over skipped text.

**A6. Canned made-up phrases pass the silence-phrase filter** — medium, production too.
- The en finetune appends `I want to buy a new car.`, `I'm sorry, sir.`, `I'm afraid I can't help
  you.`. `KNOWN_SILENCE_RE` (`qwen3_seg.py:49-58`) must match the whole candidate and covers only
  `I'm sorry, but I can't hear/help/understand you…`, a bare `I'm sorry,…`, and
  `I want to see the movie…`/`I want to know…`.
- en × en-ko: 36/268 items commit at least one (51 commits); `I want to buy a new car.` twice in 9
  items (en_1660, 1765, 1826, 1911, 1916, 1949, 1963, 1973, 1976). en × mix: 29 items, 64 commits.
- `REPEAT-COLLAPSE` keeping two copies is intended (production `_collapse_repetition(max_repeats=2)`);
  it was never a hallucination filter. A1 is the better fix; extending the regex is a patch.

**A7. Filler hallucinations are committed and translated** — medium.
- `아니, 아니.` alone is committed 33 times in cafe (translated `No, no.`). Filler commits: cafe 58
  in 55 items, en × ko-en 27 (incl. `啊。` 7×, `다.` 9×), hall 17, ko × mix 17, ko × ko-en 11.
  Mostly at the VAD flush after a seg reset leaves a short tail decoded alone (ko_1677).
- `KNOWN_SILENCE_RE` / `KO_TAIL_FILLER_RE` don't cover `아니`, `응`, `그렇지`, `그거`.
  `UTTERANCE-RETRY` also commits filler (ko_1766 `그거 뭐`, ko_1820 `으으으.`, ko_1999 `그렇죠.`).

**A8. A language-header list reached a commit** — medium.
- ko_1829 (en × ko-en): the model looped `language Korean, English, Chinese, Japanese, …`;
  `strip_lang_headers` removed only the leading `language Korean,`, and `English, Chinese, Japanese,
  Korean, …` ×4 was committed and passed through as the "translation". The real
  `특히 주류나 담배를 살 때 그렇다` was lost (WER 0.76).

**A9. Partials show language headers** — low (UI only), production too.
- `_offer_partial` → `_uncommitted_display` never calls `strip_lang_headers`, so partials show
  `language English …`: en × en-ko 146 partials in 75 items, ko × ko-en 190/80, hall 186/76. Single
  `l` stubs in 36–144 items per run. Committed text is clean (0 records with `language X` or `<SEG>`).

**A10. Bench and production have drifted, and nothing checks them** — medium (process).
- The parity test does not exist: `core/CLAUDE.md:113` points to
  `Qwen3-ASR/tests/test_qwen_seg_parity.py`; the directory has only `test_dot_commit_boundary.py`,
  `test_final_residual_commit.py`, `test_streaming_architecture.py`.
- The B4 fix (header-only re-decode guard, `qwen3_seg.py:1213-1215`) is bench-only; production
  still compares raw strings (`streaming_websocket_server.py:2555`
  `if not _post_uncommitted and _pre_uncommitted`).
- Production splits the chunk at the VAD sample (`:2496`); bench transcribes the whole 200 ms
  chunk, then flushes (`core/pipeline/stages.py:28-34`).
- Any fix to A1–A9 has to go into both, until one of them calls the other.

### T. Translation and the pipeline

**T1. Text labelled with the target language is never translated** — high for mixed audio.
- `stages.translate` sets `translation = original` when `item.language == target_lang`
  (`core/pipeline/stages.py:49-50`); the Qwen3.5 translator does the same (`translator.py:75-76`),
  and so does production's local translator (`core/translator/local_translator.py:156, 399`). No
  script check anywhere in `core/`, `bench/` or `server/app`.
- en × mix: 279 of 286 Korean-turn commits are labelled `en`; 555 of 562 "translations" equal the
  source, including 177 Hangul commits sent to English readers. 48 of the 279 are the English model
  translating Korean speech into English itself (conv0000 `Madagascar is currently…`).
- northstar: the same skip, plus the per-window label (B7), gives 125/298 Korean lines in English
  output for the en model and English left in Korean output for the ko model.
- The only script check is the web demo proxy (`STiTy-Mobile/demo-web/partial_demo/demo_proxy.py`
  `fix_direction` :311, `script_lang` :355), not the mobile app.
- Fix idea: a script check in `stages.translate` (Hangul labelled `en` → treat as `ko`).

**T2. Translation failures are invisible** — medium.
- `Qwen35Translation.translate` catches every exception, logs `TRANS-ERROR` and returns `("", "")`
  (`translator.py:78-82`). Nothing counts these in `summary.json`, the run still ends `ok`, and the
  server's `/healthz` always says `ok` (`server/app/services/health.py:10`) even if vLLM died. (No
  TRANS-ERROR in the 11 runs.)
- Fix idea: count failures in the summary and fail the run above a threshold; have healthz ping
  the translator.

**T3. Bench and server translate differently** — medium.
- Bench's `CascadePipeline._translated` doesn't pass `allowed` and doesn't catch errors; the
  server's `TranslationProcessor.process` does (`core/pipeline/cascade.py:35-42` vs `:78-91`). Bench
  should drive the same processor.

**Model quality (not code bugs, but they cost points).**
- Qwen3.5-4B writes Chinese into Korean output: en × en-ko 18 items (en_1663, 1695, 1749, 1757,
  1763, 1845, 1847, 1911, 1925, 1929, 1930, 1948, 1951, 1961, 1963, 1969, 1973, 1996; en_1695 also
  Arabic `디مر`); ko × en-ko 17 items (e.g. en_1712 `동물원去过`). English left untranslated:
  en_1786 `FTIR.`, en_1994 `Needless to say…` in both runs.
- ASR in the wrong script: ko_1678 `세克斯` in 3 Korean-audio runs; ko_1992 has a Georgian
  character (`머ჩ`); en × ko-en has 10 commits in Chinese (`啊。` in ko_1796, 1800, 1849, 1900,
  1904, 1923, 1993, 2009; `如果他` ko_1949; `对吗？` ko_1989), translated into `Ah.`, `If he`,
  `Is that correct?`.
- The ko finetune invents text: ko_1833 replaces the sentence (all 3 commits at 2.0 s, nothing for
  the remaining 14.5 s); ko_1709 (WER 1.4) and ko_1813 insert invented sentences between real ones;
  ko_1917 (hall).
- The ko model beats the en model on English audio (WER 0.069 vs 0.120, COMET 0.860 vs 0.845),
  mostly because of A6. Worth knowing before choosing which server English speakers use.

### M. Metrics — numbers that mislead

**M1. "Language accuracy" on single-language runs is forced by the harness** — high (reading).
- `qwen3.py::_allowed_language_names` restricts the model's language header to the dataset's
  languages; `restrict_languages` defaults to true (`qwen3.py:72`, `:136-149`). So 590/590,
  520/520, 0.978 and 0.996 mostly reflect the lock. It also props up "the English model writes
  Korean well" (WER 0.166 on ko-en). Only the mix runs (and northstar), which allow both
  languages, measure detection.
- ko × mix: see its run check (label-vs-script agreement 100%; the 4 "Korean labelled `en`" are
  time-window artifacts of `bench/metrics/asr.py::_detections`).

**M2. COMET rewards untranslated copies** — high (reading).
- en × mix: COMET rows whose output is mostly Hangul (Korean left untranslated) average 0.637;
  mostly-English rows average 0.534; the mock floor is 0.24–0.27. COMET 0.604 hides the failure;
  BLEU 2.1 is the honest number.

**M3. YAAL averages over different items per run** — medium.
- YAAL drops items whose first output comes at or after the end of the source. Dropped: en × en-ko
  26, ko × en-ko 49, en × ko-en 9, ko × ko-en 8, cafe 9, hall 10. So "YAAL 3.83 vs 3.64 s" for ko vs
  en on English audio isn't comparable; LAAL (4.55 vs 3.92 s) is.
- 10–15 per run are dropped only because commit times sit on 200 ms chunk boundaries (en_1850
  committed at 18.0 s for a 17.98 s clip). In LAAL the same edge makes an item count as its full
  duration (en_1850 17.98 s, en_1733 17.58 s) while 0.1 s earlier counts about half. The LAAL CA
  outliers (ko_1961 ≈ 19 s in every ko run, en_1961 14.8 s) come from this rule.

**M4. Hallucinations give negative latency** — low.
- ko_1833 LAAL −2777 ms; conv0041 LongYAAL −18.6 ms (en × mix). The lowest-latency items in each run
  are hallucinations (ko_1813, ko_1709, ko_1917, ko_1902), which pulls means down.

**M5. Mixed-set per-language WER is per talk, not per language** — low–medium.
- `per_lang` groups by the row's `src_lang`, which for a talk is `en+ko`/`ko+en`
  (`bench/metrics/common.py:31-40`), so `wer_by_lang` keys are talk orders and `wer` averages them.
  Mixed rows also use the basic normalizer, so English numbers/contractions aren't normalized
  (`bench/metrics/transcription.py`).

**M6. Wall-clock metrics carry harness artifacts** — low.
- bench feeds audio one chunk at a time and waits for translation inside `listen()`
  (`bench/__main__.py:75`, `cascade.py:72,78-91`); the server runs them in separate workers.
  Measured effect on commits is a few ms on average (B6).
- `fsl` is stamped before the commit's own translation (`qwen3_seg.py:1072`), so it only includes
  waits caused by earlier chunks.
- A partial's `t` is stamped after the whole `listen()` returns (`bench/__main__.py:54-59`), so it
  includes the translation of any commit in the same chunk; commits use `committed_elapsed_sec`.
  `token_emission_ca_ms` is biased late for partials only.
- Partials are throttled on wall time (`PARTIAL_MIN_INTERVAL_SEC = 0.12`, `qwen3_seg.py:22,596`), so
  back-to-back catch-up chunks lose partials and `token_emission_ms` depends on machine speed.
- `laal_ca − laal` isn't pure compute: audio-clock emission is capped at the source length, the
  wall-clock one isn't (`latency.py:81` vs `:83`); 103/590 commits (en × en-ko) land after the
  audio ends.
- `realtime_factor` divides by audio + 4 s trailing silence per item (by design, `bench/report.py`);
  against audio alone it's ~0.14, not ~0.10.

### H. Bench harness

**H1. The translator's vLLM server is never stopped properly** — medium.
- Every model run ends with `WARNING CLOSE-FAILED Qwen35Translation: Event loop is closed`. `main`
  loads and runs the pipeline in one `asyncio.run` (`bench/__main__.py:185`) and closes it in a
  second (`:193`); `close()` calls `aclose()` on an httpx client bound to the first, closed loop.
  The vLLM child is kept only in a local variable (`translator.py:139-141`), so
  `ManagedProcess.stop()` is never called; the only cleanup is the `atexit` SIGTERM to `uv`, which
  doesn't wait.
- Risk in chained runs: if the old vLLM still answers on :8100, the next run reuses it
  (`_start_server_unless_running`); when it dies, every translation silently becomes `""` (T2) and
  the run still ends `ok`. The last FLEURS run reported 1 GiB less free GPU memory at start than
  the other 7 (13.45 vs 14.47 GiB) — possibly a lingering process, unverified.
- Fix idea: close the pipeline inside the same loop (try/finally in `_run`), keep the process on
  the translator and `await stop()` in `close()`.

**H2. Unexpected exceptions leave no summary** — medium.
- `bench/__main__.py:184-193` catches only `STiTyError` and Ctrl-C. A CUDA/httpx/scoring error
  leaves no `summary.json` and no `run_close`. Fix idea: catch `Exception` → `status: failed`,
  write the report, re-raise.

**H3. `item_close` events have `t: null`** — low.
- `stop_clock()` runs before `record("item_close")` (`bench/__main__.py:79-81`).

### V. Server (`server/app`) and clients

- Not implemented (not bugs): HTTP conversation create, join tickets, late-join backlog. Only
  `GET /healthz` and `/ws` exist; a new participant only gets future events
  (`models/conversation/conversation.py:72-75`).
- Malformed messages vanish (low–medium): `_parse` (`api/websocket.py:81-89`) drops any validation
  error without a field location at DEBUG level — invalid JSON, missing `type`, or a typo like
  `"strat"`. A client with a typo in `start` waits forever. Log a WARNING or send an error.
- Two servers share one vLLM (low): both pipeline configs use `url: http://127.0.0.1:8100`, so an en
  and a ko server started together would share the first one's translator and lose translation
  when it stops (and 0.30 + 0.30 + 0.52 of the card doesn't fit anyway). `server/README.md` says
  to run one at a time; its "stops it when the server exits" overstates H1's cleanup.
- `TranslationWorker.settle` busy-waits on `asyncio.sleep(0)` while the inbox is non-empty
  (`services/conversation/workers.py:67-69`) — one CPU spins for the length of a translation.
- A dead translation worker stays subscribed with an undrained inbox (`workers.py:87`); clients
  wait for translations that never come.
- Settings: `settings_customise_sources` discards pydantic's `.env` source (`**_`); it works only
  because `load_env()` loads `.env` into the environment first.
- `server/scripts/client.py`: mock detection (`MOCK_TRANSLATION`) is set only in `case_single`
  (:251), so other cases run alone misjudge the mock.
- Mobile app: the server URL falls back to the hard-coded RunPod URL unless
  `EXPO_PUBLIC_SERVER_URL` is set at build time (`STiTy-Mobile/src/context/WebSocketContext.tsx:34`).
  Not tested on a phone.

### N. northstar-260928 line-by-line review (2026-09-29)

Every script line of the 6 scenes (121 lines) was read against the commits and both translations
of the 4 ASR streams (ko / en finetune × clean / restaurant+cafe; the two targets share each
transcript), plus a script scan of all 8 runs' events. "Production too" is
marked only where checked; everything in the ASR commit logic is a port of the production server,
so assume it applies there unless noted.

**Overview (script scan, all 6 scenes)**

| | ko clean | ko r+c | en clean | en r+c |
|---|---|---|---|---|
| commits | 307 | 294 | 298 | 275 |
| script lines lost (<20% of the line's words/bigrams in commits near it) | 13 | 32 | 11 | 28 |
| script lines partly lost (20–60%) | 13 | 26 | 20 | 27 |
| label differs from the text's script (B7/T1) | 39 | 16 | 125 | 90 |
| filler-only commits (A7) | 19 | 26 | 6 | 7 |
| `DEDUP-SKIP` (loops, A5) — scenes | 21 — s1 17 | 56 — s1 37, s6 18 | 21 — s4 18 | 55 — s5 52 |
| committed-prefix units never emitted (N5) | 5 | 5 | 8 | 5 |
| DROP rules fired | tail-filler 5, no-speech 1 | 7 | silence-phrase 16, tail-filler 5, no-speech 2, header-only 2 | silence-phrase 8, no-speech 7, tail-filler 2 |
| Chinese in a translation | 0 | 2 | 1 | 0 |

Loss rate by script tag (tag meanings inferred from the names; there is no legend in the dataset):
worst are BC (backchannel) 2–5/7, OVL (overlap) 4–6/14, MUM (mumbled) 1–9/19, OFF 0–8/16. Under
restaurant+cafe, HON lines go from 0–1/16 to 7/16 lost for the en model and CS from 2–5/21 to 6–8/21.

**N1. Speech segments are too long for conversation; the window rarely resets** — high.
- Silero VAD with `min_silence_ms: 800` hardly ever sees 0.8 s of silence in fast turn-taking:
  segment median 9.0 s, p90 36.4 s, max 89.6 s (clean); scene5 had one segment 3.07–92.64 s.
  Resets come almost only from `<SEG>` (seg 79–100 per stream vs vad 50–56), so one decode window
  holds several speakers and languages — the direct cause of B7's per-window label.
- Under restaurant+cafe, VAD missed the start of scene1: first speech at 15.1 s, script starts at
  5.0 s (commits still came from `<SEG>`). 3/121 lines are <50% covered by VAD speech (0 clean).

**N2. Wrong label → the translator rewrites same-language text and changes its meaning** — high.
Mirror of T1: Korean labelled `en` with target `ko` (or English labelled `ko` with target `en`) is
sent as "English → Korean" on Korean text. The LLM paraphrases it. Mostly the en model: ~20–25
commits per scene pair.
- Numbers: `다섯 명이요, 김태호예요.` → `네 명이에요…`; `시월 십오일` → `1 월 15 일`; `이억삼천` →
  `3 억 3 천` / `1 억 3 천`; `총 십팔만 칠천오백 원입니다.` → `총 일흔 팔만 칠천오백 원입니다`;
  `지난 석 달` → `지난 한 달`.
- Negation / meaning: `삼겹살이랑 목살은 하나도 안 매워요.` → `…전혀 매워요.` (both en streams);
  `먹으면서 말하지 마요.` → `말하면서 먹지 마세요.`; `부장님 고향이 부산이세요.` → a question;
  `여보세요?` → `안녕하세요?`; `위하여로 받아 주시기` → `모두를 위해 기꺼이 받아…`.
- The en model writes Korean numbers as words (`시월 십오일`, `이억삼천`), which is what gets
  mis-converted.
- Fix follows T1/B7: label per sentence (by script) so the translator gets the right direction.

**N3. The en model invents fluent English for Korean speech** — high, worst under noise.
Beyond T1's "ASR translates Korean itself": the English is plausible and false, and the translator
carries it faithfully into Korean.
- s5 35.8 (r+c): `부산은 KTX로 두 시간 반…` → `Do you want to pay cash or do you want a credit card?`
- s5 41.8–43.8 (r+c): `해운대 가가 회 한 접시 드이소…` → `you have to go to the seafood market…`,
  `The one in Ganghwan is famous for its fried fish.`
- s4 36.4 (r+c): `부장님 가시면 가는 거죠.` → `I think the store manager is a woman.`; s4 70.0
  `목살 좀 더 시켜야 되나…` → `You should eat more fish.`; s6 41.8–56.2 (r+c) `I'll go in the room
  and turn on the light.`, `I will never buy this bag again.`, `Twenty-eight million seven thousand
  five hundred dollars.`
- Clean: s3 `아, 괜찮아요` → `Oh, it's alright, Mr. Guan.`; s1 `네, 이쪽으로 앉으세요.` → `Please sit
  this way.` (right), s6 `다음에 미국 가면 그때 마이크 씨가 사세요!` → `…then buy a new one.` (wrong).
- Counted by reading: ~2 (clean) and ~10 (r+c) in scenes 5–6 alone.

**N4. The ko model writes whole English sentences in Hangul** — high (ASR model).
- s2 124.8–130: `웨이 웨이.` / `230 밀리언.` / `에스 원, 라이.` / `플리스 텔 미디얼스 원.` →
  `Es One, Ray.`, `Fleece Tell Medials One.`; s2 161.8 `이 니는 엇 이 업데이트 데크 아이 센드스 모닝.`;
  s5 `마이크 나이 하드 위켄드 프리.` → `The mic is hard weekend free.`; s6 (r+c) `어나쓰리, 디스 왓스
  디스 왓스, 러플리, 치어.`; s3 `디스 랩스. 디스 압소리에 슬랩스.`
- Counts (whole phrases): s1–2 11 / ~5, s5–6 7 / 12 (ko clean / ko r+c).
- The label (`ko`) matches the script, so B7's label checks and `lang_detect_accuracy` miss these;
  for target `ko` they pass through as nonsense Hangul.

**N5. A re-decode rewrites the committed prefix and a real unit is never emitted** — high.
The cursor counts committed `<SEG>` units (`committed_seg_count`). When a later decode merges,
splits or rewrites units that were already committed, the next real unit is counted as done. No
DROP or DEDUP line is logged. Same family as A5's cursor desync, different trigger.
- Verified count (units in a `REASSEMBLY committed_text` that match no commit and no DROP/DEDUP
  text, ≥4 characters): ko clean 5, ko r+c 5, en clean 8, en r+c 5.
- `3시 반이요?` (s2 152.6) is lost this way in **all four** streams.
- en r+c s4 66.4: commits `그럼 먼저 들어가 봐요.` + `얘가 아픈데.`; the next decode merges them, so
  `아, 아니에요, 괜찮습니다.` is never committed. ko r+c s4: laughter `아하하하` re-decoded as `2차요.`,
  which is then never emitted.
- ko clean s6: committed `대신 2차는 아마 아 2차 2차 2차였습니다.` ("was"), the revised `…없습니다.`
  ("there is none") never went out — meaning flipped. `Thank you` → revised `Thank you so much,
  Mr. Kim.` lost the tail.
- Held-fragment variant: ko clean s1 22.6–26.6 `COMMIT-HOLD '어우,'`, the next decode rewrote the
  first unit as `Well, we didn't have to queue.`, counted as committed, so only `어우, So, someone
  clearly pulled some strings.` went out.
- Fix idea: track the committed text, not the unit count; re-anchor by text after a revision.

**N6. A `repetition-loop` reset fires on real speech and discards the rest of the window** —
medium (production too: `hallucination_detected`, server :1230).
- ko clean s4 56.6 `SLOT-RESET cause=repetition-loop` though the logged text has no loop; D's
  `응, 응. 나 아홉 시 반까지 갈게. 응, 끊어.` (54.3–57) is lost, 9:30 with it. Likely trigger: the
  backchannel `응 응 응 응` (4 identical tokens) — the logged text is after the cut, so unverified.
- The opposite of A5 (detector misses loops): here it fires on a genuine backchannel, and
  `_recover_from_hallucination` resets without carrying the audio.

**N7. A dot-switch carry re-commits the sentence just committed** — high for the instance.
- ko clean s4 64.6 `SLOT-RESET cause=dot carry_sec=2.00`, then 66.6 commits `괜찮습니다.` again
  (the script says it once). The 2 s carry has no dedup against the last commit. Frequency not counted.

**N8. The last words before a VAD flush are lost** — medium (mechanism unverified).
- `My favorite thing in the whole world` → `…in the whole` (s2 43.2, 3 of 4 streams); `Whichever one
  is the… not-spicy one` → `Whichever one is the not` / `of them is spicy.` (s5 131.0, all four;
  alignment has `notspicy` 129.29–129.85, before the flush) — a negation flip.
- Candidates: `_trim_tail_silence` cutting 0.7 s before the final decode, or the flush committing
  the last partial without re-decoding. Related to A4 but without a header reset.

**N9. Speakers or languages merged into one commit** — medium.
- No `<SEG>` at a speaker change: s2 140.6 `테스트는 꼭 필요한 기종만 하는 걸로 합시다. 맥센스.`
  (C + B's "Makes sense").
- `COMMIT-JOIN` glues a held Korean fragment to English: en clean s3 42.2 `자, 잔도 채우시고, Ms. NMC,`
  labelled `en`, passed through in both targets.

**N10. Empty language label** — low (cause not traced).
- 8 commits across the 4 streams have `language=''` (e.g. ko clean s5 51.0 `Because I saw a
  video…`). The translator then gets no source and both targets translate — harmless here, but
  `state.language` and `last_text_lang` should not both be empty mid-speech.

**N11. Translator (Qwen3.5-4B, `context_turns: 1`) errors on correct input** — model quality.
- Negation / polarity: `I'm really not.` → `저는 정말로 그렇습니다.`; `from er just over seven` →
  `7% 미만`; `Yeah, no.` → `아니요.`
- Idioms literal: `pulled some strings` → `권력을 휘두른`; `ballpark` → `야구장`; `When in Rome` →
  `로마에 있을 때면`; `발등에 불 떨어졌거든요` → "the heat fall on our heads"; `배보다 배꼽이…` literal;
  `This slaps` → `이거 진짜 잘 나가`.
- Numbers, dates, names: `Two hundred thirty million` → `이백삼십만 명`; `세 시 반이요?` → "Is it the
  third half?"; `지난 석 달` → "past six months"; `3차, 4차까지` → "3rd or 4th floor"; `하준이 열 나?`
  → "Is Yeol-nah there?"; `수서역` → "Sujeok Station"; digit strings `zero, one, zero, one, two…` →
  `영, 자, 영, 자…` / `영원한 사랑…`.
- Honorifics / register: `Mr. Kim` → `김 씨` every time (reference `김 부장님`); casual speech to a
  senior (`진지하게 말해, 내가 강요해.`, `너 이렇게 흥분한 건 처음 봤어.`); `Is she your aunt?` →
  `그녀가 당신의 오빠의 언니인가요?`; `친정엄마` → "my mother at home".
- Script leaks: English words left in Korean (`젊음을pretend하고`, `upstairs 에`); Chinese
  (`우리这边数据不好…`, `김 씨.今夜는…`, `…를但我们只有…`).
- Previous line's translation returned once (s2 ko r+c 140.6 `Make sense.` → the translation of the
  line before) — the only exact case across 8 runs.

**Known issues seen again (ids above)**
- A1/A6: `I'm sorry, sir.` + `I'm afraid we're out of that size.` at 2.0 s before anyone speaks
  (en, both audios); `I want to buy a new car.` ×3 in s1–2; `I'm sorry, sir.` at 111.8 s after the
  audio ends (s6); `I'm sorry, I'm not sure what you mean.` and `How may I help you?` are not covered
  by `KNOWN_SILENCE_RE`. Filters did catch `I want to know the weather in Beijing.` and others.
- A3: E's `원이요, 원.` dropped by `seg-boundary-dedup` in 3 of 4 streams.
- A5: ko r+c s1 `안녕하세요.` ×19 loses C's `다섯 명이요. 김태호로 예약했습니다.`; ko r+c s1 50.6
  `아 정말로.` ×18 loses A's and B's lines 39.5–47.3; en r+c s5 `I thought that's it.` ×18,
  `That's it.` ×18, `What?` ×19; en clean s4 `I'm not.` ×18; ko r+c s6 `오,` ×17.
- A7: `아니, 아니.`, `그렇지.`, `응.`, `그거`, `아 음` committed and translated.
- A10 (whole 200 ms chunk after the VAD sample): a fragment of the next speaker's first word at
  `vad` flushes — `뭐 뭐` / `Would you` / `The` / `라이트.` — 4, 8, 10, 15 per stream.
- B7 / T1: see B7. Also counts above.
- H1: `CLOSE-FAILED Qwen35Translation: Event loop is closed` in all 8 runs.

**ASR word errors (model quality; all streams unless noted)**
- Names: 지은 → `Ji Hyun`/`G-N`/`Jin`/`Jane`; Mr. Kim → `Mr. Campbell`/`Mr. Cameron`/`미스터 캔`;
  Whitfield → `Woodfield`; 다온 → `다운`/`Da Hong Gua`; 엠마 씨 → `Ms. NMC`; 부장님 → `부상님`/`부담님`.
- Words: `won` → `one` in every en stream (`그건 하나 맞죠?`); `beta numbers` → `better/bad numbers`;
  `last fortnight` → `last four`; `EOD` → `음 25일`; `이모님` → `임원님`; `해열제` → `해결책`/`해물찌개`;
  `응급실` → `룩실`; `산낙지` → `삼나끼`; `카톡` → `텃독`/`핫톡`; `임금이 왕이잖아요` → `임금이 아니잖아요`
  (all four; kills the joke); `안 지세요` → `안 드세요`.
- Dialect (DIA) survives partly: `드이소` → `드리소` ("I'll bring you"), `직입니더` → `찍입니다`.
- Numbers under noise: `010 1144` (5678 lost); `58만 7천 5백 원` vs `18만` clean.

**Dataset / reference problems (northstar-260928)** — the reference is the script, not a verbatim
transcript. Where all four streams agree against it, the audio probably differs (not verified by
listening). These count as errors against every pipeline.
- Numbers swapped: s1 script `삼겹 3인분에 목살 2인분`, all streams hear 2 and 3 (a NUM line).
- `JMT` (s3): every stream hears `존맛탱`-like Korean and A answering `John matting…?`.
- Lines said differently: s1 `삼겹 셋` → `삼겹살,`; s2 117.4 `대략 이천삼백만 원 정도 추가돼요` →
  `…뭐 한 2300만원 정도 추가되겠죠. 예산이 어차피 2억3천이니까`; s2 108 an unscripted stumble
  (`The ballpark, ball… ah, sorry, the ballpark`); s5 `야구 보느라…`, `you're such a lifesaver`;
  s6 `카톡도 이 번호로 돼 있어요` → `…어차피 이 번호라서 보실 수 있어요`, `보통 제일 윗사람이 내요` →
  `보통 윗사람이 좀 내죠`, `2차 없습니다!` → `2차 2차 2차 없습니다`, `완벽해요! 파이팅!` →
  `완전 완벽합니다… 한국어 천재!`.
- Unscripted speech: staff lines at the start of s1 (`몇 명이세요?`, `네, 이쪽으로 앉으세요.`,
  `안녕하세요`), a cashier `총 18만 7천 5백원입니다.` in s6 (~67 s), `Cut!` after the last line of s4,
  laughter (`하하하`, `아하하하`), `안녕.` after s6's last line.
- Overlap lines (`D+E 위하여!`, `A+B Wee-ha-yuh! Cheers!`) are one reference line each and are never
  recovered by any stream.
- Number formats are mixed in the references (`이십 년`, `사백오십 명` vs `10월 15일`, `3인분`); the ko
  model writes digits and the en model writes words, and the WER/CER normalizer
  (`whisper_normalizer` `BasicTextNormalizer`, `bench/metrics/common.py`) does not unify them — each
  model is penalised on different lines.
- No tag legend in the dataset (`dataset.yml` lists speakers and scenes only).

### D. Docs that are wrong

- `bench/README.md:170-172` says the FLEURS en/ko sets have 270 sentences; they have 268 (B1). The
  install line (`:14`) may not reproduce the current `translations: [de, …]` build.
- `core/CLAUDE.md:113` describes a parity test that doesn't exist (A10).
- `server/README.md` says the translator is stopped when the server exits (H1).

### Housekeeping

- `bench/runs/mock-20260912T142448` is an old run with no summary; safe to delete.
- 11 old-layout run directories (`bench/runs/<pipeline>-<dataset>/`, e.g.
  `asr.qwen-seg-en+mt.qwen3.5-4b-fleurs-ko-en`) are left over, empty or holding only a
  `summary.json`; the runs themselves are under `bench/runs/<dataset>/<pipeline>/`.
