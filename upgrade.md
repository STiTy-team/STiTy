# STiTy Architecture Upgrade Report

Sep 29, 2026 · @Hyoungjoo Jin

## Summary

Eleven changes are proposed, ordered by priority tier and then by build complexity. The first two target defects measured in STiTy's own ACL 60/60 runs: SEG-commit p90 first-token latency of 11.8–13.9 s with a 48.9 s maximum commit gap, and lower COMET than punctuation commits.

**Priority tiers.** P0 fixes a defect measured in the STiTy repository's evaluation results. P1 covers a conversational situation in the project scope (domain terms, turn endings, noise, unclear speech, overlapping speakers) with published evidence. P2 is required for hands-free speech output and consumes the output of P0 components.

**Complexity tiers.** Low: an open-source model or method, integration only. Medium: a new model plus a new pipeline stage or client work. High: requires training data or model training.

| # | Proposal | Problem addressed | Priority | Complexity |
| --- | --- | --- | --- | --- |
| 1 | Decouple ASR stability from translation emission | One SEG token gates both source stability and translation timing; long latency tail | P0 | Low |
| 2 | Context-aware simultaneous MT with an emission policy | Fragments translated once, without context; SEG COMET below punctuation commits | P0 | Medium |
| 3 | Contextual biasing and terminology memory | No biasing input for names and domain terms in the current pipeline config | P1 | Low |
| 4 | Semantic end-of-turn detection | Fixed 800 ms VAD silence timer decides turn ends | P1 | Low |
| 5 | ASR-safe noise handling (dual audio path) | Denoised audio can raise ASR error rates | P1 | Medium |
| 6 | Reference-free quality estimation gate | No signal for when a translation is unreliable | P1 | Medium |
| 7 | Streaming diarization and multitalker ASR | Overlapping speakers merged or misattributed | P1 | High |
| 8 | Streaming TTS and output rate control | No speech output; lag accumulates with fast speakers | P2 | Medium |
| 9 | Capture layer: echo cancellation, multi-mic, ducking | Played-back translation re-enters the microphone | P2 | Medium |
| 10 | Target speaker extraction | Competing voices in one audio channel | P2 | High |
| 11 | Training for simultaneity | Offline-trained models run in streaming mode | P2 | High |

## Current STiTy baseline

On ACL 60/60 dev (57.6 min, 468 reference sentences, MADLAD-400-3B translator), SEG commits give the lowest median latency and ASR WER but the longest tail; punctuation commits give the highest COMET at the highest latency. Numbers below are copied from the repository's [evaluation/ast/RESULTS.md](https://github.com/STiTy-team/STiTy/blob/main/evaluation/ast/RESULTS.md).

| Commit policy | Calls/min | StreamLAAL de/ja/zh (s) | COMET de/ja/zh | FTL median, de (s) | FTL p90 de/ja/zh (s) | ASR WER, de (%) |
| --- | --- | --- | --- | --- | --- | --- |
| static@6s | 9.98 | 3.166 / 3.293 / 2.741 | 0.6632 / 0.7339 / 0.7334 | 3.849 | 6.27 / 6.65 / 5.75 | 10.15 |
| SEG | 13.09 | 6.075 / 5.879 / 5.121 | 0.7171 / 0.7519 / 0.7887 | 4.485 | 13.85 / 13.48 / 11.79 | 7.10 |
| static@10s | 6.01 | 5.316 / 5.439 / 4.546 | 0.6790 / 0.7219 / 0.7589 | 5.983 | 10.21 / 10.11 / 9.16 | 8.63 |
| static@12s | 5.02 | 6.525 / 5.878 / 5.572 | 0.6872 / 0.7261 / 0.7686 | 7.024 | 12.29 / 11.39 / 10.88 | 8.45 |
| punct | 8.17 | 9.044 / 7.353 / 8.436 | 0.7884 / 0.7939 / 0.8294 | 8.543 | 13.62 / 12.19 / 13.45 | 9.81 |

Recorded facts from the same file:

- SEG beats interpolated static at equal latency by +0.024 to +0.033 COMET, significant in paired bootstrap for all three languages.
- SEG's mean latency is driven by its tail; the maximum commit gap is 48.9 s.
- SEG needs 13.09 translation calls/min, 1.6× punct and 2.2× static@10s.

LibriSpeech test-clean, 16 concurrent clients ([paper\_result/ASR](https://github.com/STiTy-team/STiTy/tree/main/evaluation/LibriSpeech/paper_result/ASR)):

| Mode | Commit trigger | WER (%) | First-token latency (s) |
| --- | --- | --- | --- |
| mode2 | always | 7.39 | 2.07 |
| mode3 | dot + confirmation gate | 2.13 | 8.66 |
| mode4 | SEG (fine-tuned) | 2.31 | 4.70 |

Current Korean pipeline config ([asr.qwen-seg-ko+mt.qwen3.5-4b.yml](https://github.com/STiTy-team/STiTy/blob/main/configs/pipelines/asr.qwen-seg-ko%2Bmt.qwen3.5-4b.yml)): Qwen3-ASR-1.7B Korean fine-tune, 2.0 s chunks, SEG commits, Qwen3.5-4B translation with 1 turn of context, Silero VAD with 800 ms minimum silence. [LOCAL\_TRANSLATION.md](https://github.com/STiTy-team/STiTy/blob/main/core/translator/LOCAL_TRANSLATION.md) records that clause fragments are translated without context, and that Korean output register drifted to plain or written style with some models.

## Competitive landscape

Google's production system is a single native-audio model (Gemini 3.5 Live Translate, released 2026-06-09); the strongest open research systems at IWSLT 2025 and 2026 are ASR→LLM cascades.

| System | Architecture | Languages | Reported latency / quality | Availability |
| --- | --- | --- | --- | --- |
| [Gemini 3.5 Live Translate](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-live-3-5-translate/) (Google) | Native speech-to-speech model; generates speech continuously | 70+ auto-detected; Meet supports 2,000+ combinations | "A few seconds behind the speaker"; keeps intonation, pacing and pitch; claims noise robustness | Gemini Live API (public preview), Translate app, Meet (private preview) |
| [AirPods Live Translation](https://www.apple.com/ie/newsroom/2025/11/live-translation-on-airpods-expands-to-the-eu/) (Apple) | On-device Apple Intelligence; ANC lowers the other speaker's voice | 10 languages incl. Korean | Not published | iOS 26 with AirPods Pro 2/3, AirPods 4 ANC |
| [Seed LiveInterpret 2.0](https://arxiv.org/abs/2507.17527) (ByteDance) | End-to-end duplex speech-to-speech with voice cloning; pretraining + RL | Zh↔En | Self-reported \~3 s; independent ACL 60/60 run: 7.94 s En→De, 9.41 s En→Ja, 5.31 s En→Zh ([Practical Evaluation](https://arxiv.org/pdf/2606.15059)) | Volcano Engine API |
| [Hibiki-Zero](https://arxiv.org/html/2602.11072) (Kyutai) | Decoder-only transformer + streaming codec; GRPO latency training | Fr, Es, Pt, De → En | Audio-NTREX-L De→En: 3.31 s, xCOMET 79.50 ([Practical Evaluation](https://arxiv.org/pdf/2606.15059)) | Open weights (3B) |
| [MLLP-VRAIN IWSLT 2026](https://arxiv.org/abs/2606.17255) | Cascade: Parakeet ASR + Qwen 3.5 MT, adaptive black-box policies | All IWSLT 2026 directions | +5.82 XCOMET-XL over their 2025 system on MCIF En→De | Paper |
| [NeMo IWSLT 2026](https://aclanthology.org/2026.iwslt-1.23/) (NVIDIA) | Cascade: dual-mode unified ASR transducer + multilingual LLM | En→De/It/Zh, Cs→En | Low- and high-latency regimes | Paper |
| [AlignAtt4LLM](https://arxiv.org/abs/2606.03967) | Cascade: Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B + Gemma-4 E4B with AlignAtt | En→De/It/Zh | Beats IWSLT baselines for En→De/It at \~2 s and <4 s CU-LongYAAL | Open code |

The MLLP-VRAIN paper states its cascade choice was motivated by cascaded systems achieving the best IWSLT 2025 results. Google's 2025 predecessor (Gemini 2.5 Flash Native Audio) also [switches output language automatically based on who is speaking](https://blog.google/products/gemini/gemini-audio-model-updates/) in two-way conversation.

## 1. Decouple ASR stability from translation emission

**Priority P0 · Complexity Low.** The same Qwen3-ASR + forced-aligner front-end, with a longest-common-prefix commit rule, is used by AlignAtt4LLM, which beats the IWSLT 2026 baselines for En→De and En→It at \~2 s and <4 s latency.

**Problem.** In STiTy, one ASR-side trigger (SEG, dot, or VAD) both freezes the source text and sends it to translation. On ACL 60/60 the SEG policy has p90 first-token latency of 11.8–13.9 s, a 48.9 s maximum commit gap, and a German mean of 6.17 s against a 4.49 s median. Audio arrives in 2.0 s chunks.

**Proposed architecture.**

- The ASR stage re-transcribes the live tail each chunk. Words that agree between consecutive hypotheses (longest common prefix, up to sentence-final punctuation) become stable; the remaining tail may still change.
- Qwen3-ForcedAligner-0.6B assigns word start and end times to the transcript online.
- A word becomes available to translation once its aligned end time is observed, minus a hold-back margin.
- The SEG token stays as a segmentation signal, but no longer gates translation.

**How to build.**

1. Add a stability stage in `core/pipeline` between transcription and translation.
2. Lower `chunk_size_sec` from 2.0 to the AlignAtt4LLM operating points: 0.85 s (low latency) or 1.5 s (high latency).
3. Implement LocalAgreement-2: a prefix is confirmed when two consecutive updates agree on it, as in Whisper-Streaming.
4. Run Qwen3-ForcedAligner-0.6B on each hypothesis to get word end times.
5. Send stable words plus the unstable tail to the MT stage; the AlignAtt4LLM MT server protocol accepts exactly this pair.
6. Keep a buffer-trimming threshold so the re-decoded audio buffer cannot grow without bound, as Whisper-Streaming does.
7. Re-run the existing ACL 60/60 harness and report mean, median and p90.

**Effectiveness (published).**

- Whisper-Streaming (LocalAgreement) reports 3.3 s average latency for English ASR on the unsegmented ESIC long-form test set.
- AlignAtt4LLM (Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B + Gemma-4 E4B, one NVIDIA A40) outperforms the IWSLT 2026 baselines for En→De and En→It in both the \~2 s and <4 s CU-LongYAAL regimes; En→Zh results were mixed.
- CUNI's SimulStreaming (Whisper with AlignAtt or LocalAgreement) improved 13–22 BLEU over the IWSLT 2025 organizers' baseline for En→De/Zh/Ja on dev sets.

**Other notes.**

- CUNI reports that LocalAgreement reprocesses the audio buffer from the start on every chunk, which costs more compute than AlignAtt.
- AlignAtt4LLM used a 0 ms hold-back in official runs; its reliability analysis suggests a conservative 250 ms tail hold-back. It also delays the first translation until 2 s of audio has accumulated.
- Qwen3-ASR and Qwen3-ForcedAligner are released under Apache 2.0.

**Sources.** [AlignAtt4LLM paper](https://arxiv.org/html/2606.03967v1) · [AlignAtt4LLM code](https://github.com/QuentinFuxa/Alignatt4LLM) · [Whisper-Streaming paper](https://arxiv.org/abs/2307.14743) · [Whisper-Streaming code](https://github.com/ufal/whisper_streaming) · [CUNI IWSLT 2025](https://arxiv.org/abs/2506.17077) · [Qwen3-ASR Technical Report](https://arxiv.org/pdf/2601.21337)

## 2. Context-aware simultaneous MT with an emission policy

**Priority P0 · Complexity Medium.** Top IWSLT 2025–2026 cascades translate a growing source prefix with an LLM and let a policy decide how much of the draft to emit, instead of translating each ASR segment once.

**Problem.** STiTy translates each committed segment once, with `context_turns: 1`. The repository records that clause fragments are translated without context, because with context the model borrowed a verb from the previous line to finish the fragment. On ACL 60/60, SEG COMET is 0.717–0.789 against 0.788–0.829 for punctuation commits.

**Proposed architecture.**

- Each source update sends the LLM the current source prefix, the already accepted target prefix, and a fixed instruction; the LLM drafts a short continuation.
- A policy accepts the longest safe part of the draft. Output is append-only, the mode AlignAtt4LLM describes as preferred in deployments because re-translation flicker disrupts readers.
- **Black-box policy (first):** LocalAgreement on the target side; CUNI used it with EuroLLM because it needs no attention weights.
- **Attention policy (second):** AlignAtt4LLM, which accepts draft tokens only while their source alignment stays inside the words already heard.
- **Context:** a memory of prior source sentences in the prompt, as in BeaverTalk's conversational prompting.

**How to build.**

1. Change the translator interface from `translate(segment)` to `translate(source_prefix, target_prefix, context)`.
2. Implement target-side LocalAgreement: emit only the target prefix that two consecutive MT updates agree on.
3. Add source-sentence memory to the prompt; BeaverTalk used a single prior sentence.
4. Swap in AlignAtt4LLM's `alignatt-mt-server`, which takes committed source words plus the unstable tail over WebSocket and returns append-only translation deltas.
5. For a new backbone, recalibrate the top-8 alignment heads per language pair offline; AlignAtt4LLM scored heads against GPT-5-mini word alignments.
6. Emit whole words in spaced scripts and single characters in CJK, the "stability unit" AlignAtt4LLM uses so subword fragments are never shown.

**Effectiveness (published).**

| System | MT model | Direction | Latency (StreamLAAL) | Quality |
| --- | --- | --- | --- | --- |
| BeaverTalk (IWSLT 2025) | Gemma 3 12B + LoRA | En→De | 1.84 s / 3.34 s | 24.64 / 27.83 BLEU |
| BeaverTalk (IWSLT 2025) | Gemma 3 12B + LoRA | En→Zh | 2.22 s / 3.52 s | 34.07 / 37.23 BLEU |
| AlignAtt4LLM (IWSLT 2026) | Gemma-4 E4B | En→De, En→It | \~2 s / <4 s CU-LongYAAL | Beats organizer baselines |
| MLLP-VRAIN (IWSLT 2026) | Qwen 3.5 | En→De (MCIF) | Low and high regimes | +5.82 XCOMET-XL over their 2025 system |

AlignAtt4LLM cites Macháček and Polák (2025): LocalAgreement generally introduces more latency than AlignAtt.

**Other notes.**

- AlignAtt4LLM launches one MT request per ASR chunk, so call volume rises with shorter chunks (about 70 requests/min at 0.85 s chunks, against STiTy SEG's 13.09/min).
- AlignAtt4LLM reports mixed En→Zh results and names HY-MT-1.5 and MiLMMT-46 as stronger translation-focused backbones for Chinese; heads and thresholds must be recalibrated per backbone.
- STiTy's own Korean demo test measured `gemma-3-4b-it` 4-bit at COMET 0.920 and chrF 67.3, and it was the only candidate that held 해요체 register.

**Sources.** [BeaverTalk](https://arxiv.org/pdf/2505.24016) · [AlignAtt4LLM paper](https://arxiv.org/html/2606.03967v1) · [AlignAtt4LLM code](https://github.com/QuentinFuxa/Alignatt4LLM) · [CUNI IWSLT 2025](https://arxiv.org/pdf/2506.17077) · [MLLP-VRAIN IWSLT 2026](https://arxiv.org/abs/2606.17255) · [STiTy LOCAL\_TRANSLATION.md](https://github.com/STiTy-team/STiTy/blob/main/core/translator/LOCAL_TRANSLATION.md)

## 3. Contextual biasing and terminology memory

**Priority P1 · Complexity Low.** Qwen3-ASR, the model STiTy already runs, was fine-tuned to use context tokens in its system prompt, so names and domain terms can be biased without new training.

**Problem.** The current pipeline config passes no hotword or context input to ASR, and MT sees one previous turn. The evaluation harness has no entity-recall metric.

**Proposed architecture.** One terminology store feeds both ASR and MT.

- **ASR biasing:** relevant terms and background text go into Qwen3-ASR's system-prompt context.
- **Per-segment filtering:** only a few likely-relevant terms are sent per segment, because speech-LLM biasing degrades as unrelated terms accumulate.
- **Running context cache:** recently recognized text from the same session is fed back as context, the mechanism Qwen-Audio-3.0-ASR uses in place of a manual hotword list.
- **MT side:** the same entries go into the MT prompt as a glossary and as retrieved example translations. MLLP-VRAIN's IWSLT 2026 context-track system combined ASR word-boosting with retrieval of offline pre-translated exemplars; CUNI 2025 injected in-domain terminology by prompting.

**How to build.**

1. Create a glossary store per user and session: source term, fixed target translation, domain tag.
2. Retrieve the top few matching entries per segment from the running transcript.
3. Pass them, plus the last part of the session transcript, as Qwen3-ASR context.
4. Pass the same entries to the MT prompt as a glossary and as few-shot examples.
5. Add entity recall and biased/unbiased WER to the harness.

**Effectiveness (published).** Hotword recall in the Qwen-Audio-3.0-ASR report, without and with hotword conditioning:

| Category (P0 tier) | Doubao-ASR | Fun-ASR-Flash | Qwen-Audio-3.0-ASR |
| --- | --- | --- | --- |
| Person names | 32.27 → 82.64 | 37.33 → 95.26 | 62.12 → 99.43 |
| Subject terms | 75.14 → 93.06 | 71.84 → 92.53 | 69.36 → 99.42 |
| Trending buzzwords | 48.24 → 91.95 | 43.26 → 82.53 | 63.08 → 99.46 |
| Product entities | 67.59 → 78.70 | 73.64 → 89.92 | 72.87 → 99.07 |

A 2026 comparison found dedicated Whisper-based biasing methods cut biased WER by up to 88% relative while leaving other words largely unaffected.

**Other notes.**

- The same comparison tested Qwen3-ASR directly: speech-LLM biased WER degraded 39–570% relative as distractors accumulated, and depended on word order in the prompt.
- Speech LLMs were strong on read speech but generalized less well to non-read speech. Filtering the list with an auxiliary model recovered much, but not all, of the loss.
- The recall table above is for Qwen-Audio-3.0-ASR, a newer model than the Qwen3-ASR-1.7B in STiTy.

**Sources.** [Qwen3-ASR Technical Report](https://arxiv.org/pdf/2601.21337) · [Qwen-Audio-3.0-ASR Technical Report](https://arxiv.org/pdf/2609.07549) · [How to Recognize New Words](https://arxiv.org/html/2608.05759) · [MLLP-VRAIN IWSLT 2026](https://arxiv.org/abs/2606.17255) · [CUNI IWSLT 2025](https://arxiv.org/abs/2506.17077)

## 4. Semantic end-of-turn detection

**Priority P1 · Complexity Low.** Smart Turn v3.2 is an open (BSD-2) audio model that predicts whether a speaker has finished, supports Korean among 23 languages, and runs in about 10–100 ms on CPU.

**Problem.** STiTy decides turn ends with Silero VAD and a fixed 800 ms minimum silence. A silence timer cannot tell a finished turn from a mid-sentence pause, so it either cuts speakers off or waits too long.

**Proposed architecture.**

- Silero VAD keeps detecting silence. When silence starts, Smart Turn scores the current turn's audio (up to 8 s, left-padded) from prosody and content.
- The end-of-turn probability, not the timer, closes the sentence buffer, triggers final translation, and (with Proposal 8) starts speech output.
- A minimum and maximum endpoint delay bound the decision; LiveKit's defaults are 0.55 s VAD silence, 0.5 s minimum delay and 3.0 s maximum delay.

**How to build.**

1. Add a `turn` component to `core/components` that wraps Smart Turn v3.2 (int8 CPU build, 8 MB).
2. Lower Silero's minimum silence; the Smart Turn LiveKit plugin requires at least 0.25 s.
3. On each silence, pass the whole current turn to Smart Turn; if new speech arrives, re-run on the full turn, as the model card instructs.
4. Fine-tune on Korean conversational turns with the open `train.py` and datasets.
5. Measure cut-off rate and endpoint latency p50/p90, using LiveKit's eot-bench harness or the same metrics in STiTy's harness.

**Effectiveness (published).** Measured in a LiveKit Agents pipeline on 300 recorded telephony turns (Scicom model card):

| Turn detector | Latency p50 / p90 | Turns cut off |
| --- | --- | --- |
| VAD only | 0.63 / 0.71 s | 14.3% |
| Smart Turn v3, threshold 0.5 | 0.65 / 3.04 s | 10.0% |
| Scicom semantic-VAD whisper-tiny, threshold 0.5 | 0.64 / 0.74 s | 10.0% |
| Scicom semantic-VAD whisper-base, threshold 0.3 | 0.64 / 0.74 s | 9.7% |

Language-specific fine-tuning matters: on 4,168 real Tamil telephone clips, Smart Turn v3 zero-shot scored 70.30% accuracy, and Tamil fine-tuned versions scored 83.71% (tiny) and 86.13% (base).

**Other notes.**

- Smart Turn inference completed in about 65 ms on a standard Pipecat Cloud instance.
- All numbers above come from voice-agent and telephony audio, not translation sessions.
- The model works on audio only; text-conditioned turn detection is listed as a medium-term goal of the project.

**Sources.** [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn) · [Scicom semantic-VAD (whisper-tiny) model card](https://huggingface.co/Scicom-intl/semantic-vad-eot-whisper-tiny) · [Scicom semantic-VAD (whisper-base) model card](https://huggingface.co/Scicom-intl/semantic-vad-eot-whisper-base) · [smart-turn-livekit (Tamil results)](https://pypi.org/project/smart-turn-livekit/)

## 5. ASR-safe noise handling (dual audio path)

**Priority P1 · Complexity Medium.** Recent studies show that feeding denoised audio straight into modern ASR can raise error rates, while mixing a share of the original signal back in (observation adding) recovers accuracy.

**Problem.** Noisy environments are in scope, and STiTy has no enhancement stage. Adding a standard denoiser in front of Qwen3-ASR is the obvious fix, but the evidence below shows it can make recognition worse.

**Proposed architecture.**

- **Enhanced path:** a real-time speech enhancement model produces a cleaned stream for VAD, turn detection, diarization, and anything played to a listener.
- **ASR path:** Qwen3-ASR receives either the raw audio or an observation-added mix of raw and enhanced audio, never the enhanced audio alone.
- The mix weight is fixed or learned; the PTBM study reports both.

**How to build.**

1. Add an `enhancement` component to `core/components` that outputs a second audio stream beside the raw one.
2. Route the enhanced stream to VAD and turn detection; route raw or mixed audio to ASR.
3. Start with the fixed mix the PTBM study tested: 0.2 × noisy input + 0.8 × enhanced output.
4. Build an SNR ladder test set by mixing noise into KsponSpeech and LibriSpeech at several SNR levels.
5. Compare CER/WER for raw, enhanced-only and mixed input at each SNR before enabling any path in production.

**Effectiveness (published).**

| Study | Setup | Finding |
| --- | --- | --- |
| SAM-Audio preprocessing (2026) | Whisper variants; Bengali YouTube and English noisy sets | Enhancement consistently raised WER and CER vs raw noisy speech, and errors grew with larger Whisper models |
| Medical ASR study (2025) | Gemini ASR, Gaussian noise at amplitude 0.017 | Semantic WER rose from 11% to 58% with enhancement |
| Joint SE–ASR training study | WavLM-based ASR | Interpolating enhanced and observed signals improved ASR without jointly retraining either module |
| PTBM with learned observation adding (2026) | Whisper Large, 0.96 M-parameter front-end | CHiME-4 eval WER 6.24 (learned mix) vs 6.42 (fixed 0.2 mix) vs 6.60 (enhanced only) |

**Other notes.**

- The medical-ASR authors give two hypotheses: modern ASR already learned noise robustness, and enhancement adds artifacts ASR is sensitive to even when listeners do not notice them.
- The Qwen3-ASR report lists noisy speech recognition among the model's strengths, which is consistent with measuring before adding enhancement.
- Google states Gemini 3.5 Live Translate is noise-robust but does not publish its method.

**Sources.** [SAM-Audio and zero-shot ASR](https://arxiv.org/abs/2603.04710) · [When De-noising Hurts](https://arxiv.org/pdf/2512.17562) · [How does E2E ASR training impact SE artifacts?](https://cs.paperswithcode.com/paper/how-does-end-to-end-speech-recognition) · [PTBM with learned observation adding](https://arxiv.org/pdf/2608.30326) · [Qwen3-ASR Technical Report](https://arxiv.org/pdf/2601.21337) · [Gemini 3.5 Live Translate](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-live-3-5-translate/)

## 6. Reference-free quality estimation gate

**Priority P1 · Complexity Medium.** IWSLT 2026 ran its first speech-translation quality estimation (QE) shared task; the best systems score a translation without a reference, from the ASR transcript or the source audio.

**Problem.** STiTy emits every translation with equal confidence. Mumbled, noisy or misrecognized input produces a wrong translation with no signal to the listener or the pipeline.

**Proposed architecture.**

- A QE model scores each emitted translation unit from (source transcript or source audio, translation).
- The score is attached to the output event. Low scores can hold emission for more context, mark the text, or trigger a repeat request.
- None of the cited papers evaluates QE as a live gate inside a simultaneous system; they measure correlation with human judgments. The gate thresholds therefore need STiTy's own calibration.

**How to build.**

1. Start with CometKiwi-22 on (ASR transcript, translation). STiTy already runs `Unbabel/wmt22-cometkiwi-da` in its translator benchmarks.
2. Measure QE latency per unit on the serving GPU; CometKiwi-22 has 580 M parameters.
3. Apply tie calibration (score bucketing) if ranking within a session matters.
4. Move to a speech-aware QE model with a Qwen3-ASR backbone, as HydraQE does, once audio-side errors dominate.
5. Calibrate thresholds against human ratings on STiTy's Korean demo and ACL 60/60 outputs.

**Effectiveness (published).** IWSLT 2026 Speech Translation Metrics task, development set:

| QE system | Input | Segment-level Kendall τ | System-level SPA |
| --- | --- | --- | --- |
| Tie-calibrated CometKiwi (Kyung Hee Univ.) | ASR transcript + translation | 39.4% | 88.0% |
| Pairwise-ranking CometKiwi (Cisco) | ASR transcript + translation | 35.2% | — |
| CometKiwi-22 baseline | ASR transcript + translation | 34.6% | 89.4% |
| SpeechQE | Audio + translation | 29.2% | 86.0% |
| BLASER 2.0 QE | Audio + translation | 24.4% | — |

HydraQE (Ohio State) feeds source audio and translation into a Qwen3-ASR backbone with three prediction heads (human DA, MetricX-24 and xCOMET pseudo-labels). It reports outperforming cascaded text-based baselines and prior speech-based systems at segment and system level.

**Other notes.**

- Tie calibration needs no audio, no retraining, and one hyperparameter per target language.
- The IWSLT task covers English→German and English→Chinese; no Korean QE results were found.

**Sources.** [IWSLT 2026 Metrics track](https://iwslt.org/2026/metrics) · [Tie-Calibrated COMETKiwi](https://aclanthology.org/2026.iwslt-1.36/) · [Pairwise Ranking CometKiwi](https://aclanthology.org/2026.iwslt-1.38.pdf) · [HydraQE](https://aclanthology.org/2026.iwslt-1.37/) · [STiTy LOCAL\_TRANSLATION.md](https://github.com/STiTy-team/STiTy/blob/main/core/translator/LOCAL_TRANSLATION.md)

## 7. Streaming diarization and multitalker ASR

**Priority P1 · Complexity High.** NVIDIA's open stack pairs Streaming Sortformer diarization with a multitalker ASR that runs one instance per speaker, transcribing fully overlapped speech without speaker enrollment; its released checkpoint is English-only, so Korean needs training.

**Problem.** STiTy transcribes one mixed stream. When two people talk at once, their words are merged or attributed to the wrong speaker, and translation direction cannot follow who is speaking.

**Proposed architecture.**

- **Streaming Sortformer** outputs per-frame speaker activity. An arrival-order speaker cache stores embeddings of speakers already heard, so speaker order stays consistent across chunks.
- **Multitalker Parakeet** runs one instance per active speaker. Each instance gets the same mixed audio plus that speaker's activity, and speaker kernels injected into the encoder focus it on that speaker.
- Each speaker stream then enters its own stability and translation path (Proposals 1–2).

**How to build.**

1. Stand up NeMo with `diar_streaming_sortformer_4spk-v2.1` and `multitalker-parakeet-streaming-0.6b-v1` using the `SpeakerTaggedASR` streaming helper.
2. Choose latency via `att_context_size`: 0.08, 0.16, 0.56 or 1.12 s chunks.
3. Map speaker IDs to a session registry: language, target language, translation direction.
4. For Korean, fine-tune the multitalker model from a Korean single-speaker base on real conversation data plus simulated mixtures (the English model used Fisher, AMI, NOTSOFAR, ICSI and simulated LibriSpeech mixtures).
5. Or, as a lower-effort interim, cut Qwen3-ASR input by diarization segments, accepting NVIDIA's documented limitation below.
6. Evaluate with cpWER and DER on 2-speaker and 3–4 speaker Korean conversations.

**Effectiveness (published).** Multitalker Parakeet + Streaming Sortformer v2, cpWER at 1.12 s latency (all sets include overlapping speech):

| Test set | Speakers | cpWER (%) |
| --- | --- | --- |
| CH109 | 2 | 15.81 |
| Mixer 6 | 2 | 23.81 |
| AMI IHM | 3–4 | 21.26 |
| AMI SDM | 3–4 | 37.44 |

Self-reported DER on CALLHOME part 2 rises with speaker count: 6.57% (2 speakers), 10.05% (3), 12.44% (4), 21.68% (5). In single-speaker mode the model averages 7.44% WER on the Open ASR Leaderboard sets, against 7.16% for its base model.

**Other notes.**

- NVIDIA warns that cutting audio by diarization segments and running a normal ASR still leaves every overlapping voice in each cut, so the ASR may merge or pick the wrong speaker's words.
- NVIDIA's voice-agent docs state the diarization model is not robust to noise and speaker numbering restarts after a reset; this is why it should read the enhanced path from Proposal 5.
- The architecture needs one ASR instance per speaker, which scales compute with speaker count.
- Nemotron-3-Diarization, a newer Sortformer release, supports 8 speakers.
- Multitalker Parakeet is released under the NVIDIA Open Model License.

**Sources.** [Multitalker Parakeet model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) · [Streaming Sortformer v2 model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) · [Nemotron-3-Diarization ASR integration guide](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md) · [NeMo voice agent diarization docs](https://docs.nvidia.com/nemo/labs-voice-agent/about/core-concepts/speech-pipeline/speaker-diarization) · [Streaming Multitalker ASR tutorial](https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/asr/Streaming_Multitalker_ASR.ipynb)

## 8. Streaming TTS and output rate control

**Priority P2 · Complexity Medium.** Open streaming TTS now emits its first audio packet in about 0.1 s with 3-second voice cloning in Korean, and published S2ST work shows that controlling translation length and speech duration keeps latency from accumulating.

**Problem.** STiTy outputs text only, so users must look at a screen. When translated speech is added, speech that runs longer than the source makes delay grow over a session; Seed LiveInterpret 2.0 lists this "translated speech inflation" among the core problems of product-level simultaneous interpretation.

**Proposed architecture.**

- **Streaming TTS:** accepted translation text (Proposal 2) streams into a TTS model that starts speaking before the sentence is complete.
- **Voice cloning:** a few seconds of each speaker's audio condition the TTS so translated speech keeps the speaker's voice.
- **Rate control:** (a) translation length adapts to source speech rate, as in Self-Adaptive Translation; (b) synthesized speech duration is scaled down when lag grows, as in Liu et al.

**How to build.**

1. Serve Qwen3-TTS (0.6B or 1.7B, 12 Hz tokenizer) in streaming mode behind a new `synthesis` component.
2. Capture a 3-second reference clip per speaker from the diarized stream and cache the speaker prompt.
3. Track output lag = end of played speech − end of the matching source speech.
4. When lag exceeds a threshold, apply duration scaling; when it persists, instruct the MT prompt to shorten output.
5. Report Average Proportion (target speech duration / source speech duration) alongside latency.

**Effectiveness (published).**

| Component | Source | Reported result |
| --- | --- | --- |
| Qwen3-TTS (Apache 2.0) | Qwen3-TTS Technical Report, Jan 2026 | 97 ms first-packet emission; 3-second voice cloning; 10 languages including Korean; trained on 5M+ hours |
| CosyVoice 2 | CosyVoice repository | First-packet synthesis as low as 150 ms; bidirectional streaming; Korean supported |
| Duration scaling + pseudo-lookahead | Liu et al., Interspeech 2022 | Latency reduced by 0.2–0.5 s without lowering translation quality |
| Self-Adaptive Translation | Zheng et al., EMNLP Findings 2020 | At similar BLEU, more fluent speech (MOS) with substantially lower latency, Zh↔En |

**Other notes.**

- Google states Gemini 3.5 Live Translate keeps the speaker's intonation, pacing and pitch, and watermarks all generated audio with SynthID.
- Apple's AirPods Live Translation uses active noise cancellation to lower the original speaker's voice under the translation (see Proposal 9).
- Liu et al. report that incremental TTS typically needs future text for best quality, which is why they generate pseudo-lookahead from the translator.

**Sources.** [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621v1) · [CosyVoice](https://github.com/xxnuo/cosyvoice) · [Liu et al. 2022, From Start to Finish](https://arxiv.org/abs/2110.08214v3) · [Zheng et al. 2020, Self-Adaptive Translation](https://preview.aclanthology.org/moar-dois/2020.findings-emnlp.349) · [Seed LiveInterpret 2.0](https://arxiv.org/abs/2507.17527) · [Average Proportion metric](https://arxiv.org/pdf/2409.00965) · [Gemini 3.5 Live Translate](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-live-3-5-translate/)

## 9. Capture layer: echo cancellation, multi-mic, ducking

**Priority P2 · Complexity Medium.** Once STiTy plays translated speech (Proposal 8), the microphone also hears that playback; shipping products handle this at the device layer with echo cancellation, extra microphones and ducking of the original voice.

**Problem.** STiTy's client streams raw 16 kHz mono PCM over WebSocket from one microphone. With speech output on a phone speaker or open earbuds, played-back translation re-enters the microphone and can be transcribed again. In noise, one far microphone captures the other speaker poorly.

**Proposed architecture.**

- **Echo cancellation:** the client passes the audio it is playing (the far-end reference) to an echo canceller before sending microphone audio upstream. DeepVQE does echo cancellation, noise suppression and dereverberation in one real-time model.
- **Extra microphones:** Apple's Live Translation lets the iPhone's microphones assist the AirPods' microphones in noisy places, and tells users to move the phone closer to the other speaker.
- **Ducking:** when both people wear AirPods, active noise cancellation lowers the other person's voice so the translation is easier to follow.
- **Private playback:** Google's Android "listening mode" plays translated audio through the phone earpiece, held like a call, so others do not hear it.

**How to build.**

1. In STiTy-Mobile, route TTS playback through the platform audio path that exposes a far-end reference to the echo canceller.
2. Start with the platform's voice-processing echo cancellation; evaluate a DeepVQE-style model if residual echo is transcribed.
3. Add an echo test to the harness: play known TTS audio while recording, and count words from the playback that reach ASR output.
4. Support a second microphone stream (phone + earbuds) and select or combine by signal quality.
5. Add ducking: lower passthrough of the original voice while translated speech plays, on devices that support it.
6. Evaluate the WebRTC media platforms Google lists for Gemini Live Translate (LiveKit, Agora, Pipecat) for transport.

**Effectiveness (published).**

- DeepVQE reports state-of-the-art results on the non-personalized tracks of the ICASSP 2023 Acoustic Echo Cancellation and Deep Noise Suppression challenge test sets, runs in real time, and was tested on the Microsoft Teams platform.
- Apple and Google publish these capture features as product behavior; neither publishes measured translation-quality gains from them.

**Other notes.**

- Apple shows a live transcript on the iPhone for a partner without AirPods, and plays each side's translation in their own AirPods when both wear them.
- Google's two-way conversation mode switches output language automatically based on who is speaking.
- Per Proposal 5, echo-cancelled audio should also be checked for ASR artifacts before it replaces the raw stream.

**Sources.** [DeepVQE](https://arxiv.org/pdf/2306.03177) · [Apple Live Translation on AirPods (newsroom)](https://www.apple.com/ie/newsroom/2025/11/live-translation-on-airpods-expands-to-the-eu/) · [Apple Support: Use Live Translation with AirPods](https://support.apple.com/en-au/123185) · [Gemini 3.5 Live Translate](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-live-3-5-translate/) · [Gemini audio model updates](https://blog.google/products/gemini/gemini-audio-model-updates/) · [STiTy CLAUDE.md (protocol)](https://github.com/STiTy-team/STiTy/blob/main/CLAUDE.md)

## 10. Target speaker extraction

**Priority P2 · Complexity High.** Personalized enhancement conditioned on a known voice can isolate one speaker from competing voices; the September 2026 AV-PVQE paper cut target confusion in two-speaker mixtures from 46% to 1.6% by adding mouth features, at 20 ms delay.

**Problem.** When a known user and a nearby talker share one microphone, diarization (Proposal 7) labels who spoke but does not remove the other voice from the audio each ASR instance hears.

**Proposed architecture.**

- **Enrollment:** each user's device captures a short clean reference of that user's voice.
- **Extraction:** a personalized enhancement / target-speaker extraction model conditioned on that reference outputs the target voice in real time.
- **Visual cue (optional):** on video calls or camera-equipped devices, mouth features are added at the speaker-conditioning input, as in AV-PVQE.
- **ASR input:** because enhancement artifacts can hurt ASR (Proposal 5), the target is reinforced in the mixture rather than fed alone; target speaker reinforcement improved ASR accuracy in noise without retraining the acoustic model.

**How to build.**

1. Add an enrollment step in STiTy-Mobile that stores a speaker embedding or reference clip per user.
2. Integrate a real-time personalized enhancement model on the enhanced path.
3. Build a Korean two-speaker test set from KsponSpeech mixtures plus recorded real conversations.
4. Measure target-confusion rate and ASR CER with extraction-only, reinforcement and raw input.
5. For video contexts, fine-tune the visual front-end jointly with the reconstruction network, as AV-PVQE did.

**Effectiveness (published).**

| System | Setting | Result |
| --- | --- | --- |
| Personalized speech enhancement baseline (AV-PVQE paper) | Two-speaker mixtures, clean enrollment | Target confused in 46% of mixtures |
| AV-PVQE (+ mouth features, joint fine-tuning) | Same | Confusion 1.6%; no future frames; 20 ms algorithmic delay; larger gains on recorded meetings than on synthetic benchmarks |
| Target speaker reinforcement (2022) | Noisy ASR, no acoustic-model retraining | Improved ASR accuracy; a perceptual loss added a modest but consistent gain |

**Other notes.**

- The AV-PVQE authors note that most extractors are built and evaluated on synthetic mixtures, leaving listening quality and meeting behavior largely untested.
- Audio-only personalized models confuse the target far more often, so the audio-only case should be measured on STiTy data before deployment.

**Sources.** [AV-PVQE (arXiv 2609.30631)](https://arxiv.org/abs/2609.30631) · [Speaker Reinforcement Using Target Source Extraction for Robust ASR](https://arxiv.org/pdf/2205.04433) · [TargetVoice (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/pallala25_interspeech.pdf)

## 11. Training for simultaneity

**Priority P2 · Complexity High.** Proposals 1–2 run offline-trained models in streaming mode; published work gains further quality and latency by training the MT model on partial inputs and by optimizing latency directly with reinforcement learning.

**Problem.** STiTy's translators (MADLAD, Qwen3.5-4B, Gemma 3) were trained on complete sentences. The repository records that fragments are either translated awkwardly or completed with borrowed context. STiTy's only simultaneity-specific training today is the SEG-token fine-tune of Qwen3-ASR.

**Proposed architecture.** Three training tracks, each independent:

- **MT on partial input:** fine-tune the MT LLM so it translates source prefixes well. SimulMask does this by masking attention during fine-tuning to match a chosen decision policy; MLLP-VRAIN 2025 used document-level adaptation with prefix training.
- **RL for latency:** first train for quality at high latency, then use GRPO to reduce latency while keeping quality, as Hibiki-Zero does. Seed LiveInterpret 2.0 also combines large-scale pretraining with reinforcement learning.
- **ASR for stability:** train ASR to give stable transcripts across a range of latencies, as NVIDIA's dual-mode unified ASR transducer does for its IWSLT 2026 cascade.

**How to build.**

1. Build Korean↔English prefix–translation pairs from existing parallel data plus STiTy's DailyTalk and KsponSpeech transcripts.
2. Fine-tune the Proposal 2 MT model with LoRA using SimulMask (code in OSU-STARLAB/Simul-LLM) or prefix training.
3. Define a reward of translation quality minus a latency penalty, using the metrics in the Evaluation plan.
4. Apply GRPO to the emission policy or MT model, starting from the high-latency checkpoint.
5. Compare against the untrained Proposal 2 system on the same ACL 60/60 and Korean test sets.

**Effectiveness (published).**

| Method | Setting | Reported result |
| --- | --- | --- |
| SimulMask (EMNLP 2024) | Falcon LLM, IWSLT 2017 | Significant quality gain over prompting-based fine-tuning across five language pairs, with lower compute |
| Hibiki-Zero (ICML 2026) | Sentence-level training, then GRPO | State of the art on five X→English tasks in accuracy, latency, voice transfer and naturalness; new input language added with <1,000 h of speech |
| Seed LiveInterpret 2.0 (2025) | Pretraining + RL, Zh↔En | Cloned-speech latency cut from nearly 10 s to about 3 s (\~70%); over 70% correctness judged by human interpreters in complex scenarios |
| MLLP-VRAIN 2025 | Whisper Large-V3-Turbo + NLLB-3.3B, prefix training | Long-form simultaneous cascade built from adapted pre-trained models, without end-to-end training from scratch |

**Other notes.**

- Hibiki-Zero's final model was trained on 48 NVIDIA H100 GPUs; the weights (3B) and inference code are open.
- Hibiki-Zero removes the need for word-level alignments, which otherwise rely on language-specific heuristics; this matters for Korean, whose word order differs from English.

**Sources.** [SimulMask (EMNLP 2024)](https://arxiv.org/html/2405.10443v4) · [Simul-LLM code](https://sotaverified.org/papers/simultaneous-masking-not-prompting) · [Hibiki-Zero](https://arxiv.org/html/2602.11072) · [Hibiki-Zero weights](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16) · [Seed LiveInterpret 2.0](https://arxiv.org/abs/2507.17527) · [MLLP-VRAIN IWSLT 2025](https://arxiv.org/pdf/2506.18828) · [NeMo IWSLT 2026](https://aclanthology.org/2026.iwslt-1.23/)

## Evaluation plan

Each proposal is judged with the metrics its cited work reports, run on STiTy's existing sets plus the IWSLT 2026 dev set, and compared head-to-head with Gemini 3.5 Live Translate through the public Gemini Live API.

| Metric | Measures | Used by | Proposals |
| --- | --- | --- | --- |
| StreamLAAL | Latency on unsegmented long-form speech | IWSLT 2025; STiTy RESULTS.md | 1, 2 |
| LongYAAL (CU and CA) | Long-form latency, computation-unaware and -aware | IWSLT 2026 (AlignAtt4LLM, MLLP-VRAIN) | 1, 2 |
| FTL mean / median / p90 | Per-commit delay and tail | STiTy RESULTS.md | 1, 2 |
| XCOMET-XL, chrF, BLEU | Translation quality | IWSLT 2026 | 2, 3, 11 |
| COMET-DA, CometKiwi | Translation quality (reference / reference-free) | STiTy LOCAL\_TRANSLATION.md | 2, 6 |
| Entity recall, biased / unbiased WER | Names and domain terms | Qwen-Audio-3.0-ASR report; New Words comparison | 3 |
| Cut-off rate, endpoint latency p50/p90 | Turn detection | LiveKit eot-bench; Scicom model cards | 4 |
| CER/WER across an SNR ladder | Noise robustness | SE-for-ASR studies in Proposal 5 | 5, 9 |
| Kendall τ vs human ratings | QE reliability | IWSLT 2026 Metrics track | 6 |
| DER, cpWER | Speaker attribution under overlap | NVIDIA Sortformer / Multitalker Parakeet cards | 7 |
| First-packet latency, speaker similarity, Average Proportion | Speech output delay, voice match, length inflation | Qwen3-TTS, CosyVoice, S2ST latency work | 8 |
| Target-confusion rate | Wrong-speaker extraction | AV-PVQE | 10 |

**Test sets.**

- **Existing in STiTy:** ACL 60/60 dev, LibriSpeech, KsponSpeech, DailyTalk, and the Korean demo script set.
- **IWSLT 2026 MCIF dev set:** about 2.1 hours, 21 long academic talks; used by AlignAtt4LLM and MLLP-VRAIN.
- **Audio-NTREX-L:** used to compare Seed LiveInterpret 2.0, Hibiki-Zero and SeamlessStreaming.
- **Hibiki-Zero benchmark:** Kyutai released multilingual speech-translation evaluation data with the model.

**Head-to-head.** Gemini 3.5 Live Translate is in public preview on the Gemini Live API, so the same audio can be run through both systems. The long-form S2ST evaluation method used on Seed LiveInterpret 2.0 and Hibiki-Zero applies to speech output.

**Sources.** [CMU IWSLT 2025 (StreamLAAL task description)](https://arxiv.org/pdf/2506.13143) · [AlignAtt4LLM](https://arxiv.org/html/2606.03967v1) · [MLLP-VRAIN IWSLT 2026](https://arxiv.org/pdf/2606.17255) · [Practical Evaluation of Long-Form S2ST](https://arxiv.org/pdf/2606.15059) · [Hibiki-Zero](https://arxiv.org/html/2602.11072) · [Gemini Live API](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-live-3-5-translate/)

## Sources

**STiTy repository**

- [evaluation/ast/RESULTS.md](https://github.com/STiTy-team/STiTy/blob/main/evaluation/ast/RESULTS.md) · [LibriSpeech paper\_result/ASR](https://github.com/STiTy-team/STiTy/tree/main/evaluation/LibriSpeech/paper_result/ASR) · [core/translator/LOCAL\_TRANSLATION.md](https://github.com/STiTy-team/STiTy/blob/main/core/translator/LOCAL_TRANSLATION.md) · [Korean pipeline config](https://github.com/STiTy-team/STiTy/blob/main/configs/pipelines/asr.qwen-seg-ko%2Bmt.qwen3.5-4b.yml) · [CLAUDE.md](https://github.com/STiTy-team/STiTy/blob/main/CLAUDE.md)

**Products**

- [Gemini 3.5 Live Translate (Google, 2026-06-09)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-live-3-5-translate/)
- [Gemini 2.5 Native Audio and live speech translation (Google, Dec 2025)](https://blog.google/products/gemini/gemini-audio-model-updates/)
- [Live Translation on AirPods (Apple Newsroom, Nov 2025)](https://www.apple.com/ie/newsroom/2025/11/live-translation-on-airpods-expands-to-the-eu/) · [Apple Support](https://support.apple.com/en-au/123185)
- [Seed LiveInterpret 2.0 (ByteDance)](https://arxiv.org/abs/2507.17527)
- [Hibiki-Zero (Kyutai)](https://arxiv.org/html/2602.11072) · [weights](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16)
- [A Practical Evaluation Method for Long-Form Simultaneous S2ST](https://arxiv.org/pdf/2606.15059)

**IWSLT simultaneous systems**

- [AlignAtt4LLM (IWSLT 2026)](https://arxiv.org/html/2606.03967v1) · [code](https://github.com/QuentinFuxa/Alignatt4LLM)
- [MLLP-VRAIN (IWSLT 2026)](https://arxiv.org/abs/2606.17255) · [MLLP-VRAIN (IWSLT 2025)](https://arxiv.org/pdf/2506.18828)
- [NeMo@IWSLT 2026](https://aclanthology.org/2026.iwslt-1.23/)
- [CUNI (IWSLT 2025)](https://arxiv.org/abs/2506.17077)
- [BeaverTalk (IWSLT 2025)](https://arxiv.org/pdf/2505.24016)
- [CMU (IWSLT 2025)](https://arxiv.org/pdf/2506.13143)

**ASR, stability, biasing**

- [Qwen3-ASR Technical Report](https://arxiv.org/pdf/2601.21337) · [Qwen-Audio-3.0-ASR Technical Report](https://arxiv.org/pdf/2609.07549)
- [Turning Whisper into Real-Time Transcription System](https://arxiv.org/abs/2307.14743) · [whisper\_streaming](https://github.com/ufal/whisper_streaming)
- [How to Recognize New Words: Context Biasing vs Speech LLMs](https://arxiv.org/html/2608.05759)

**Turn detection**

- [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn) · [Scicom semantic-VAD whisper-tiny](https://huggingface.co/Scicom-intl/semantic-vad-eot-whisper-tiny) · [whisper-base](https://huggingface.co/Scicom-intl/semantic-vad-eot-whisper-base) · [smart-turn-livekit](https://pypi.org/project/smart-turn-livekit/)

**Noise, echo, extraction**

- [SAM-Audio preprocessing and zero-shot ASR](https://arxiv.org/abs/2603.04710) · [When De-noising Hurts](https://arxiv.org/pdf/2512.17562) · [E2E ASR training and SE artifacts](https://cs.paperswithcode.com/paper/how-does-end-to-end-speech-recognition) · [PTBM with learned observation adding](https://arxiv.org/pdf/2608.30326)
- [DeepVQE](https://arxiv.org/pdf/2306.03177)
- [AV-PVQE](https://arxiv.org/abs/2609.30631) · [Speaker Reinforcement for Robust ASR](https://arxiv.org/pdf/2205.04433) · [TargetVoice](https://www.isca-archive.org/interspeech_2025/pallala25_interspeech.pdf)

**Quality estimation**

- [IWSLT 2026 Metrics track](https://iwslt.org/2026/metrics) · [Tie-Calibrated COMETKiwi](https://aclanthology.org/2026.iwslt-1.36/) · [Pairwise Ranking CometKiwi](https://aclanthology.org/2026.iwslt-1.38.pdf) · [HydraQE](https://aclanthology.org/2026.iwslt-1.37/)

**Diarization and multitalker ASR**

- [Multitalker Parakeet Streaming 0.6B](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) · [Streaming Sortformer v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) · [Nemotron-3-Diarization integration guide](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md) · [NeMo voice agent diarization docs](https://docs.nvidia.com/nemo/labs-voice-agent/about/core-concepts/speech-pipeline/speaker-diarization)

**Speech output and training**

- [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621v1) · [CosyVoice](https://github.com/xxnuo/cosyvoice)
- [Liu et al. 2022, latency reduction for incremental TTS](https://arxiv.org/abs/2110.08214v3) · [Zheng et al. 2020, Self-Adaptive Translation](https://preview.aclanthology.org/moar-dois/2020.findings-emnlp.349) · [Average Proportion metric](https://arxiv.org/pdf/2409.00965)
- [SimulMask (EMNLP 2024)](https://arxiv.org/html/2405.10443v4)
