# DialogueContext — 설계 (에이전트 공용 계약서)

대화 문맥을 **어떤 형태로(A/B/C/D), 얼마나(n=0/1/3/5)** 주면 번역 품질과 지연이 어떻게 바뀌는지
보는 실험이다. 이 문서는 데이터·실행기·채점기가 서로 맞물리도록 정한 계약이다. 필드 이름을
바꾸려면 이 문서부터 고친다.

## 고정된 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 방향 | ko→en 한 방향 | STiTy 주 사용자는 한국어 화자. 한국어는 주어·목적어 생략, 성별 없는 대명사(걔), 존댓말이 있어 문맥 의존 현상이 영어 쪽보다 훨씬 많다. 영어 출력이라 COMET·판정기도 더 믿을 만하다 |
| 데이터 | 새로 만든 5개 대화 × 목표 턴 4개 = 20 인스턴스 | 기존 FLEURS 는 독립 문장, DialogueMT 는 대화당 6~7문장이라 앞 턴 5개를 가진 목표 턴이 거의 없고 현상도 의도적으로 심지 않았다 |
| 모델 | `Qwen/Qwen3.5-4B`, `unsloth/gemma-3-4b-it` (둘 다 4bit nf4), DeepL `quality_optimized` (free key), OpenAI `gpt-6-luna` (`reasoning_effort: none`) | 요청 목록. 로컬 두 개는 기존 `bench/configs/mt-ko-en/*.yml` 과 같은 식별자·양자화 |
| TGT 문맥 | 주 실험은 **정답(reference) 앞 번역** | 이전 번역 오류 누적을 섞지 않으려고. 자기 번역을 쓰는 경우는 보조 실험(S1) |
| 시드 | 20260925 | 작업 순서 셔플 |
| 판정 모델 | OpenAI `gpt-6-sol`, `reasoning_effort: low` | 후보 모델과 다른 급의 상위 모델. 같은 회사라 자기 선호 편향 가능성은 한계로 적는다 |

## 파일 배치

```
evaluation/DialogueContext/
  DESIGN.md                  이 문서
  README.md                  재현 방법
  data/dialogues.jsonl       대화 원본 (한 줄 = 한 턴)
  data/instances.jsonl       평가 인스턴스 20개 (한 줄 = 목표 턴 하나)
  data/preview.md            사람이 훑어보는 표
  scripts/                   파이썬 코드 (아래)
  configs/experiment.yml     실험 설정 (모델, 조건, 시드, 경로, 예산)
  results/<run_id>/          원시 결과 · 채점 · 집계 · 그래프 (config·프롬프트 사본 포함)
  logs/                      실행 로그, 완료 표시 파일
  report/REPORT.md           최종 보고서
```

파이썬 환경은 두 개다 (예전 세션이 만든 것, 경로는 `configs/experiment.yml` 의 `envs`):
- `venv-tf5` (transformers 5.17, bitsandbytes, httpx, sacrebleu): **번역 실행**
- `venv-metrics` (transformers 4.57, unbabel-comet 2.2.7, matplotlib, sacrebleu): **COMET·XCOMET·그래프**

API 키는 저장소 루트 `.env` 에서 `core.utils.env.load()` 또는 python-dotenv 로 읽는다
(`OPENAI_API_KEY`, `DEEPL_API_KEY`, `HF_TOKEN`, `GEMINI_API_KEY`). 코드에 키를 적지 않는다.

## 데이터 스키마

### `data/dialogues.jsonl` — 한 줄이 한 턴

```json
{"dialogue_id": "c01", "turn_id": 0, "speaker": "A", "ko": "...", "en": "...",
 "fragment_of_next": false}
```
- `turn_id` 는 0부터. 같은 화자가 말을 끊어 이어 가면(스트리밍 ASR 조각) 턴이 여러 개로 나뉜다.
  이때 앞 조각에 `fragment_of_next: true`.
- `en` 은 정답 번역. 대화 전체에서 일관되게 쓴다 (같은 고유명사·용어·말투).

### `data/instances.jsonl` — 한 줄이 평가 인스턴스 하나

```json
{"instance_id": "c01-t07", "dialogue_id": "c01", "target_turn_id": 7,
 "source_language": "ko", "target_language": "en",
 "speakers": {"A": "지민 (여, 20대, 대학생)", "B": "..."},
 "current_speaker": "A",
 "previous_turns": [{"turn_id": 2, "speaker": "B", "ko": "...", "en": "..."}, ...],
 "current_source": "...", "reference_translation": "...",
 "challenge_tags": ["pronoun_coreference", "omitted_argument"],
 "checks": [{"tag": "omitted_argument", "requirement": "…'them' (the socks) or equivalent object must be present"}],
 "expected_context_effect": "...", "note": "..."}
```
- `previous_turns` 는 목표 턴 **바로 앞 5개 이상**(해당 대화의 처음부터 전부를 넣는다. 실행기가 뒤에서 n개 자른다).
- `checks` 는 판정기가 통과/실패를 매기는 구체 조건. 태그별 성공률이 이것으로 계산된다.
  한 인스턴스에 1~3개. 문맥 없이는 맞히기 어려운 조건이어야 한다.
- `challenge_tags` 어휘 (이 10개만 쓴다):
  `pronoun_coreference`, `omitted_argument`, `gender_reference`, `register_politeness`,
  `lexical_consistency`, `word_sense`, `fragment_incremental`, `discourse_connective`,
  `entity_consistency`, `context_trap`

## 조건

- `context_strategy` ∈ `SRC`, `TGT`, `SRC_TGT`, `SPK_SRC_TGT`, 기준선 `NONE`
- `context_n` ∈ 1, 3, 5 (기준선은 0)
- 주 실험: 20 인스턴스 × 12 조건 × 4 모델 = 960, 기준선 20 × 4 = 80
- 보조 S1 (자기 번역 문맥): 전략 `TGT`, `SRC_TGT`, n=3. 각 모델이 대화를 첫 턴부터 그 대화의 마지막
  목표 턴까지 차례로 번역하며 자기 출력을 앞 번역으로 쓴다. 채점은 20 목표 턴만.

### 정준 요청 (canonical request)

모든 모델이 같은 정보를 받는다. 실행기는 조건마다 이것을 먼저 만들고 결과 행에 그대로 저장한다.

```json
{"source_language": "ko", "target_language": "en", "current_utterance": "...",
 "current_speaker": "Speaker 2",          // SPK_SRC_TGT 에서만 값, 나머지 null
 "context_strategy": "SRC_TGT", "context_n": 3,
 "context_items": [{"speaker": null, "src": "...", "tgt": "..."}, ...]}   // 오래된 것부터
```
- `SRC`: `src` 만, `TGT`: `tgt` 만, `SRC_TGT`: 둘 다, `SPK_SRC_TGT`: 둘 다 + `speaker`.
- 화자 이름표는 대화 안 첫 등장 순서로 `Speaker 1`, `Speaker 2`. 실제 이름·성별 설명(`speakers`)은
  모델에 주지 않는다 (운영 ASR 도 모른다). 대사 안에 이름이 나오는 것은 그대로다.
- `SPK_SRC_TGT` 에서는 현재 발화의 화자 이름표도 준다. 안 주면 화자 표시가 현재 발화에 쓸모가 없다.

### LLM 프롬프트 (Qwen·Gemma·GPT 동일)

system:
```
You are a real-time dialogue translator.
Translate only the CURRENT UTTERANCE from SOURCE LANGUAGE to TARGET LANGUAGE.
Use CONTEXT only to resolve ambiguity and maintain discourse consistency.
Do not translate or repeat the context.
Preserve all information in the current utterance and do not add information that is not supported by it or the context.
Preserve speaker intent, register, pronoun/reference consistency, and natural conversational style.
Output only the translation.
```
user (예: SPK_SRC_TGT, n=2):
```
SOURCE LANGUAGE: Korean
TARGET LANGUAGE: English

CONTEXT (previous turns, oldest first):
[Speaker 1]
SRC: 양말 말하는 거 맞지?
TGT: You mean the socks, right?
[Speaker 2]
SRC: 응. 그런데 집에 두고 왔어.
TGT: Yeah. But I left them at home.

CURRENT UTTERANCE (Speaker 1):
그래서 다시 가지러 갔지.
```
- `SRC` 줄은 `SRC: ...` 만, `TGT` 는 `TGT: ...` 만, 화자 없는 전략은 `[Speaker k]` 줄이 없다.
  항목 사이 구분은 줄바꿈 하나. `SRC_TGT` 는 항목마다 `SRC:`/`TGT:` 두 줄.
- 기준선: `CONTEXT (previous turns, oldest first):` 아래에 `(none)`.
- 로컬 모델은 chat template, `enable_thinking=False`, greedy, `max_new_tokens=200`, BOS 중복 없이
  (`add_special_tokens=False`). GPT 는 chat completions, `temperature` 0, `reasoning_effort: none`,
  스트리밍(TTFT 측정). 출력 형식 JSON 강제 없음 — 지시대로 번역만.

### DeepL

- `text` = 현재 발화, `context` = 위 user 프롬프트의 CONTEXT 블록 본문과 같은 문자열
  (`[Speaker k]`/`SRC:`/`TGT:` 표기 그대로, 기준선은 context 생략). 현재 발화 화자 표시는 DeepL
  에 넣을 자리가 없어 빠진다 — 한계로 적는다. `model_type: quality_optimized`.
- DeepL context 는 과금되지 않는다. free key 라 비용 0, 병렬 금지(429).

### 출력 정리 (모든 모델 공통 함수 하나)

앞뒤 공백·감싼 따옴표 제거, 맨 앞 `Translation:`/`TGT:`/`English:`/`CURRENT UTTERANCE…:` 같은 표지
제거. 여러 줄이면 합치지도 고르지도 않고 줄바꿈만 공백으로 바꿔 둔 채 `format_violation` 에
사유를 기록한다 (판정기가 그대로 보게). 원출력(`raw_output`)은 항상 보존.

## 실행 규칙 (지연)

- 한 프로세스(venv-tf5)에서 로컬 두 모델을 모두 올려 두고, **전 모델·전 조건 작업 1,040개를 시드로
  섞어** 한 줄로 차례로 실행한다(동시성 1). 모델 로드 시간은 따로 기록, 지연에서 제외.
- 측정 전 워밍업: 로컬 모델마다 5회, API 마다 2회 (실험에 쓰지 않는 문장). 결과는 버린다.
- 기록: `latency_ms`(요청 시작~최종 출력, API 는 네트워크 포함), `ttft_ms`(로컬: 첫 생성 토큰,
  GPT: 첫 스트림 내용 조각, DeepL: null), `generation_ms`, `input_tokens`/`output_tokens`
  (로컬: 토크나이저, GPT: usage), `input_chars`, `context_chars`, `output_chars`.
- 행마다 바로 append (중간에 죽어도 남게). 재시작하면 끝난 `job_id` 는 건너뛴다.
- API 비용: 호출마다 usage·모델·추정 비용·누적 비용을 `results/<run_id>/api_usage.jsonl` 에 append.
  gpt-6-luna 단가 $0.10 / cached $0.01 / out $0.50 per 1M (2026-09-25 OpenAI 가격표).
  gpt-6-sol $2.00 / $0.20 / $10.00. 예산 상한은 config 의 `budget_usd`.
- 다른 세션이 같은 GPU 를 쓰고 있을 수 있다. 실행 시작·끝에 `nvidia-smi` 스냅숏을 남긴다.

## 결과 행 (`results/<run_id>/translations.jsonl`)

```
job_id, phase (warmup|main|baseline|s1), order_index, model, instance_id, dialogue_id,
target_turn_id, context_strategy, context_n, request (정준 요청 전체), prompt_system, prompt_user
(DeepL 은 context 문자열), current_source, reference, raw_output, hypothesis, format_violation,
latency_ms, ttft_ms, generation_ms, input_tokens, output_tokens, input_chars, context_chars,
output_chars, error, started_at
```
채점기는 `job_id` 기준으로 두 파일을 쓴다. `scores_reference.jsonl` 에 `comet`, `xcomet`,
`xcomet_error_spans`, `chrf_pp`, `bleu`, `scores_judge.jsonl` 에 판정 차원 6개(`judge_<차원>`, 1~5),
`judge_mean`, `judge_error_labels`, `judge_checks`, `judge_check_pass_rate`, `judge_rationale`.
`aggregate.py` 가 번역 행과 두 파일을 합쳐 `merged.jsonl`/`merged.csv` 와 `tables/`, `figures/` 를 만든다.
