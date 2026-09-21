# bench 번역 쪽 — 번역기만 다시 돌리기, API 번역기, 대화 번역 품질

[README.md](README.md) 의 실행(`make bench`) 위에 얹은 번역 쪽 기능이다. 저장된 ASR 결과로
번역기만 바꿔 다시 돌리고(`retranslate`), 외부 API 번역기를 쓰고, 대화 번역 품질 6축과 그
재료를 기존 run 에서 파생 run 으로 만든다.

```bash
make retranslate SOURCE=bench/runs/<원본> \
  CONFIG=bench/configs/examples/retranslate.yml     # 저장된 ASR로 번역만 재실행
make annotate-quality SOURCE=bench/runs/<원본> OUTPUT=bench/runs/<원본>-quality
make asr-text-robustness SOURCE=bench/runs/<원본> \
  CONFIG=bench/configs/examples/quality_ko_en.yml OUTPUT=bench/runs/<원본>-text-robustness
```

## 파일

```
retranslate.py          저장된 확정 조각을 번역기만 바꿔 다시 번역하는 실행
translation_config.py   retranslate 설정(번역기 하나 + 언어 + context_scope)
annotate_quality.py     기존 run 에 중요 정보·유창성 주석을 붙인 파생 run
asr_text_robustness.py  기존 run 의 전사로 clean·noisy 번역 강건성을 재는 파생 run
derived.py              파생 run 이 같이 쓰는 원본 run 읽기·결과 쓰기
configs/examples/       위 명령 예시가 쓰는 설정
configs/mt-ko-en*/      번역기 비교 실험 설정
```

채점과 그 재료는 `core/utils/metrics` 에 있다 — 6축 집계, 중요 정보 규칙 추출기
(`critical_values`), LLM 판정기(`judge`)와 그 주석(`critical_information`·`fluency` 의
`annotate_*`), ASR 흉내 잡음(`text_noise`), 사람 정답 입력을 고르는 `metric_inputs`. 여기 있는
CLI 는 run 을 읽고 번역기·판정기를 불러 결과를 쓰는 일만 한다.

## 저장된 ASR로 번역기만 비교

`python -m bench.retranslate <source-run> <config.yml>`은 원본 run의
`items.jsonl`을 읽고 `segments[*].original`을 같은 순서로 새 번역기에 전달한다. 오디오,
VAD, ASR은 로드하지 않는다. 원본에 commit segment가 없는 옛 결과만 전체 `hypothesis`를 한
세그먼트로 사용하는 fallback이 명시적으로 기록된다.

`context_scope: item`은 매 발화에서 문맥을 초기화한다 — 같은 발화 안에서 먼저 확정된 조각만
문맥이 된다. `context_scope: group`은 같은 대화 group의 앞 발화에서 확정된 문장(원문·번역)까지
이어서 전달한다. **`make bench` 의 cascade 파이프라인은 `group` 과 같게 돈다.** 두 조건은 결과
config와 비교 없이 섞으면 안 된다.

새 run의 `summary.json.source_run`에는 원본 경로, `items.jsonl` SHA-256, 원본 commit과 dataset
정보가 남는다. ASR timing은 원본 segment의 `source_timing`으로만 보존하고 새 run의
WER/FSL/LAAL로 보고하지 않는다. 새로 측정하는 시간은 segment별 `translation_elapsed_sec`과
집계된 `translation_runtime`뿐이다.

원본의 후보 번역에 종속된 COMET, span, judge, intent 판정은 새 번역에 재사용하지 않는다.
사람이 만든 reference span/label만 전달되고, 새 후보 주석을 생성하기 전까지 관련 지표는
`unavailable`로 남는다.

## API 번역기 — DeepL · Gemini · GPT

`translation.name` 에 `deepl`, `gemini`, `gpt` 를 쓰면 로컬 모델 대신 외부 API 로 번역한다.
키는 저장소 루트 `.env` 에 백엔드마다 하나씩 둔다. 키가 없거나 틀리면 **번역을 한 건도
보내기 전에** 멈춘다 — `load()` 가 돈이 안 드는 조회(DeepL `/v2/usage`, 모델 정보 조회)로
키와 모델 이름을 먼저 확인한다.

| `name` | 키 | 고르는 설정 | 기본값 |
|---|---|---|---|
| `deepl` | `DEEPL_API_KEY` | `model_type`: `latency_optimized` · `quality_optimized` · `prefer_quality_optimized` (**필수**), `formality` | — |
| `gemini` | `GEMINI_API_KEY` | `model`, `thinking_level`: `minimal` · `low` · `medium` · `high`, `temperature` | `gemini-3.1-flash-lite`, `minimal`, 온도는 API 기본값 |
| `gpt` | `OPENAI_API_KEY` | `model`, `reasoning_effort`: `none` · `minimal` · `low` · `medium` · `high` · `xhigh` | `gpt-5.4-nano`, `none` |

셋이 같이 받는 설정은 `context`(앞 확정 문장 몇 개를 문맥으로 줄지, 기본 10, `0` 이면 끔),
`context_chars`(그 문장들의 원문 글자 수 합 상한, 기본 500), `budget_usd`, `max_retries`
(기본 3), `timeout_sec`(기본 30)다. 429·5xx·연결 오류만 다시 시도하고, 나머지 4xx(DeepL 할당량
초과 456 포함)는 바로 그 세그먼트를 실패로 남긴다.

**문맥은 앞 확정 문장(final) N개다.** 같은 `group` 안에서 ASR 이 확정한 조각이 순서대로
쌓이고, 발화(item)가 바뀌어도 이어진다. 가장 최근 것부터 거꾸로 담다가 개수(`context`)나
원문 글자 수 합(`context_chars`)이 넘치기 직전에 멈춘다 — 문장을 중간에서 자르지 않으므로,
바로 앞 문장 하나가 상한보다 길면 문맥이 빈다. 번역 줄은 담긴 문장의 것만 따라 들어가므로
글자 수에 따로 세지 않는다. 담긴 문장 중 하나라도 번역이 비었으면(실패했거나 다른 타깃으로
번역됐으면) 번역은 모두 뺀다 — 어긋난 번역을 보여 주지 않기 위해서다. 운영 프록시(`--context`)가 "화자·언어 무관 최근 final
N 개" 를 넘기는 것과 같은 단위다. 기록은 `core/pipelines/translation/dialogue.py` 의
`Dialogue` 하나가 맡고, `bench` 의 cascade 파이프라인과 `bench.retranslate` 가 같은 것을 쓴다.

**화자는 ASR 이 알려 줄 때만 쓴다.** 지금 ASR 은 화자를 나누지 못하므로 문맥은 **한 덩어리**로
들어간다. 데이터셋 매니페스트의 `speaker` 는 쓰지 않는다 — 운영에서는 없는 정답이라, 쓰면
bench 가 실제보다 좋은 조건에서 잰다. ASR 의 `transcribed` 이벤트(또는 저장된 세그먼트)에
`speaker` 가 실려 오면 그때부터 대화 안에서 처음 나온 순서대로 `A`·`B`… 이름표를 붙여
화자별 줄로 나눈다.

**모든 번역기가 받는 정보는 같다.** 어느 앞 문장을 몇 개, 번역과 함께 줄지는
`dialogue.recent` 하나가 정하고, API 셋과 `local` 이 전부 그 결과를 그대로 받는다(기본값도
`context: 10`·`context_chars: 500` 으로 같다). 모델마다 다른 것은 그 정보를 **보여 주는 모양**
뿐이다. `core/pipelines/translation/test_local_context.py` 가 로컬과 API 가 받는 문맥이 같은지
직접 비교한다.

- **Gemini 와 GPT 는 앞 문장과 새 조각을 나눠 받고 JSON 으로 답한다.** 화자를 모를 때의
  사용자 메시지:

  ```
  What was said before, oldest first. Speakers are not identified: this may be one person,
  or several people taking turns.
  어제 그 영화 봤어? 응 봤어. 근데 결말이

  What the listener has already seen in English:
  Did you see that movie yesterday? Yeah, I did. But the ending

  New piece, in Korean. Translate it into English:
  좀 이상하지 않았어?
  ```

  둘째 덩어리는 듣는 사람이 이미 본 번역이다. 앞 문장의 언어가 섞여 있으면 문장마다
  `[Korean]`·`[English]` 가 붙는다. 화자를 알면 앞부분이
  `A (Korean): …` / `= 번역` 줄로 바뀌고, 새 조각 앞에 `speaker change, now B` 나
  `A keeps talking` 이 붙는다. 답은 `{"translation": "..."}` 하나로 받아(GPT
  `response_format`, Gemini `responseMimeType`) 그 필드만 꺼내므로, 모델이 앞 문장을 다시
  옮겨 붙여도 섞이지 않는다. 필드가 없으면 그 세그먼트는 실패로 남는다(호출 비용은 기록된다).
  지시문은 `core/pipelines/translation/prompt.py` 에 있다.
- **DeepL 은 같은 내용을 `context` 필드로 넘긴다** — 화자를 모르면 원문 덩어리 다음 줄에
  번역 덩어리, 알면 `A (Korean): …` / `= 번역` 줄. 이 글자는 번역되지도 과금되지도 않으므로
  꺼낼 것이 없다 — 번역되는 것은
  `text` 의 새 조각뿐이다. `en` 타깃은 `EN-US`, `pt` 는 `PT-BR`, `zh` 는 `ZH-HANS` 로 보낸다.
- **`local` 도 같은 문맥을 받는다.** 앞 문장은 한 줄씩(`- [Korean] 원문 → [English] 번역`)
  들어간다. bench 는 로컬 LLM 을 `always_use_context=True` 로 올린다 — 운영 번역 서버에서는
  켜져 있는 두 안전장치(문장 부호로 안 끝나는 조각이면 문맥 빼기, 번역이 너무 짧으면 문맥 없이
  다시 하기)가 꺼져서, 받은 문맥을 API 번역기와 똑같이 그대로 쓴다. 운영 서버의 기본 동작은
  바뀌지 않는다.
- **문맥을 넣을 자리가 없는 모델은 `context: 0` 으로만 돈다.** `translategemma`(chat template
  이 입력 형식을 고정한다)와 seq2seq 백엔드(NLLB·MADLAD)다. 이 모델에 `context` 를 1 이상 주면
  설정 단계에서 오류가 난다 — 조용히 버리면 같은 정보를 받은 줄 알고 비교하게 되기 때문이다.
  이 모델들의 결과는 "문맥 없음" 조건이라 문맥을 받는 조건과 나란히 놓으면 안 된다.
- **GPT 는 `reasoning_effort: none` 일 때만 `temperature: 0` 을 보낸다.** 사고를 켜면
  API 가 온도 지정을 거부한다. Gemini 는 온도를 1.0 아래로 내리면 반복 출력이 생길 수
  있다고 Google 이 권고하므로 기본으로는 보내지 않는다 — 그래서 **Gemini 결과는 실행마다
  조금씩 다를 수 있다.**

**비용은 호출마다 남는다.** `runs/<name>/translation_usage.jsonl` 에 호출 하나가 한 줄로
append 된다 — 모델, 입력·캐시·출력·사고 토큰(DeepL 은 과금 글자 수와 실제 쓰인
`model_type_used`), 그 호출의 추정 비용과 누적 추정 비용. 이 파일은 다시 돌려도 지우지
않는다. 25호출마다 누적 비용이 로그에 찍히고, 끝나면 `summary.json.usage` 에 합계가 들어간다.

```bash
python -m core.meaning_segmentator.autoseg.infra.cost_report \
  --run-id $PWD/bench/runs/<name> --budget 1
```

`budget_usd` 에 닿으면 그 뒤 세그먼트는 API 를 부르지 않고 빈 번역으로 남긴다. 단가는
GPT 가 `core/meaning_segmentator/autoseg/infra/gateway.py` 의 `_PRICES`, Gemini 가
`core/pipelines/translation/gemini.py` 의 `PRICES` 다. 표에 없는 모델은 비용이 0 으로
잡히므로 시작할 때 경고하고, 그 상태로 `budget_usd` 를 주면 설정 오류로 멈춘다. DeepL 은
글자당 과금이라 `usd_per_million_characters`(기본 25, 옛 Pro 종량 단가라 실제보다 크게 잡힐
수 있다)로 계산하고, 무료 키(`:fx` 로 끝남)는 0 으로 잡는다. 과금 글자 수는 그대로 남으므로
단가가 달라도 나중에 다시 계산할 수 있다.

`bench/configs/mt-ko-en/` 에 네 조건의 설정이 있다 (`deepl-latency`, `deepl-quality`,
`gemini-3.1-flash-lite`, `gpt-5.4-nano`).

## 로컬 번역기

`translation: local` 은 `model`·`device`·`quant`·`context` 를 받는다. `quant` 는 지시형 LLM 백엔드의
정밀도(`4bit` | `8bit` | `none`)이고, 안 적으면 4bit 다. 모델을 비교하는 설정에는 적어 둔다 —
결과의 `config.yml` 만 보고 정밀도를 알 수 있어야 한다. 로컬 모델 네 조건(`gemma3-4b`,
`hy-mt2-1.8b`, `qwen3.5-4b`, `translategemma-4b`, 전부 4bit)의 설정도 API 번역기와 같은
`configs/mt-ko-en/` 에 있다.

## 대화 번역 품질 6축

`core/utils/metrics`는 다음 scorecard도 매 실행에서 계산한다. 각 축은 세 지표를 가지며
`summary.json.metrics.<axis>`에 중첩해 저장된다.

| 축 | 지표 |
|---|---|
| `meaning` | `comet`, `metricx_24`, `chrfpp` |
| `critical_information` | `normalized_value_accuracy`, `critical_span_f1`, `critical_fact_error_rate` |
| `fluency` | `spoken_fluency_judge`, `mqm_fluency_error_rate`, `target_lm_pseudo_perplexity` |
| `context` | `context_mqm`, `dialogue_inconsistency_rate`, `context_contrastive_accuracy` |
| `asr_robustness` | `quality_drop`, `translation_invariance`, `quality_noise_degradation` |
| `intent` | `polarity_preservation`, `speech_act_macro_f1`, `modality_stance_preservation` |

매니페스트 항목의 선택적인 `metric_inputs` 블록은 그대로 `items.jsonl`로 전달된다. 세부
계약과 예시는 [`core/utils/metrics/TRANSLATION_QUALITY.md`](../core/utils/metrics/TRANSLATION_QUALITY.md)에
있다. 주석이 없는 축은 값이 생기지 않고 `unavailable`에 구조화된 이유가 남는다.

COMET과 MetricX-24는 큰 learned metric이라 평범한 CPU 스모크 실행에서 자동 다운로드하지
않는다. `metric_inputs.meaning`에 사전 계산 값을 넣거나 각각 `STITY_COMET_MODEL`,
`STITY_METRICX_MODEL`을 설정한 scoring 환경에서 실행한다. masked-LM pseudo-perplexity도 같은
이유로 사전 계산 값 또는 `STITY_FLUENCY_LM`을 사용한다.

## 한↔영 중요 정보·유창성 주석 생성

기존 run의 후보 번역을 오프라인에서 보강한다. 원본 run은 수정하지 않으며 출력 디렉터리에
새 `items.jsonl`, `summary.json`, `quality_config.json`, `judge_usage.jsonl`을 만든다.

```bash
pip install -r bench/requirements-translation-metrics.txt
python -m bench.annotate_quality \
  bench/runs/<source-run> bench/runs/<source-run>-quality
```

판정기 키는 저장소 루트의 `.env`에 `OPENAI_API_KEY=...` 한 줄로 둔다. 셸에 같은 이름의
환경변수가 있으면 그쪽이 이긴다.

**판정기 비용은 호출마다 남는다.** `judge_usage.jsonl`에 호출 하나가 한 줄로 append 된다 —
용도(`purpose`), 모델, 입력·출력 토큰, 그 호출의 추정 비용과 누적 추정 비용. JSON 파싱에
실패해 재시도한 호출도 과금되므로 같이 기록한다. 중간에 죽어도 거기까지는 남고, `--overwrite`
로 다시 돌려도 이 파일은 지우지 않고 이어 쓴다. 25호출마다 누적 추정 비용이 로그에 찍히고,
끝나면 `summary.json.usage`에 합계가 들어간다. 단가표는
`core/meaning_segmentator/autoseg/infra/gateway.py`의 `_PRICES` 하나를 같이 쓴다 — 표에 없는
모델은 비용이 0으로 잡히므로 시작할 때 경고한다.

중요 정보는 한↔영만 지원한다. 날짜·시간·금액·수량·단위·서수·전화번호는 결정론적
추출기(`core/utils/metrics/critical_values.py`)로 canonical value를 만들고,
인명·지명·기관·제품·전문용어는 고정 JSON 판정기(`core/utils/metrics/judge.py`)로 뽑는다.
판정기는 두 번 부른다.

- **참조 span은 후보를 보지 않고 문장마다 한 번만 뽑는다.** 모든 번역기가 같은 참조 span으로
  채점돼야 서로 비교가 되기 때문이다. 결과는 `--reference-cache`(기본
  `bench/runs/_cache/critical_reference_spans.jsonl`)에 쌓이고, 다른 번역기 run은 이 파일에서
  꺼내 쓴다. 여러 run이 동시에 돌아도 파일에 **먼저 쓰인 값**을 모두가 쓴다.
- **후보 span은 그 고정된 참조 목록에 맞춰 뽑는다.** 같은 개체면 참조의 canonical value를
  그대로 쓰고, 참조에 없는 사실은 따로 뽑아 환각으로 잡는다. 규칙이 못 읽은 같은 값의 다른
  표현("a dozen" = 12)은 판정기가 채울 수 있다. 다만 규칙이 이미 읽은 값을 덮거나 참조에 없는
  값을 새로 만들 수는 없다.

span마다 `origin`(`gold`·`rule`·`llm`·`llm_value`)이 붙는다. 데이터셋에 사람이 검수한
`reference_spans`가 있으면 `origin: gold`로 우선 보존한다. 이전 주석 run이 자동으로 만든
span은 사람 정답으로 넘기지 않는다(`retranslate`도 버린다). 후보에 종속된 `candidate_spans`와
판정 provenance는 파생 run에만 저장된다.

유창성 판정기는 원문과 참조를 받지 않는다. 후보와 같은 대화의 앞선 목표 언어 최대 3턴만
보고 1~5 Spoken Fluency 점수와 MQM fluency/style 오류 span을 한 번에 생성한다. 판정기가
준 오류 위치가 틀리면 같은 문자열 중 가장 가까운 곳으로 옮기고, 후보에 없는 문자열은 버린
뒤 그 개수를 `mqm_unlocated_errors`로 남긴다. pseudo-perplexity 기본 checkpoint는 영어
`FacebookAI/roberta-base`, 한국어 `klue/roberta-base`다. 문장별 값과 함께 NLL 합과 토큰 수를
저장해 코퍼스 값은 토큰 기준으로 모은다. MQM 오류율과 pseudo-perplexity 모두 언어 간 값은
합치지 않고 `by_target`으로 보고한다. 모델을 받지 않을 실행은 `--skip-pseudo-perplexity`를
지정한다.

## 전사문 기반 ASR 강건성

오디오나 ASR을 다시 실행하지 않는다. **clean 입력은 정답 전사(`reference`)**다. 정답 전사가
없는 행만 `hypothesis`를 clean으로 쓰고 `clean_source`에 그렇게 적는다. 같은 번역기로 세 가지를
번역한다.

- clean 전사
- **실제 ASR 출력(`hypothesis`)** — 정답 전사와 다를 때만. `quality_drop`과
  `translation_invariance`는 이 조건 하나로 계산한다.
- clean 전사에 합성 노이즈를 강도별로 넣은 변형 — 품질 저하 곡선용

합성 노이즈(`core/utils/metrics/text_noise.py`)는 단어마다 요청 강도의 확률로 오류를 하나
넣는다. 종류는 치환 60%, 삭제 25%, 삽입(반복) 15%라서 실제로 잰 WER이 요청 강도에 가깝다.
치환은 혼동어 사전, 숫자 한 자리 바꾸기, 비슷한 소리로 바꾸기(한국어는 모음·받침·된소리,
영어는 복수·과거 어미와 모음) 중 하나다. 영어 모음 바꾸기는 사전에 없는 단어를 만들 수
있어서, 합성 곡선은 실제 ASR 출력을 보충하는 용도로 쓴다.

```bash
python -m bench.asr_text_robustness \
  bench/runs/<source-run> bench/configs/examples/quality_ko_en.yml \
  bench/runs/<source-run>-text-robustness \
  --noise-levels 0.08,0.18,0.32 --seed 20260921
```

품질은 문장 chrF++다. `relative_drop`은 **평균 하락폭 / 평균 clean 품질**이다(항목별 비율은
`per_item`에만 둔다 — clean 품질이 0에 가까운 항목 하나가 비율 평균의 부호를 뒤집는다).
clean/noisy 번역의 의미 invariance는 기본적으로
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` cosine similarity로 계산한다.

저하 곡선의 가로축은 요청 강도가 아니라 **각 변형의 실측 WER**이다. 기울기·AUC·최악 구간은
강도별 평균점으로 이은 곡선에서 구한다(`curve`에 점이 남는다). 합성 변형의 의미 유사도는
곡선의 `similarity` 지표로 따로 나온다.

실제 ASR 출력이 비었거나 번역이 비면 **치명적 실패**로 보고 빼지 않는다. 품질은 빈 문자열의
점수, 유사도는 0으로 들어가고 `catastrophic_failures`에 따로 센다. 번역기 자체가 오류를 낸
조건은 인프라 실패라서 drop과 invariance 양쪽에서 똑같이 빠진다. 모든 noisy 전사, 조작 목록,
실측 WER·CER, 번역, 모델과 seed는 결과에 남는다.
