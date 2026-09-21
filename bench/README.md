# bench/ — 설정 파일로 도는 벤치마크

설정 YAML 하나가 실행을 정의하고, 그 이름으로 레지스트리에서 파이프라인을 조립한다.
**WebSocket 서버를 띄우지 않는다** — VAD·커밋 로직·번역기를 전부 그대로 쓰되 한 프로세스
안에서 직접 조립한다.

```bash
# 데이터셋은 별도 리포다. 체크아웃한 디렉토리가 곧 STITY_DATA_ROOT 다.
git clone git@github.com:STiTy-team/datasets.git ~/datasets
export STITY_DATA_ROOT=~/datasets
bash $STITY_DATA_ROOT/fleurs/install.sh        # 내려받고 변환까지

make bench CONFIG=<설정>.yml                        # 실행
make replay RUN=bench/runs/<이름>                   # 그 실행을 브라우저에서 다시 본다
make retranslate SOURCE=bench/runs/<원본> \
  CONFIG=bench/retranslate.example.yml              # 저장된 ASR로 번역만 재실행
```

`make replay` 는 http://localhost:9130 에 페이지 하나를 띄운다. 왼쪽은 휴대폰이 `final` 로
그린 화면 그대로(번역이 본문, 전사가 그 아래), 오른쪽은 **휴대폰에는 안 보이는** 이벤트
줄기 전부다. 둘 다 같은 시계 — 이벤트의 `audio` 위치 — 가 움직인다.

### 저장된 ASR로 번역기만 비교

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

### API 번역기 — DeepL · Gemini · GPT

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

데이터셋은 `fleurs` 와 `acl6060` 둘이다. 계약(`dataset.yml` + `manifest.jsonl`)과 새
코퍼스 붙이는 법은 그 리포의 README 에 있다. bench 에는 데이터셋별 분기가 없다.

## 설정

```yaml
name: acl6060-en-de-seg

dataset:
  name: acl6060
  limit: 20

languages:
  lang: en
  target: de

stity:
  pipeline:
    name: cascade
    transcription:
      name: qwen3
      chunk_size_sec: 2.0      # 모델별 설정은 그 모델 밑에 둔다
      max_new_tokens: 128
    translation: {name: local, model: google/madlad400-3b-mt}
    vad: {name: silero, min_silence_ms: 800}
  commit: seg                  # seg | punct | always
  gpu_memory_utilization: 0.5
```

**부품은 파이프라인 안에 쓴다.** transcription·translation·vad 는 파이프라인의 인자이지
형제가 아니다. 다른 파이프라인이 쓰지 않는 부품을 받는 파이프라인도 자기 options 에 적으면
되고, 바깥이 그 종류를 먼저 알아야 할 일이 없다.

`translation: gpt` 처럼 문자열만 쓰면 `{name: gpt}` 로 풀린다.

**오디오를 어떻게 먹이는지는 설정이 아니다.** 청크 200ms, 클립 뒤에 붙이는 침묵 4000ms,
실시간 속도 — 셋 다 `__main__.py` 의 상수다. 이건 클라이언트를 기술하는 값이지 재는
대상이 아니고, 실행마다 다르게 먹인 두 결과는 애초에 비교가 안 된다. 쓰인 값은
`summary.json` 의 `pacing` 에 남는다.

**모델별 설정은 그 모델 밑에 쓴다.** 청크 크기나 토큰 예산은 체크포인트에 딸린 값이라
최상위로 평평하게 펴지 않는다. 컴포넌트가 모르는 키를 주면 **에러로 죽는다** — 조용히
버려지면 `max_new_tokens: 256` 을 적어 놓고 아무 일도 안 일어난 채 그럴듯한 숫자가 나온다.

`translation: local` 은 `model`·`device`·`quant`·`context` 를 받는다. `quant` 는 지시형 LLM 백엔드의
정밀도(`4bit` | `8bit` | `none`)이고, 안 적으면 4bit 다. 모델을 비교하는 설정에는 적어 둔다 —
결과의 `config.yml` 만 보고 정밀도를 알 수 있어야 한다. 로컬 모델 네 조건(`gemma3-4b`,
`hy-mt2-1.8b`, `qwen3.5-4b`, `translategemma-4b`, 전부 4bit)의 설정도 API 번역기와 같은
`configs/mt-ko-en/` 에 있다.

### `commit` 은 명시적으로 풀린다

| `commit` | `always_commit` | `enable_dot_commit` | `hide_seg` |
|---|---|---|---|
| `seg` | false | false | false |
| `punct` | false | true | true |
| `always` | true | false | true |

프로덕션 서버는 `enable_dot_commit` 기본값을 **가중치 경로에서 유도한다**
(`_infer_dot_commit_default`). 그래서 같은 명령이 체크포인트에 따라 다른 정책으로 돈다.
bench 는 `parse_args` 를 부르지 않고 위 표로만 푼다. 해석된 값은 결과 파일에 전부 남는다.

`hide_seg` 는 SEG 를 뱉는 가중치로 `punct`/`always` 축을 돌릴 때 필요하다. 없으면 축이
조용히 섞인다 — 실측으로 punct 축 커밋의 36%가 seg 였고, 그 커밋만 확정 게이트를 건너뛰어
지연이 실제보다 좋게 잡혔다.

### 한 실행은 한 방향이다

`lang` → `target` 한 쌍이다. 한 서버가 한 모델이고 모델은 언어로 대상을 고르지 않는다 —
프로덕션에서는 클라이언트가 접속하는 포트가 그 선택이다. 그래서 한 실행의 방향도 하나다.

`lang`/`target` 은 그대로 ASR 의 `allowed_languages` 가 되어 언어 이름 토큰에 로짓 바이어스를
건다. 즉 **언어 설정은 번역만이 아니라 WER 도 바꾼다.** (`restrict: false` 로 끌 수 있다.)

`routing` 지표가 언어 판정과 라우팅을 잰다. BLEU 는 **올바르게 라우팅된 세그먼트만**
으로 내고(`bleu`), 전체 기준은 `bleu_all` 로 따로 낸다. 한국어 출력을 프랑스어 참조와
비교하면 번역 품질 문제와 언어 판정 문제가 한 숫자에 섞여 둘 다 못 읽는다.

## 나오는 것

**설정 하나가 디렉토리 하나다.** 이름은 설정의 `name` 그대로고 타임스탬프가 붙지 않는다.
같은 설정을 다시 돌리면 그 디렉토리를 **덮어쓴다** — 그 설정의 현재 답이 하나만 남는다는
뜻이다. 지우려면 그 디렉토리만 지우면 된다.

```
runs/<name>/
  config.yml                        이 결과를 낸 설정. 실행이 복사해 넣는다
  summary.json                      이 실행의 요약
  items.jsonl                       항목 단위 행. append (죽어도 채점된다)
  events.jsonl                      이벤트 전부. 이게 원본이고 리플레이가 읽는 것이다
  translation_usage.jsonl           API 번역기의 호출별 비용. append, 지우지 않는다
```

실행이 시작할 때 `translation_usage.jsonl` 을 뺀 네 파일을 먼저 지운다. 스트림은 append 로 열리므로(중간에 죽어도
거기까지 채점된다) 안 지우면 지난 실행 뒤에 이어 붙어 두 실행이 한 기록으로 섞인다.

| 파일 | git |
|---|---|
| `runs/<name>/config.yml` | 추적한다 (무엇을 돌렸는지) |
| `runs/<name>/summary.json` | 추적한다 (작다, 실행 비교가 diff 로 보인다) |
| 나머지 | 안 한다 |

요약과 리플레이는 이벤트 스트림을 다시 읽어 만든 **투영**이다. 전부 append 라 중간에
죽어도 그때까지가 남고 채점된다.

`summary.json` 의 `config.resolved` 에는 설정에 안 쓴 값까지 **실제로 쓰인 값**이
들어간다. 이게 없으면 두 실행을 비교할 수 없다.

## 이벤트

표준 `logging` 위에 올려서 한 줄 JSON 으로 흘린다. 시각 `t` 는 **그 항목의 첫 오디오
청크로부터의 초**다(벽시계가 아니다) — 서버가 내는 모든 시간이 그 원점에서 재므로
`fsl`/`laal` 검산이 그 시계에서만 성립한다.

서버 로그가 같은 줄기에 섞인다. 그래서 서버가 `send_message` 전에 버리는 커밋
(`[SILENCE-DROP]`, `[HALLUC-DROP]`, `[EMPTY-DROP]`, `[TAIL-DROP]`, `[AST-LATE]`)이
**공짜로 보인다** — 소켓 모양의 싱크로는 아예 안 보이는 것들이다. final 이 0건인 항목의
이유를 여기서 찾는다.

빈 전사와 실패한 항목은 버리지 않는다. `n_empty_hypothesis` 로 세고 리플레이에 **항상**
넣는다(top-k 와 무관하게).

**`partial` 은 아직 확정되지 않은 줄이다.** 커밋만 기록하면 한 발화가 끝에서 통째로
튀어나오는 실행만 남는다 — 화면에 실제로 보이는 것도, 이 시스템이 하려는 것도 그게
아니다. 서버와 같은 모양으로 낸다: 통째로 교체할 전체 문자열, 120ms 간격 제한,
프레임마다 강제 재동기화(그게 없으면 청크마다 첫 콜백 하나 — 글자 몇 개짜리 — 만
남는다). 빈 문자열은 "지우라"는 신호이고 커밋 직후에만 나간다. 리플레이의 휴대폰
화면에서 흐린 말풍선이 이것이다.

## 지표

| 지표 | 비고 |
|---|---|
| `wer` | 주 숫자. 빈 가설을 전체 삭제로 센다 |
| `wer_scored_only` | 옛 숫자(빈 가설 제외). 대조용 |
| `cer` | 문자 오류 합 / 참조 문자 합 |
| `fsl` | 커밋이 오디오보다 얼마나 늦게 도착했나 = `recv_elapsed_sec − decision_audio_sec`. **기록하지 않고 유도한다** — 두 시계가 이미 있으니 파이프라인이 따로 내면 어긋날 수 있다 |
| `laal` | `decision_audio_sec` 이 `d_i` 다 |
| `bleu` | 라우팅이 맞은 것만. 전체는 `bleu_all`. **발화 하나가 한 쌍**이다 — 세그먼트를 다시 이어 붙여 채점한다 |
| `commit` | 사유별 개수·비율. `finish_ratio` 가 크면 축의 커밋 경로가 안 도는 것이다 |
| `routing` | 언어 판정 정확도, 라우팅 정확도, 혼동 행렬 |

`wer` 과 `wer_scored_only` 를 둘 다 내는 이유: 기존 `compute_wer_for_rows` 는 가설이 빈
행을 버리고(`scoring.py:19`), `process_batch` 가 그 행을 또 버린다. 실패한 발화가 두 번
숨어서 점수가 좋아 보인다. 실측으로 한쪽이 완전 실패인 두 발화에서 0.0 과 0.5 가 갈린다.

**지표는 고르지 않는다. 매 실행이 낼 수 있는 것을 전부 계산한다.** 설정으로 부분집합을
고르는 길은 없다 — 이미 GPU 시간을 치르고 얻은 숫자를 가릴 뿐이다.

**데이터가 못 받치는 지표는 없는 채로 둔다. 실패하지 않는다.** 값이 아예 빠지고
`diagnostics.unavailable` 에 이유가 남는다. `null` 은 나오지 않는다 — 지표는 숫자이거나
이유이고, 읽는 사람이 빈칸을 추측할 일은 없다.

| 데이터 | 나오는 것 | 빠지는 것 |
|---|---|---|
| 오디오만 (전사·번역 참조 없음) | `fsl`·`commit`·`routing` | `wer`·`cer`·`bleu`·`laal` |
| 전사는 있고 번역 참조 없음 | 위 + `wer`·`cer` | `bleu`·`laal` |
| 일부 언어만 번역 참조 있음 | 그 언어들의 `bleu` | 참조 없는 언어 (`bleu_by_target` 의 이유에 적힌다) |

### 대화 번역 품질 6축

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

### 한↔영 중요 정보·유창성 주석 생성

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
추출기(`bench/critical_values.py`)로 canonical value를 만들고, 인명·지명·기관·제품·전문용어는
고정 JSON 판정기로 뽑는다. 판정기는 두 번 부른다.

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

### 전사문 기반 ASR 강건성

오디오나 ASR을 다시 실행하지 않는다. **clean 입력은 정답 전사(`reference`)**다. 정답 전사가
없는 행만 `hypothesis`를 clean으로 쓰고 `clean_source`에 그렇게 적는다. 같은 번역기로 세 가지를
번역한다.

- clean 전사
- **실제 ASR 출력(`hypothesis`)** — 정답 전사와 다를 때만. `quality_drop`과
  `translation_invariance`는 이 조건 하나로 계산한다.
- clean 전사에 합성 노이즈를 강도별로 넣은 변형 — 품질 저하 곡선용

합성 노이즈는 단어마다 요청 강도의 확률로 오류를 하나 넣는다. 종류는 치환 60%, 삭제 25%,
삽입(반복) 15%라서 실제로 잰 WER이 요청 강도에 가깝다. 치환은 혼동어 사전, 숫자 한 자리
바꾸기, 비슷한 소리로 바꾸기(한국어는 모음·받침·된소리, 영어는 복수·과거 어미와 모음) 중
하나다. 영어 모음 바꾸기는 사전에 없는 단어를 만들 수 있어서, 합성 곡선은 실제 ASR 출력을
보충하는 용도로 쓴다.

```bash
python -m bench.asr_text_robustness \
  bench/runs/<source-run> bench/quality_ko_en.example.yml \
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

참조 번역이 없으면 `laal` 도 빠진다. 분모가 `max(|Y_hyp|, |Y_ref|)` 라서 `|Y_ref|` 를 빼면
근사가 아니라 **다른 지표(AL)** 가 되고, AL 은 짧게 생성할수록 점수가 좋아지는 구멍이 있다.
그걸 `laal_ms` 칸에 적으면 비교가 불가능한 두 숫자가 한 열에 섞인다.

## 구조

```
__main__.py   CLI + 실행. ASR 서버를 import 하는 유일한 모듈이고, 그 import 는
              Engine 안에서 늦게 일어난다 — 설정 오류는 모델을 올리기 전에 걸린다
config.py     YAML 블록 하나가 클래스 하나. 점수를 바꾸는 값 전부 명시 해석
report.py     이벤트 스트림 → summary.json
replay.py     이벤트 스트림 → :9130 웹 페이지 (replay.html 이 화면 전부)
dataset.py    dataset.yml + manifest.jsonl 읽기
```

채점은 `core.utils.metrics` 가 한다. bench 에는 지표 모듈이 없고 `__main__.py` 의
`score_item`·`score_run` 둘이 전부다 — 설정에서 무엇을 계산할지 고르는 자리가 없으니
고를 코드도 없다.

`config.py`·`dataset.py`·`report.py`·`replay.py` 는 `__main__.py` 를
import 하지 않는다.
그래야 GPU 없이 돌릴 수 있다.

데이터셋 리포는 반대로 **STiTy 를 import 하지 않는다.** 두 리포가 나란히 체크아웃돼
있다는 가정은 곧 깨진다.

## 아직 안 되는 것

- **동시 실행.** v1 은 순차 고정이다. `asr_lock` 이 생성을 직렬화하고 flush 가 겹친다.
- **서버가 버리는 커밋을 싱크로 잡기.** 로그로는 보이므로 이벤트 줄기에서 읽는다.
- **WebSocket 경로와의 대조 검증(S6).** 데이터와 가중치가 있는 머신에서 해야 한다.
  같은 항목을 두 경로로 돌려 **전사 문자열과 커밋 사유 분포가 같은지** 보는 단계이고,
  이걸 통과하기 전에는 bench 숫자를 보고서에 쓰지 않는다.
