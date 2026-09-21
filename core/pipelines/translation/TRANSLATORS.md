# 번역 부품 — API 번역기와 대화 문맥

`translation` 레지스트리의 부품을 bench 설정에서 어떻게 고르고, 모든 번역기가 어떤 문맥을
받는지 적는다. 부품을 쓰는 법 자체는 [../../CLAUDE.md](../../CLAUDE.md) 의 `pipelines/` 절.

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
`Dialogue` 하나가 맡고, `bench` 의 cascade 파이프라인과 `core.utils.metrics.retranslate` 가 같은 것을 쓴다.

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
