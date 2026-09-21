# Conversational translation metric contract

대화 번역 품질 지표는 기존 `score_run`에 연결돼 있으며 GPU가 없어도 import와 집계가 된다.
큰 모델은 함수 안에서 지연 import하므로, 일반 WER/CER 실행과 config 검증에는 영향을 주지
않는다.

여섯 축이 각각 세 지표를 가지며 `summary.json.metrics.<axis>`에 중첩해 저장된다. 주석이 없는
축은 값이 생기지 않고 `unavailable`에 구조화된 이유가 남는다.

| 축 | 지표 |
|---|---|
| `meaning` | `comet`, `metricx_24`, `chrfpp` |
| `critical_information` | `normalized_value_accuracy`, `critical_span_f1`, `critical_fact_error_rate` |
| `fluency` | `spoken_fluency_judge`, `mqm_fluency_error_rate`, `target_lm_pseudo_perplexity` |
| `context` | `context_mqm`, `dialogue_inconsistency_rate`, `context_contrastive_accuracy` |
| `asr_robustness` | `quality_drop`, `translation_invariance`, `quality_noise_degradation` |
| `intent` | `polarity_preservation`, `speech_act_macro_f1`, `modality_stance_preservation` |

## 매니페스트 입력

각 `manifest.jsonl` 항목에 선택적인 `metric_inputs`를 넣을 수 있다. 아래는 모든 축을 보여
주기 위한 축약 예시다. 실제 주석은 사람이 검수한 canonical value와 고정된 판정기 출력이어야
한다.

```json
{
  "id": "dialogue-1-turn-3",
  "audio": "audio/turn-3.wav",
  "duration": 2.4,
  "src_lang": "ko",
  "reference": {
    "transcript": "오후 세 시에 강남역에서 만나요",
    "translations": {"en": "Let's meet at Gangnam Station at 3 PM."}
  },
  "metric_inputs": {
    "meaning": {
      "comet": {"score": 0.84},
      "metricx_24": 1.7
    },
    "critical_information": {
      "reference_spans": [
        {"type": "time", "text": "오후 세 시", "canonical_value": "15:00"},
        {"type": "location", "text": "강남역", "canonical_value": "gangnam station",
         "accepted_values": ["gangnam stn."]}
      ],
      "candidate_spans": [
        {"type": "time", "text": "3 PM", "canonical_value": "15:00"},
        {"type": "location", "text": "Gangnam Station", "canonical_value": "gangnam station"}
      ]
    },
    "fluency": {
      "judge": {"score": 5, "reason": "natural spoken English"},
      "mqm_errors": [],
      "target_token_count": 9,
      "target_lang": "en",
      "target_lm_pseudo_perplexity": 4.2,
      "target_lm_nll_sum": 14.35,
      "target_lm_token_count": 10
    },
    "context": {
      "mqm": {"errors": [], "errors_without_context": [{"severity": "major"}]},
      "inconsistencies": [],
      "contrastive": {"selected_correct": true, "phenomenon": "ellipsis"}
    },
    "asr_robustness": {
      "clean_quality": {"comet": 0.86, "metricx_24": 1.4},
      "noisy_quality": {"comet": 0.81, "metricx_24": 2.1},
      "similarity": 0.94,
      "noise_level": 0.18,
      "catastrophic": false,
      "quality_by_noise": [
        {"noise_level": 0.0, "requested_noise_level": 0.0, "quality": {"comet": 0.86}},
        {"noise_level": 0.13, "requested_noise_level": 0.18, "quality": {"comet": 0.81}}
      ]
    },
    "intent": {
      "polarity": {"reference": "positive", "candidate": "positive"},
      "speech_act": {"reference": ["suggestion"], "candidate": ["suggestion"]},
      "modality": {"reference": "certain", "candidate": "certain"}
    }
  }
}
```

`candidate_spans`, judge 결과, 후보 intent label처럼 시스템 출력이 있어야 생성 가능한 필드는
오프라인 판정 단계가 `items.jsonl`에 보강한 뒤 다시 `score_run`에 넣어도 된다. bench의
dataset adapter는 `metric_inputs`를 해석하거나 버리지 않고 전달만 한다.

## 지표 정의에서 헷갈리기 쉬운 것

- **COMET·MetricX의 원문**은 정답 전사(`reference`)다. 정답 전사가 없는 항목만 ASR 전사를
  쓴다. ASR 전사는 ASR 시스템마다 달라서, 그걸 원문으로 쓰면 시스템마다 다른 원문으로
  채점된다. 결과의 `source_text`에 각각 몇 건인지 남는다.
- **Critical Fact Error Rate**의 분모는 참조나 후보 **어느 한쪽에라도** 중요 span이 있는
  발화다. 참조에 사실이 없는데 후보가 번호나 이름을 지어낸 경우도 오류로 센다
  (`invented_only_items`).
- **MQM fluency error rate**와 **pseudo-perplexity**는 목표 언어별(`by_target`)로만 합친다.
  토큰 단위(영어 단어, 한국어 어절)와 언어 모델이 달라서 언어끼리 섞으면 뜻이 없다. 언어가
  하나뿐일 때만 최상위 값이 생긴다. 저장된 pseudo-perplexity는 문장별 NLL 합과 토큰 수가
  있으면 토큰 기준으로 모으고(`aggregation: token_weighted`), 없으면 문장 값의 기하평균으로
  모은다(`sentence_geometric_mean`). 산술평균은 문장 하나에 끌려가서 쓰지 않는다.
  pseudo-perplexity가 낮다고 좋은 번역은 아니다 — 뻔하고 짧은 문장일수록 낮다.
- **Context-MQM** 점수는 MQM 점수(심각도 가중 오류 합에 마이너스, 0이 최고)다. `errors`
  목록을 주면 유창성 축과 같은 가중치(minor 1, major 5, critical 25)로 계산한다. 숫자
  `score`를 직접 줄 때는 0 이하여야 한다. 1~5 같은 루브릭 점수가 섞이면 방향이 뒤집히므로
  양수는 거부한다. `errors_without_context`(또는 `score_without_context`)가 있으면 문맥을 준
  점수와의 차가 `mean_context_gain`이다.
- **ASR 강건성**의 `quality_drop`과 `similarity`는 clean(정답 전사)과 실제 ASR 출력 한 쌍으로
  계산한다. 저하 곡선의 `noise_level`은 **실제로 잰 WER**이고, 같은
  `requested_noise_level`을 가진 점들의 평균이 곡선의 한 점이다. 기울기·AUC·최악 구간은 그
  평균 곡선에서 구한다. 빈 ASR 출력·빈 번역은 `catastrophic: true`로 평균에 포함하고
  `catastrophic_failures`에 따로 센다.

## Learned metric 실행

- 공통 선택 의존성: `pip install -r core/utils/metrics/requirements.txt`
- COMET: `pip install unbabel-comet`, `STITY_COMET_MODEL=Unbabel/wmt22-comet-da`. 참조 기반
  회귀 모델이라 문장 점수와 코퍼스 평균을 내고 오류 span은 내지 않는다. 게이트가 없어 Hugging
  Face 라이선스 동의 없이 받아진다
- MetricX-24: 공식 `google-research/metricx` 저장소의 `metricx24`를 `PYTHONPATH`에 두고
  `STITY_METRICX_MODEL=google/metricx-24-hybrid-large-v2p6`. 디코더 캐시(`use_cache`)는 끄고
  올린다 — 공식 모델의 디코더가 self/cross attention 캐시를 한 칸에 같이 써서
  `transformers==4.57.6`에서는 켠 채로 돌리면 텐서 크기 불일치로 죽는다
- COMET 체크포인트를 못 받으면 `unavailable["meaning.comet"]`에 Hub가 돌려준 실제 사유
  (`GatedRepoError` 등)가 남는다. 게이트 모델을 지정했다면 로그인한 계정으로 그 모델 페이지에서
  라이선스에 동의해야 한다
- pseudo-perplexity: `transformers`, `torch`와 목표 언어를 지원하는 masked-LM을 설치하고
  `STITY_FLUENCY_LM=<checkpoint>`

환경 변수가 없으면 모델을 다운로드하지 않는다. 세 지표 모두 사전 계산 값을 받을 수 있어
GPU 벤치와 CPU 집계를 분리할 수 있다. COMET과 MetricX 결과에는 모델 이름, chrF++에는
sacreBLEU signature, pseudo-perplexity에는 LM checkpoint를 기록한다.

## 기존 bench run 에서 파생 run 만들기

세 CLI 가 bench run(`bench/runs/<name>`)을 읽어 번역을 다시 하거나 주석을 붙인다. 이 패키지에서
run 형식을 아는 것은 이 CLI 들과 `derived.py`·`translation_config.py` 뿐이다. 집계 함수는
여전히 run·설정·보고서 형식을 모른다.

```bash
make retranslate SOURCE=bench/runs/<원본> \
  CONFIG=bench/configs/examples/retranslate.yml
make annotate-quality SOURCE=bench/runs/<원본> OUTPUT=bench/runs/<원본>-quality
make asr-text-robustness SOURCE=bench/runs/<원본> \
  CONFIG=bench/configs/examples/quality_ko_en.yml OUTPUT=bench/runs/<원본>-text-robustness
```

| 파일 | 하는 일 |
|---|---|
| `retranslate.py` | 저장된 확정 조각을 번역기만 바꿔 다시 번역하는 실행. 결과는 `bench/runs/<설정 name>` |
| `translation_config.py` | retranslate 설정 — 번역기 하나, 언어 한 쌍, `context_scope` |
| `annotate_quality.py` | 중요 정보·유창성 주석을 붙인 파생 run |
| `asr_text_robustness.py` | 정답 전사·실제 ASR 출력·합성 교란 전사를 번역한 강건성 파생 run |
| `derived.py` | 원본 run 읽기, 파생 run 쓰기, `bench/runs` 위치 |

번역 부품(API 번역기, 대화 문맥) 설정은
[../../pipelines/translation/TRANSLATORS.md](../../pipelines/translation/TRANSLATORS.md).

## 번역기만 다시 돌리기 — `retranslate`

원본 run의 `items.jsonl`을 읽고 `segments[*].original`을 같은 순서로 새 번역기에 전달한다.
오디오, VAD, ASR은 로드하지 않는다 — commit 경계와 ASR 문자열이 고정된 채 번역기만 바뀐다.
원본에 commit segment가 없는 옛 결과만 전체 `hypothesis`를 한 세그먼트로 사용하는 fallback이
명시적으로 기록된다.

`context_scope: item`은 매 발화에서 문맥을 초기화한다 — 같은 발화 안에서 먼저 확정된 조각만
문맥이 된다. `context_scope: group`은 같은 대화 group의 앞 발화에서 확정된 문장(원문·번역)까지
이어서 전달한다. **`make bench` 의 cascade 파이프라인은 `group` 과 같게 돈다.** 두 조건은 결과
config와 비교 없이 섞으면 안 된다.

새 run의 `summary.json.source_run`에는 원본 경로, `items.jsonl` SHA-256, 원본 commit과 dataset
정보가 남는다. ASR timing은 원본 segment의 `source_timing`으로만 보존하고 새 run의
WER/FSL/LAAL로 보고하지 않는다 — 그 시계는 이번 실행에서 잰 것이 아니다. 새로 측정하는 시간은
segment별 `translation_elapsed_sec`과 집계된 `translation_runtime`뿐이다. GPU 에서 돌면
`translation_runtime.cuda_memory`에 모델을 올린 직후 할당량과 번역 중 최대 예약량이 남는다.
4bit 모델은 bf16 조각을 거쳐 올라가서 올리는 순간의 최댓값은 번역기가 실제로 쥐는 양이 아니므로,
올린 뒤 최댓값을 초기화하고 잰다. torch 가 import 되지 않은 CPU 실행에는 이 값이 없다.

원본의 후보 번역에 종속된 COMET, span, judge, intent 판정은 새 번역에 재사용하지 않는다.
사람이 만든 reference span/label만 전달되고, 새 후보 주석을 생성하기 전까지 관련 지표는
`unavailable`로 남는다. 무엇이 사람 정답으로 남는지는 `metric_inputs.gold_inputs`가 정한다.
데이터셋의 사람 span은 `origin`이 없거나 `gold`다. 이전 주석 run이 만든 span은 다음 주석 run이
다시 만들므로 넘기지 않고, `origin`이 기록되기 전의 블록(`annotation_source`는 있는데 span에
`origin`이 없음)은 사람 것과 구별할 수 없어서 하나도 넘기지 않는다.

## 한↔영 중요 정보·유창성 주석 — `annotate_quality`

기존 run의 후보 번역을 오프라인에서 보강한다. 원본 run은 수정하지 않으며 출력 디렉터리에
새 `items.jsonl`, `summary.json`, `quality_config.json`, `judge_usage.jsonl`을 만든다.

```bash
pip install -r core/utils/metrics/requirements.txt
python -m core.utils.metrics.annotate_quality \
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
추출기(`critical_values.py`)로 canonical value를 만들고, 인명·지명·기관·제품·전문용어는 고정
JSON 판정기(`judge.py`, 주석은 `critical_information.py`)로 뽑는다. 판정기에 주는 원문은 정답
전사다 — ASR 전사는 ASR 시스템마다 달라서 참조 span이 시스템마다 달라진다. 판정기는 두 번
부른다.

- **참조 span은 후보를 보지 않고 문장마다 한 번만 뽑는다.** 모든 번역기가 같은 참조 span으로
  채점돼야 서로 비교가 되기 때문이다. 결과는 `--reference-cache`(기본
  `bench/runs/_cache/critical_reference_spans.jsonl`)에 쌓이고, 다른 번역기 run은 이 파일에서
  꺼내 쓴다. 캐시(`ReferenceSpanCache`)는 append만 하는 JSONL이다. 여러 run이 같은 문장을
  동시에 놓쳐 둘 다 판정기를 부르면 답이 다를 수 있어서, 쓴 뒤 파일을 다시 읽어 그 문장의
  **처음 쓰인 값**을 모두가 쓴다.
- **후보 span은 그 고정된 참조 목록에 맞춰 뽑는다.** 같은 개체면 참조의 canonical value를
  그대로 쓰고, 참조에 없는 사실은 따로 뽑아 환각으로 잡는다. 규칙이 못 읽은 같은 값의 다른
  표현("a dozen" = 12)은 판정기가 채울 수 있다. 다만 규칙이 이미 읽은 값을 덮거나 참조에 없는
  값을 새로 만들 수는 없다.

span마다 `origin`(`gold`·`rule`·`llm`·`llm_value`)이 붙는다. 데이터셋에 사람이 검수한
`reference_spans`가 있으면 `origin: gold`로 우선 보존한다. 후보에 종속된 `candidate_spans`와
판정 provenance는 파생 run에만 저장된다.

판정기가 준 문자 위치는 그 자리에 정말 그 문자열이 있을 때만 믿는다. 아니면 같은 문자열 중
판정기가 말한 위치에 가장 가까운 곳으로 옮기고, 텍스트에 아예 없으면 버린 뒤
`*_unlocated_*`로 센다. 같은 단어가 두 번 나오는 반복 오류를 잃지 않으려는 것이다.

결정론적 추출기는 평범한 단어이기도 한 형태를 숫자로 읽지 않는다. 참조와 후보가 같은 규칙을
지나므로 규칙이 못 읽는 표현은 양쪽이 똑같이 못 읽지만, 숫자가 아닌 말을 숫자로 읽으면 맞는
번역이 치명적 오류가 되기 때문이다.

- 한국어 "네"(대답)·"이"(지시어, "이 분"·"이 번")·"한국"의 "한", 영어 대명사 "one",
  "COVID-19"의 19는 숫자가 아니다.
- 관형사형 수사(한·두·세·네·스무)와 동사·형용사이기도 한 열·쉰은 뒤에 단위 명사가 올 때만,
  영어 "one"은 뒤에 단위나 통화가 올 때만 숫자다. 한 글자 한자어 수사는 단위 명사와 한 칸
  띄어 있을 때만 숫자다("오 분"). 붙여 쓰면 "사원" 같은 단어다.
- 라틴 문자 단위·통화는 단어가 끝나야 한다("5 m"는 되고 "5 more"는 안 된다). 한글 단위와
  기호는 조사가 붙어도 되고("70킬로미터를", "3만 원이"), 숫자는 단위에 붙어 있어도 된다
  ("35mm").
- 한국어 큰 단위는 숫자와 띄어 쓸 수 있고("6 만 원"), 억·만 뒤에는 띄어서 다음 묶음이 올 수
  있다("1억 2천만"). 띄어 쓴 단위는 거기서 단어가 끝나거나 다른 단위·통화·단위 명사로
  이어져야 한다. 그래서 "3 백화점"은 3이다.
- 시각의 "시"·"분" 뒤에는 조사나 서술격 어미만 온다. "시간"·"시작"·"시장"은 시각이 아니다.
  "p.m."은 끝의 마침표까지 시각이고, 점 없는 "PM."의 마침표는 문장 끝으로 남는다.
- "a second"·"per second"의 second는 서수가 아니다. 한국어 "두 번"은 서수가 아니라 2이고,
  "번째"·"째"·"제N"만 서수다.

유창성 판정기는 원문과 참조를 받지 않는다. 후보와 같은 대화의 앞선 목표 언어 최대 3턴만
보고 1~5 Spoken Fluency 점수와 MQM fluency/style 오류 span(주석은 `fluency.py`)을 한 번에
생성한다. pseudo-perplexity 기본 checkpoint는 영어 `FacebookAI/roberta-base`, 한국어
`klue/roberta-base`다. 문장별 값과 함께 NLL 합과 토큰 수를 저장해 코퍼스 값은 토큰 기준으로
모은다. MQM 오류율과 pseudo-perplexity 모두 언어 간 값은 합치지 않고 `by_target`으로 보고한다.
모델을 받지 않을 실행은 `--skip-pseudo-perplexity`를 지정한다.

## 전사문 기반 ASR 강건성 — `asr_text_robustness`

오디오나 ASR을 다시 실행하지 않는다. **clean 입력은 정답 전사(`reference`)**다. 정답 전사가
없는 행만 `hypothesis`를 clean으로 쓰고 `clean_source`에 그렇게 적는다. 같은 번역기로 세 가지를
번역한다.

- clean 전사
- **실제 ASR 출력(`hypothesis`)** — 정답 전사와 다를 때만. `quality_drop`과
  `translation_invariance`는 이 조건 하나로 계산한다.
- clean 전사에 합성 노이즈를 강도별로 넣은 변형 — 품질 저하 곡선용

합성 노이즈(`text_noise.py`)는 단어마다 요청 강도의 확률로 오류를 하나 넣는다. 종류는 치환
60%, 삭제 25%, 삽입(반복) 15%라서 실제로 잰 WER이 요청 강도에 가깝다. 치환은 혼동어 사전,
숫자 한 자리 바꾸기, 비슷한 소리로 바꾸기(한국어는 모음·받침·된소리, 영어는 복수·과거 어미와
모음) 중 하나다. 영어 모음 바꾸기는 사전에 없는 단어를 만들 수 있어서, 합성 곡선은 실제 ASR
출력을 보충하는 용도로 쓴다.

```bash
python -m core.utils.metrics.asr_text_robustness \
  bench/runs/<source-run> bench/configs/examples/quality_ko_en.yml \
  bench/runs/<source-run>-text-robustness \
  --noise-levels 0.08,0.18,0.32 --seed 20260921
```

품질은 문장 chrF++다. `relative_drop`은 **평균 하락폭 / 평균 clean 품질**이다(항목별 비율은
`per_item`에만 둔다 — clean 품질이 0에 가까운 항목 하나가 비율 평균의 부호를 뒤집는다).
clean/noisy 번역의 의미 invariance는 기본적으로
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` cosine similarity로 계산한다. 두
번역이 다 있으면 인코더가 정하고, 한쪽만 비었으면 clean 의미가 하나도 안 남은 것이라 0, 둘 다
비었으면 출력이 바뀌지 않은 것이라 1이다.

저하 곡선의 가로축은 요청 강도가 아니라 **각 변형의 실측 WER**이다. 기울기·AUC·최악 구간은
강도별 평균점으로 이은 곡선에서 구한다(`curve`에 점이 남는다). 합성 변형의 의미 유사도는
곡선의 `similarity` 지표로 따로 나온다.

실제 ASR 출력이 비었거나 번역이 비면 **치명적 실패**로 보고 빼지 않는다. 품질은 빈 문자열의
점수, 유사도는 0으로 들어가고 `catastrophic_failures`에 따로 센다. 번역기 자체가 오류를 낸
조건은 인프라 실패라서 drop과 invariance 양쪽에서 똑같이 빠진다. 모든 noisy 전사, 조작 목록,
실측 WER·CER, 번역, 모델과 seed는 결과에 남는다.

## 방향

- 높을수록 좋음: COMET, chrF++, value accuracy, span F1, fluency judge,
  Context-MQM, contrastive accuracy, invariance, intent metrics
- 낮을수록 좋음: MetricX-24, critical fact error rate, MQM fluency error rate,
  pseudo-perplexity, quality drop, degradation slope

서로 다른 목표 언어의 pseudo-perplexity와 MQM 오류율 절대값은 비교하지 않는다. `quality_drop`과 slope는
MetricX 계열처럼 낮을수록 좋은 지표의 방향을 내부적으로 뒤집어, 양수가 곧 품질 저하가 되게
보고한다.
