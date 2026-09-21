# Conversational translation metric contract

대화 번역 품질 지표는 기존 `score_run`에 연결돼 있으며 GPU가 없어도 import와 집계가 된다.
큰 모델은 함수 안에서 지연 import하므로, 일반 WER/CER 실행과 config 검증에는 영향을 주지
않는다.

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

- 공통 선택 의존성: `pip install -r bench/requirements-translation-metrics.txt`
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

기존 ASR 결과에 번역기만 다시 적용할 때는 다음처럼 translation-only run을 만든다.

```bash
python -m bench.retranslate \
  bench/runs/<source-run> \
  bench/configs/examples/retranslate.yml
```

이 경로는 commit 경계와 ASR 문자열을 고정한다. 후보 출력에 종속된 기존 `metric_inputs`는
자동으로 폐기하고 reference 주석만 전달하므로 이전 번역의 판정이 새 번역 점수로 섞이지 않는다.
무엇이 사람 정답으로 남는지는 `gold_inputs`가 정한다. 데이터셋의 사람 span은 `origin`이 없거나
`gold`다. 이전 주석 run이 만든 span은 다음 주석 run이 다시 만들므로 넘기지 않고, `origin`이
기록되기 전의 블록(`annotation_source`는 있는데 span에 `origin`이 없음)은 사람 것과 구별할 수
없어서 하나도 넘기지 않는다.

## 구현된 오프라인 생성 단계

한↔영 중요 정보와 유창성 주석은 원본 run을 수정하지 않고 다음 명령으로 파생 run에 만든다.

```bash
python -m bench.annotate_quality <source-run> <output-run>
```

- 중요 정보: 결정론적 값 추출·canonical normalization(`critical_values.py`) + JSON
  entity/span 판정기(`judge.py`, 주석은 `critical_information.py`). 참조 span은 후보 없이
  문장마다 한 번 뽑아 모든 번역기가 공유하고, 후보 span은 그 목록에 맞춰 뽑는다. span마다
  `origin`이 남는다
- 유창성: 후보만 보는 Spoken Fluency 1~5 judge + MQM 오류 span(주석은 `fluency.py`) +
  masked-LM pseudo-perplexity
- 기본 masked-LM: 영어 `FacebookAI/roberta-base`, 한국어 `klue/roberta-base`

판정기가 준 문자 위치는 그 자리에 정말 그 문자열이 있을 때만 믿는다. 아니면 같은 문자열 중
판정기가 말한 위치에 가장 가까운 곳으로 옮기고, 후보에 아예 없으면 버린 뒤
`*_unlocated_*`로 센다. 같은 단어가 두 번 나오는 반복 오류를 잃지 않으려는 것이다.

참조 span 캐시(`ReferenceSpanCache`)는 append만 하는 JSONL이다. 여러 주석 run이 같은 문장을
동시에 놓쳐 둘 다 판정기를 부르면 답이 다를 수 있다. 그래서 쓴 뒤 파일을 다시 읽어 그
문장의 **처음 쓰인 값**을 모두가 쓴다. 판정기는 규칙이 못 읽은 참조 값의 다른 표현("a dozen"
= 12)만 후보 쪽에 채울 수 있고, 규칙이 읽은 값을 덮거나 참조에 없는 값을 만들 수는 없다.

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

전사문 교란 기반 강건성 평가는 다음 명령으로 별도 파생 run을 만든다.

```bash
python -m bench.asr_text_robustness \
  <source-run> bench/configs/examples/quality_ko_en.yml <output-run>
```

이 프로토콜은 오디오와 ASR을 재실행하지 않는다. 정답 전사를 clean으로, 저장된 ASR 출력을
실제 noisy 조건으로 쓰고, 정답 전사에 seed가 고정된 ASR 유사 교란(치환 중심)을 강도별로
넣은 변형(`text_noise.py`)을 더한다. 셋 다 같은 번역기로 처리해 chrF++ 품질, quality drop, 실측 WER 기준
degradation slope/AUC, clean/noisy 번역 의미 유사도를 저장한다. 기본 의미 유사도
checkpoint는 `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`다.

## 방향

- 높을수록 좋음: COMET, chrF++, value accuracy, span F1, fluency judge,
  Context-MQM, contrastive accuracy, invariance, intent metrics
- 낮을수록 좋음: MetricX-24, critical fact error rate, MQM fluency error rate,
  pseudo-perplexity, quality drop, degradation slope

서로 다른 목표 언어의 pseudo-perplexity와 MQM 오류율 절대값은 비교하지 않는다. `quality_drop`과 slope는
MetricX 계열처럼 낮을수록 좋은 지표의 방향을 내부적으로 뒤집어, 양수가 곧 품질 저하가 되게
보고한다.
