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
      "xcomet": {"score": 0.91, "error_spans": []},
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
      "target_lm_pseudo_perplexity": 4.2
    },
    "context": {
      "mqm": {"score": 5, "score_without_context": 3, "errors": []},
      "inconsistencies": [],
      "contrastive": {"selected_correct": true, "phenomenon": "ellipsis"}
    },
    "asr_robustness": {
      "clean_quality": {"xcomet": 0.93, "metricx_24": 1.4},
      "noisy_quality": {"xcomet": 0.88, "metricx_24": 2.1},
      "similarity": 0.94,
      "noise_level": 0.18,
      "quality_by_noise": [
        {"noise_level": 0.0, "quality": {"xcomet": 0.93}},
        {"noise_level": 0.18, "quality": {"xcomet": 0.88}}
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

## Learned metric 실행

- 공통 선택 의존성: `pip install -r bench/requirements-translation-metrics.txt`
- XCOMET: `pip install unbabel-comet`, `STITY_XCOMET_MODEL=Unbabel/XCOMET-XL`
- MetricX-24: 공식 `google-research/metricx` 저장소의 `metricx24`를 `PYTHONPATH`에 두고
  `STITY_METRICX_MODEL=google/metricx-24-hybrid-large-v2p6`
- pseudo-perplexity: `transformers`, `torch`와 목표 언어를 지원하는 masked-LM을 설치하고
  `STITY_FLUENCY_LM=<checkpoint>`

환경 변수가 없으면 모델을 다운로드하지 않는다. 세 지표 모두 사전 계산 값을 받을 수 있어
GPU 벤치와 CPU 집계를 분리할 수 있다. XCOMET과 MetricX 결과에는 모델 이름, chrF++에는
sacreBLEU signature, pseudo-perplexity에는 LM checkpoint를 기록한다.

기존 ASR 결과에 번역기만 다시 적용할 때는 다음처럼 translation-only run을 만든다.

```bash
python -m bench.retranslate \
  bench/runs/<source-run> \
  bench/retranslate.example.yml
```

이 경로는 commit 경계와 ASR 문자열을 고정한다. 후보 출력에 종속된 기존 `metric_inputs`는
자동으로 폐기하고 reference 주석만 전달하므로 이전 번역의 판정이 새 번역 점수로 섞이지 않는다.

## 구현된 오프라인 생성 단계

한↔영 중요 정보와 유창성 주석은 원본 run을 수정하지 않고 다음 명령으로 파생 run에 만든다.

```bash
python -m bench.annotate_quality <source-run> <output-run>
```

- 중요 정보: 결정론적 값 추출·canonical normalization + JSON entity/span 판정기
- 유창성: 후보만 보는 Spoken Fluency 1~5 judge + MQM 오류 span + masked-LM pseudo-perplexity
- 기본 masked-LM: 영어 `FacebookAI/roberta-base`, 한국어 `klue/roberta-base`

전사문 교란 기반 강건성 평가는 다음 명령으로 별도 파생 run을 만든다.

```bash
python -m bench.asr_text_robustness \
  <source-run> bench/quality_ko_en.example.yml <output-run>
```

이 프로토콜은 오디오와 ASR을 재실행하지 않는다. 저장된 전사에 seed가 고정된 ASR 유사
교란을 적용하고 clean/noisy 입력을 같은 번역기로 처리한다. 각 강도의 chrF++ 품질,
quality drop, degradation slope/AUC, clean/noisy 번역 의미 유사도를 저장한다. 기본 의미
유사도 checkpoint는 `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`다.

## 방향

- 높을수록 좋음: XCOMET, chrF++, value accuracy, span F1, fluency judge,
  Context-MQM, contrastive accuracy, invariance, intent metrics
- 낮을수록 좋음: MetricX-24, critical fact error rate, MQM fluency error rate,
  pseudo-perplexity, quality drop, degradation slope

서로 다른 목표 언어의 pseudo-perplexity 절대값은 비교하지 않는다. `quality_drop`과 slope는
MetricX 계열처럼 낮을수록 좋은 지표의 방향을 내부적으로 뒤집어, 양수가 곧 품질 저하가 되게
보고한다.
