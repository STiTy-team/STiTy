# bench/ — 설정 파일로 도는 벤치마크

설정 YAML 하나가 실행을 정의하고, 그 이름으로 레지스트리에서 파이프라인을 조립한다.
**WebSocket 서버를 띄우지 않는다** — VAD·커밋 로직·번역기를 전부 그대로 쓰되 한 프로세스
안에서 직접 조립한다.

```bash
# 데이터셋은 별도 리포다. 체크아웃한 디렉토리가 곧 STITY_DATA_ROOT 다.
git clone git@github.com:STiTy-team/datasets.git ~/datasets
export STITY_DATA_ROOT=~/datasets
make dataset DATA=fleurs                       # 내려받고 변환까지

make dry   CONFIG=bench/configs/fleurs.yml     # 설정만 확인. 모델 안 올린다
make bench CONFIG=bench/configs/fleurs.yml
make test                                      # 단위 테스트 (GPU 불필요)
```

데이터셋은 `fleurs` 와 `acl6060` 둘이다. 계약(`dataset.yml` + `manifest.jsonl`)과 새
코퍼스 붙이는 법은 그 리포의 README 에 있다. bench 에는 데이터셋별 분기가 없다.

## 설정

```yaml
name: acl6060-en-de-seg

dataset:
  name: acl6060
  limit: 20

languages:
  lang: en                     # 또는 map: {en: ko, ko: en, fr: ko}
  target: de

stity:
  transcription:
    name: qwen3
    model: baseline
    chunk_size_sec: 2.0        # 모델별 설정은 그 모델 밑에 둔다
    max_new_tokens: 128
  translation:
    name: google               # none | google | local | gpt
  commit: seg                  # seg | punct | static
  vad: {enabled: true, min_silence_ms: 800}
  gpu_memory_utilization: 0.5

metrics: [wer, fsl, laal, bleu, commit]
logs: {top_k: 10, rank_by: laal}
```

`translation: gpt` 처럼 문자열만 쓰면 `{name: gpt}` 로 풀린다.

**모델별 설정은 그 모델 밑에 쓴다.** 청크 크기나 토큰 예산은 체크포인트에 딸린 값이라
최상위로 평평하게 펴지 않는다. 컴포넌트가 모르는 키를 주면 **에러로 죽는다** — 조용히
버려지면 `max_new_tokens: 256` 을 적어 놓고 아무 일도 안 일어난 채 그럴듯한 숫자가 나온다.

### `commit` 은 명시적으로 풀린다

| `commit` | `always_commit` | `enable_dot_commit` | `dot_commit_confirm` | `hide_seg` |
|---|---|---|---|---|
| `seg` | false | false | false | false |
| `punct` | false | true | true | true |
| `static` | true | false | false | true |

프로덕션 서버는 `enable_dot_commit` 기본값을 **가중치 경로에서 유도한다**
(`_infer_dot_commit_default`). 그래서 같은 명령이 체크포인트에 따라 다른 정책으로 돈다.
bench 는 `parse_args` 를 부르지 않고 위 표로만 푼다. 해석된 값은 결과 파일에 전부 남는다.

`hide_seg` 는 SEG 를 뱉는 가중치로 `punct`/`static` 축을 돌릴 때 필요하다. 없으면 축이
조용히 섞인다 — 실측으로 punct 축 커밋의 36%가 seg 였고, 그 커밋만 확정 게이트를 건너뛰어
지연이 실제보다 좋게 잡혔다.

### 여러 언어가 섞인 대화

```yaml
languages:
  map: {en: ko, ko: en, fr: ko}
```

`map` 의 키가 그대로 ASR 의 `allowed_languages` 가 되어 언어 이름 토큰에 로짓 바이어스를
건다. 즉 **언어 설정은 번역만이 아니라 WER 도 바꾼다.** `lang`/`target` 쌍으로는 세 언어를
각각 다른 곳으로 보낼 수 없으므로(서버의 `parse_lang_map` docstring이 그렇게 적고 있다),
섞인 데이터셋에는 `map` 을 쓴다. 둘을 동시에 주면 죽는다 — 점수가 달라지는 두 경로다.

`metrics: [routing]` 이 언어 판정과 라우팅을 잰다. BLEU 는 **올바르게 라우팅된 세그먼트만**
으로 내고(`bleu`), 전체 기준은 `bleu_all` 로 따로 낸다. 한국어 출력을 프랑스어 참조와
비교하면 번역 품질 문제와 언어 판정 문제가 한 숫자에 섞여 둘 다 못 읽는다.

## 나오는 것

| 파일 | 무엇 | git |
|---|---|---|
| `results/<name>.json` | 요약 + 해석된 설정 전부 | 추적한다 (작다, 실행 비교가 diff 로 보인다) |
| `items/<name>-<ts>.jsonl` | 항목 단위 행. append | 안 한다 |
| `logs/<name>-<ts>.jsonl` | **이벤트 전부. 이게 원본이다** | 안 한다 |
| `logs/<name>-<ts>.replay.json` | 상위 k개 항목의 리플레이 | 안 한다 |

요약과 리플레이는 이벤트 스트림을 다시 읽어 만든 **투영**이다. 전부 append 라 중간에
죽어도 그때까지가 남고 채점된다.

`results/<name>.json` 의 `config.resolved` 에는 설정에 안 쓴 값까지 **실제로 쓰인 값**이
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

## 지표

| 지표 | 비고 |
|---|---|
| `wer` | 주 숫자. 빈 가설을 전체 삭제로 센다 |
| `wer_scored_only` | 옛 숫자(빈 가설 제외). 대조용 |
| `cer` | 문자 오류 합 / 참조 문자 합 |
| `fsl` | vad 커밋은 `min_silence_ms` 만큼 더한 정규화값도 함께 |
| `laal` | `decision_audio_sec` 이 `d_i` 다 |
| `bleu` | 라우팅이 맞은 것만. 전체는 `bleu_all` |
| `commit` | 사유별 개수·비율. `finish_ratio` 가 크면 축의 커밋 경로가 안 도는 것이다 |
| `routing` | 언어 판정 정확도, 라우팅 정확도, 혼동 행렬 |

`wer` 과 `wer_scored_only` 를 둘 다 내는 이유: 기존 `compute_wer_for_rows` 는 가설이 빈
행을 버리고(`scoring.py:19`), `process_batch` 가 그 행을 또 버린다. 실패한 발화가 두 번
숨어서 점수가 좋아 보인다. 실측으로 한쪽이 완전 실패인 두 발화에서 0.0 과 0.5 가 갈린다.

요청한 지표를 계산할 수 없으면 **모델을 올리기 전에** 죽는다. `sacrebleu` 가 없거나 번역
참조가 없는데 `bleu` 를 요청하면 0초에 알려준다. 네 시간 뒤에 `null` 을 보는 것보다 낫다.

## 구조

```
__main__.py   CLI. --dry-run 은 모델 전에 끝난다
config.py     YAML → dataclass. 점수를 바꾸는 값 전부 명시 해석
registry.py   이름 → 컴포넌트
driver.py     ASR 서버를 import 하는 유일한 모듈
events.py     emit() + EventSink + JSONL 핸들러
report.py     이벤트 스트림 → results / replay
metrics/      asr · latency · translation · commit · routing
data/         manifest · audio · build(변환기용 헬퍼)
components/   transcription · translation
```

`metrics/`·`config.py`·`data/`·`report.py` 는 `driver.py` 를 import 하지 않는다.
그래야 GPU 없이 테스트가 돈다.

데이터셋 리포는 반대로 **STiTy 를 import 하지 않는다.** 두 리포가 나란히 체크아웃돼
있다는 가정은 곧 깨진다. 변환기는 스스로 검사하고, 최종 판정만 이쪽이 한다:

```bash
python -m bench.data.manifest --validate $STITY_DATA_ROOT/fleurs
make validate DATA=fleurs
```

## 아직 안 되는 것

- **동시 실행.** v1 은 순차 고정이다. `asr_lock` 이 생성을 직렬화하고 flush 가 겹친다.
- **서버가 버리는 커밋을 싱크로 잡기.** 로그로는 보이므로 이벤트 줄기에서 읽는다.
- **WebSocket 경로와의 대조 검증(S6).** 데이터와 가중치가 있는 머신에서 해야 한다.
  같은 항목을 두 경로로 돌려 **전사 문자열과 커밋 사유 분포가 같은지** 보는 단계이고,
  이걸 통과하기 전에는 bench 숫자를 보고서에 쓰지 않는다.
