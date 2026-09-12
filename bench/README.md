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
make replay RUN=bench/runs/<이름>-<ts>              # 그 실행을 브라우저에서 다시 본다
```

`make replay` 는 http://localhost:3000 에 페이지 하나를 띄운다. 왼쪽은 휴대폰이 `final` 로
그린 화면 그대로(번역이 본문, 전사가 그 아래), 오른쪽은 **휴대폰에는 안 보이는** 이벤트
줄기 전부다. 둘 다 같은 시계 — 이벤트의 `audio` 위치 — 가 움직인다.

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

**모델별 설정은 그 모델 밑에 쓴다.** 청크 크기나 토큰 예산은 체크포인트에 딸린 값이라
최상위로 평평하게 펴지 않는다. 컴포넌트가 모르는 키를 주면 **에러로 죽는다** — 조용히
버려지면 `max_new_tokens: 256` 을 적어 놓고 아무 일도 안 일어난 채 그럴듯한 숫자가 나온다.

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

실행 하나가 디렉토리 하나다. 지우려면 그 디렉토리만 지우면 된다.

```
runs/<name>-<ts>/
  summary.json                      이 실행의 요약 + 해석된 설정 전부
  items.jsonl                       항목 단위 행. append (죽어도 채점된다)
  events.jsonl                      이벤트 전부. 이게 원본이고 리플레이가 읽는 것이다
```

| 파일 | git |
|---|---|
| `runs/<name>-<ts>/summary.json` | 추적한다 (작다, 실행 비교가 diff 로 보인다) |
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

## 지표

| 지표 | 비고 |
|---|---|
| `wer` | 주 숫자. 빈 가설을 전체 삭제로 센다 |
| `wer_scored_only` | 옛 숫자(빈 가설 제외). 대조용 |
| `cer` | 문자 오류 합 / 참조 문자 합 |
| `fsl` | 커밋이 오디오보다 얼마나 늦게 도착했나 = `recv_elapsed_sec − decision_audio_sec`. **기록하지 않고 유도한다** — 두 시계가 이미 있으니 파이프라인이 따로 내면 어긋날 수 있다. `realtime: false` 로 돌리면 오디오보다 앞서 끝나 음수가 나온다 |
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

참조 번역이 없으면 `laal` 도 빠진다. 분모가 `max(|Y_hyp|, |Y_ref|)` 라서 `|Y_ref|` 를 빼면
근사가 아니라 **다른 지표(AL)** 가 되고, AL 은 짧게 생성할수록 점수가 좋아지는 구멍이 있다.
그걸 `laal_ms` 칸에 적으면 비교가 불가능한 두 숫자가 한 열에 섞인다.

## 구조

```
__main__.py   CLI + 실행. ASR 서버를 import 하는 유일한 모듈이고, 그 import 는
              Engine 안에서 늦게 일어난다 — 설정 오류는 모델을 올리기 전에 걸린다
config.py     YAML 블록 하나가 클래스 하나. 점수를 바꾸는 값 전부 명시 해석
report.py     이벤트 스트림 → summary.json
replay.py     이벤트 스트림 → :3000 웹 페이지 (replay.html 이 화면 전부)
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
