# zero 시나리오

zero 파이프라인(`core/pipeline/zero.py`, 전사기 `core/components/transcription/zero_qwen3.py`)에서
보고된 문제를 하나씩 붙잡아 두는 곳이다. 파일 하나가 시나리오 하나다.

| | 무엇 |
|---|---|
| **시나리오** (`<이름>.yml`) | bench 실행에서 본 문제 하나. 무엇이 틀렸는지(`what`·`why`), 그 일이 난 음성 구간(`item`·`span_samples`, 읽기 쉽게 `span_sec`), 그때의 로그 줄이 든 기준선 파일 이름(`baseline`), 지금 맞아야 할 것(`checks`)과 보고된 문제(`known_issues`), 설정별 기대값(`when`), 그리고 아래 버전 필드 |
| **replay** | 그 구간의 음성만 진짜 파이프라인에 흘린다 — bench 처럼 200ms 조각으로, VAD·ASR 모델 그대로. 코드를 바꿀 때마다 돌려 그 지점이 어떻게 바뀌는지 본다 |

bench manager 웹의 "Replay"(저장된 실행을 휴대폰 화면으로 다시 보여 주기)와는 다른 것이다. 여기 replay 는 모델을 다시 돌린다.

## 돌리기 전에

- GPU 와 데이터셋(`STITY_DATA_ROOT`)이 있는 머신에서 저장소 루트에서 돌린다.
- 시나리오가 가리키는 파이프라인 설정은 S3 에 있다. 먼저 `make configs-pull` 로 받는다. replay 는 로컬 `configs/` 파일을
  읽고 bench 는 S3 를 읽으므로, 로컬 파일이 S3 와 다르면 replay 가 경고한다.
- replay 는 ASR(GPU 의 0.30), aligner, whisper-small 을 올린다. 12GB 쯤 필요하다. GPU 를 다른 사람이 쓰고 있거나 남은
  메모리가 모자라면 아무것도 올리지 않고 종료 코드 3 으로 멈춘다. 들어간다는 걸 알면 `--force-gpu`. 같은 머신의 bench
  worker queue 와도 GPU 를 나눠 쓴다.

## 돌리기

```bash
# 코드를 바꿔 효과를 볼 때: 지금 기준 설정으로 전부
python -m core.tests.zero_replay --pipeline configs/pipelines/zero.asr.koen+mt.mock_segalign-w1-lang-lid-w12.yml

python -m core.tests.zero_replay                          # 시나리오마다 자기 설정(pipeline)으로
python -m core.tests.zero_replay limit-cut-mid-switch     # 이름으로 골라서
```

지금 기준 설정은 `zero.asr.koen+mt.mock_segalign-w1-lang-lid-w12` 다. SEG 뒤에 단어 하나가 붙으면 확정하고 aligner 로
슬롯을 자르며, 세션 언어 밖 머리말로 시작하는 디코딩은 버리고, `langid` 부품(오디오 언어 timeline, 1·2초 창)이 찾은
언어 전환에서 슬롯을 자르는 설정이다.

코드 변경의 효과를 기록으로 남길 때는 **커밋한 뒤** replay 한다. 결과에 커밋 해시가 붙는데, 커밋 안 한 수정이 있으면
`+수정` 이 붙어 어느 코드였는지 알 수 없다.

## 결과 읽기

replay 는 구간마다 커밋을 기준선(`before`)과 지금(`now`) 나란히 보여 주고 검사한다.

| 찍히는 것 | 뜻 |
|---|---|
| `ok` / `FAIL` | `checks` — 이 설정에서 반드시 맞아야 하는 것 |
| `known` / `FIXED` | `known_issues` — 보고된 문제. `known` 이 정상이고, 맞기 시작하면 `FIXED` |
| `not applicable` | 이 설정이나 이 checkout 에는 해당하지 않아 건너뜀 (아래 버전 필드) |
| `behavior vs baseline` | 기준선과 같은 동작인지(`same`·`text-only`·`different`). 기준선이 없는 머신은 `no baseline` |
| `expectations for this pipeline (when …)` | `when` 블록의 기대값을 썼다 |

종료 코드: 0 은 `checks` 전부 통과, 1 은 하나라도 실패(`--gate` 면 기준선과 다른 동작이나 replay 된 시나리오가 없을 때도),
2 는 없는 시나리오 이름, 3 은 GPU 가 바쁨.

`STATUS.md` 는 시나리오마다 마지막 결과다. 이번 실행에서 건너뛴 시나리오는 예전 줄이 남고 `(이번 실행에 없음)` 이 붙는다.

### 설정별 기대값 (`when`)

같은 시나리오라도 어떤 설정에서는 문제가 고쳐져 있고 어떤 설정에서는 그대로다. 위쪽 `checks`·`known_issues` 는 모든
설정에 걸리고, `when` 블록은 `if:` 에 맞는 설정에서 그 두 목록을 통째로 바꾼다(처음 맞는 블록 하나).

```yaml
known_issues:
  - must-commit: {"texts": ["diplomatic"]}
when:
  - if: {"seg_confirm": true, "aligner": "set", "part.langid": "set"}
    checks:
      - must-commit: {"texts": ["diplomatic"]}
    known_issues: []
```

어떤 설정에서 `FIXED` 가 되면 위쪽 목록을 바꾸지 말고, 그 설정에 맞는 `when` 블록을 더해 거기서 `checks` 로 옮긴다.
알려진 실패도 같은 방법으로 적는다: 그 설정의 블록에서 `known_issues` 로 내린다(`real-repeat-kept` 가 그렇다 — aligner
로 자르는 설정에서는 두 번째 `Cheers` 가 짧은 조각이 되어 틀린다). 그래서 기준 설정으로 replay 해도 알려진 실패 때문에
종료 코드 1 이 나지는 않는다.

### 글자 검사

검사 이름과 받는 값은 `core/tests/zero_replay.py` 의 `CHECKS` 에 있다.

- 글자를 보는 검사(`must-commit`, `never-commits`, `language-of` …)는 양쪽 글자를 bench 의 r2 채점과 같은 정규화(군말·표기
  변형·사투리 어미·숫자 꼴)에 거친 뒤 띄어쓰기·문장부호·대소문자를 빼고 비교한다. `2차` 와 `이차`, `This` 와 `this` 는 같은 말이다.
- `texts` 안의 목록은 "이 중 하나" 다: `[["이 차", "2차"], "Cheers"]`.
- 짧은 글자는 다른 말 속에도 들어 있다. `"네"` 는 `"정도네요"` 에도 맞는다. 그 자리에만 있는 두세 단어로 쓴다(`"네 450"`).
- 헛말 시나리오에는 그 헛말이 없어졌는지와 함께 **다른 모양의 헛말로 바뀌지 않았는지**도 검사한다. 웃음이 `哈哈` 대신
  `Ha ha ha …` 반복으로 나와도 `哈` 와 언어만 보는 검사는 통과한다. `no-repeat-loop` 은 같은 말이 지정한 횟수보다 많이
  이어지면 틀린 것으로 본다.
- 기준선과 같은 동작인지 볼 때 커밋 시각은 0.15초 안이면 같다고 본다.

## 버전 필드

| 필드 | 뜻 |
|---|---|
| `found_in` | 문제를 찾은 bench 실행이 돈 커밋 (`zero_cut cut` 이 실행 기록에서 채운다) |
| `introduced_in` | `{commit, by}` — 원인이 들어온 커밋과 무엇 때문인지. 문제가 아니라 지켜야 할 동작이면 `commit: guard`. 저장소에 없는 해시면 replay 가 에러로 멈춘다 |
| `applies_when` | 이 시나리오가 뜻이 있는 설정. 키는 전사기 설정, 또는 파이프라인 부품이면 `part.<종류>`(`part.langid: "set"`). `true` 는 켜져 있음, `"set"` 은 값이 있음, 그 밖의 값은 같음. 맞지 않는 파이프라인으로 replay 하면 GPU 를 쓰기 전에 건너뛰고 "not applicable" 로 찍는다. 없으면 모든 파이프라인에 해당한다 |
| `pipeline`, `pipeline_version` | 잘라 온 실행의 설정 파일과 그 `meta.version`. 로컬 파일의 버전이 다르면 replay 가 경고한다 |

기능을 되돌리면 그 기능 때문에 생긴 시나리오는 빠져야 한다. 설정을 끄거나 기능 코드를 지우면 `applies_when` 이 맞지 않아
빠지고, `introduced_in` 커밋이 지금 checkout 의 이력에 없으면(그 커밋 없이 다시 만든 브랜치) 그것으로도 빠진다.
`git revert` 는 원래 커밋을 이력에 남기므로 커밋으로는 못 거른다 — 그래서 어떤 기능이 원인인 시나리오에는 **원인이 된
그 설정**을 `applies_when` 에 적는다. 기능을 켜는 설정만이 아니라 값까지(`seg_min_tail_words: 1` 처럼) 적어야 그 값이 아닌
설정에서 빠진다.

## 남는 것

| 경로 | 내용 | 어디에 |
|---|---|---|
| `<이름>.yml` | 시나리오 | git |
| `fixtures/<이름>.jsonl` | 기준선: 시나리오를 만든 실행의 그 구간 로그 줄과 구간 샘플 위치 | 로컬만 |
| `replays/<이름>/<시각>.jsonl` | replay 한 번의 모든 이벤트 — 커밋, `decode`(파이프라인이 `record_decodes: true` 일 때), VAD | 로컬만 |
| `results.jsonl` | replay 마다 시나리오별 한 줄: 커밋 해시, 파이프라인, 검사 결과, 수치 | 로컬만 |
| `STATUS.md` | 시나리오마다 마지막 결과 표 | 로컬만 |

기준선과 replay 기록에는 데이터셋 원문이 들어 있어 git 에도 S3 에도 올리지 않는다. yml 에는 구간 위치(`span_samples`)가
숫자로 들어 있어서, 기준선이 없는 머신에서도 replay 와 검사는 돈다. 그때는 기준선과의 비교만 `no baseline` 으로 빠진다.
`--gate` 는 비교가 목적이라 기준선이 없으면 통과하지 않는다.

**yml 과 커밋 메시지에는 원문을 길게 옮기지 않는다.** 검사에 꼭 필요한 짧은 구절(두세 단어, `"diplomatic"`, `"네 450"`)만
쓴다. `what` 에서 어느 말인지 가리킬 때도 그 구절로 가리킨다. `zero_cut scan` 은 반복 구절만 찍고, `replay` 출력에는
원문 커밋이 나오니 그대로 붙이지 않는다.

구간은 VAD flush 자리에서 시작한다(그 자리에서는 슬롯이 비어 있다). VAD 와 `langid` 는 앞 소리를 기억하므로, 구간 앞의
오디오를 둘에만 먼저 흘린 뒤 구간을 파이프라인에 흘린다 — 말 끝 판단과 언어 timeline 이 전체 실행 때와 같아진다.

## 시나리오 올리기 — 기준선과 같게 재현될 때만

새 시나리오를 올리거나 기존 시나리오를 고치면, 커밋 전에 지금 코드로 `--gate` replay 를 돌린다.

```bash
python -m core.tests.zero_replay --gate <시나리오 이름 ...>
```

시나리오마다 `behavior vs baseline` 이 찍히고, 끝에 커밋 메시지에 붙일 `--gate:` 줄들이 나온다. `same`(글자까지 같음)이나
`text-only`(커밋 수·이유·언어가 같고 시각이 0.15초 안, 글자만 조금 다름 — GPU 흔들림)면 올린다. `different` 면 올리지
않는다. 고른 이름이 없거나, 모두 해당 없음으로 건너뛰어 replay 된 시나리오가 없으면 `--gate` 는 실패한다. 기준과 할 일은
[.claude/rules/zero-scenario-gate.md](../../../.claude/rules/zero-scenario-gate.md) 에 있다.

## 시나리오 만들기

zero 파이프라인을 `record_decodes: true` 로 bench 에서 돌리면 events.jsonl 에 디코딩마다 `decode` 이벤트가 남는다.

```bash
python -m core.tests.zero_cut scan bench/runs/<데이터셋>/<파이프라인>          # 장면별 리셋·반복 루프·긴 슬롯
python -m core.tests.zero_cut cut  bench/runs/<데이터셋>/<파이프라인> <항목> <이름> --from 120 --to 150
```

- 구간은 VAD flush 자리로 맞춰진다. flush 가 드물면 요청한 것보다 훨씬 넓게 잘린다(그러면 알려 준다). 항목 처음부터
  자르려면 `--from` 을 뺀다.
- 만든 yml 의 `what`·`why`·`introduced_in` 을 채우고, 필요하면 `applies_when`, 본 문제를 `known_issues` 에 적는다.
- 같은 이름으로 다시 자르면 기준선과 yml 의 구간(`span_samples`·`span_sec`·`run`·`found_in`·`pipeline_version`)이 새로
  바뀌고, `what`·검사 같은 나머지는 그대로 남는다. 다시 자른 뒤에는 `--gate` 를 다시 돌린다.

### 순서 — bench 와 커밋

bench 는 `core/` 아래 커밋 안 된 파일이 있으면 돌지 않는다. 그래서:

1. 문제를 찾은 bench 는 지금 코드에서, 커밋 안 된 것이 없을 때 돌린다.
2. 그 실행에서 `zero_cut cut` 으로 시나리오를 만들고 `--gate` 를 돌린다. `same`·`text-only` 면 yml 을 커밋한다.
3. `different` 라서 bench 를 다시 돌려야 하면, 아직 커밋 안 한 새 yml 은 잠시 `core/` 밖으로 옮겨 두고 bench 를 돌린
   뒤 다시 자른다.
