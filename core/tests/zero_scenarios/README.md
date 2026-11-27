# zero 시나리오

zero 파이프라인(`core/pipeline/zero.py`, 전사기 `core/components/transcription/zero_qwen3.py`)에서
보고된 문제를 하나씩 붙잡아 두는 곳이다. 파일 하나가 시나리오 하나다.

| | 무엇 |
|---|---|
| **시나리오** (`<이름>.yml`) | bench 실행에서 본 문제 하나. 무엇이 틀렸는지(`what`·`why`), 그 일이 난 음성 구간(`item`·`span_sec`), 그때의 로그 줄(`baseline`), 지금 맞아야 할 것(`checks`)과 보고된 문제(`known_issues`) |
| **replay** | 그 구간의 음성만 진짜 파이프라인에 흘린다 — bench 처럼 200ms 조각으로, VAD·ASR 모델 그대로. 코드를 바꿀 때마다 돌려 그 지점이 어떻게 바뀌는지 본다 |

GPU 와 데이터셋(`STITY_DATA_ROOT`)이 있는 머신에서 저장소 루트에서 돌린다. 코드를 바꿔 효과를 볼 때는 지금 기준
설정 `zero.asr.koen+mt.mock_segalign-w1-lang`(SEG 뒤에 단어 하나가 붙으면 확정, aligner 로 슬롯을 자름, 세션 언어 밖
머리말로 시작하는 디코딩은 버림)과 비교한다.

```bash
python -m core.tests.zero_replay                          # 전부
python -m core.tests.zero_replay limit-cut-mid-switch     # 이름으로 골라서
python -m core.tests.zero_replay --pipeline configs/pipelines/<다른 설정>.yml   # 다른 파이프라인으로
```

replay 는 구간마다 커밋을 기준선(`before`)과 지금(`now`) 나란히 보여 주고 검사한다.

- `checks` 는 지금 반드시 맞아야 한다. 하나라도 틀리면 종료 코드 1.
- `known_issues` 는 보고된 문제다. 틀리는 게 정상이고, 맞기 시작하면 `FIXED` 로 찍힌다 — 그때 `checks` 로 옮긴다.

검사 이름과 받는 값은 `core/tests/zero_replay.py` 의 `CHECKS` 에 있다. 헛말 시나리오에는 그 헛말이 없어졌는지와 함께
**다른 모양의 헛말로 바뀌지 않았는지**도 검사한다 — 웃음이 `哈哈` 대신 `Ha ha ha …` 반복으로 나와도 `哈` 와 언어만 보는
검사는 통과한다. `no-repeat-loop` 은 같은 말이 몇 번 넘게 이어지면 틀린 것으로 본다. 글자를 보는 검사(`must-commit`,
`never-commits`, `language-of` …)는 양쪽 글자를 bench 의 r2 채점과 같은 정규화(군말·표기 변형·사투리 어미·숫자
꼴)에 거친 뒤 띄어쓰기·문장부호·대소문자를 빼고 비교한다 — `2차` 와 `이차`, `This` 와 `this` 는 같은 말이다.
`texts` 안의 목록은 "이 중 하나" 다: `[["이 차", "2차"], "Cheers"]`.

## 남는 것 (전부 로컬)

| 경로 | 내용 |
|---|---|
| `replays/<이름>/<시각>.jsonl` | replay 한 번의 모든 이벤트 — 커밋, `decode`(파이프라인이 `record_decodes: true` 일 때), VAD |
| `results.jsonl` | replay 마다 시나리오별 한 줄: 커밋 해시, 파이프라인, 검사 결과, 수치 |
| `STATUS.md` | 시나리오마다 마지막 결과 표. 실행마다 다시 쓴다 |
| `fixtures/<이름>.jsonl` | 기준선: 시나리오를 만든 실행의 그 구간 로그 줄과 구간 샘플 위치 |

기준선과 replay 는 데이터셋 원문이 들어 있어 git 에 올리지 않는다. yml 만 올린다. 기준선이 없는 머신에서는 그
시나리오를 건너뛴다.

구간은 VAD flush 자리에서 시작한다(그 자리에서는 슬롯이 비어 있다). VAD 는 앞 소리를 기억하므로, 구간 앞의
오디오를 VAD 에만 먼저 흘린 뒤 구간을 파이프라인에 흘린다 — 말 끝 판단이 전체 실행 때와 같아진다.

## 시나리오 올리기 — 기준선과 같게 재현될 때만

새 시나리오를 올리거나 기존 시나리오를 고치면, 커밋 전에 지금 코드로 `--gate` replay 를 돌린다.

```bash
python -m core.tests.zero_replay --gate <시나리오 이름 ...>
```

시나리오마다 `behavior vs baseline` 이 찍힌다. `same`(글자까지 같음)이나 `text-only`(커밋 수·이유·언어·시각이 모두
같고 글자만 조금 다름 — GPU 흔들림)면 올린다. `different` 면 올리지 않는다. 기준과 할 일은
[.claude/rules/zero-scenario-gate.md](../../../.claude/rules/zero-scenario-gate.md) 에 있다.

## 시나리오 만들기

zero 파이프라인을 `record_decodes: true` 로 bench 에서 돌리면 events.jsonl 에 디코딩마다 `decode` 이벤트가 남는다.

```bash
python -m core.tests.zero_cut scan bench/runs/<데이터셋>/<파이프라인>          # 장면별 리셋·반복 루프·긴 슬롯
python -m core.tests.zero_cut cut  bench/runs/<데이터셋>/<파이프라인> <항목> <이름> --from 120 --to 150
```

구간은 VAD flush 자리로 맞춰진다. 만든 뒤 yml 의 `what`·`why` 를 채우고, 본 문제를 `known_issues` 에 적는다.
bench 는 `core/` 아래 커밋 안 된 파일이 있으면 돌지 않으므로, 새 yml 은 bench 를 돌리기 전에 커밋한다.
