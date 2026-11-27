# zero 시나리오

zero 전사기(`core/components/transcription/zero_qwen3.py`)의 슬롯 리셋을 GPU 없이 확인하는 시나리오들이다.
파일 하나가 시나리오 하나다.

```bash
python -m core.tests.test_zero_qwen3                    # 전부
python -m core.tests.test_zero_qwen3 limit-30s          # 이름으로 골라서
```

돌릴 때마다 결과가 `results.jsonl` 에 한 줄씩 쌓이고(커밋 해시, 검사 결과, 수치), 시나리오마다 마지막 결과를
`STATUS.md` 에 표로 다시 쓴다. 둘 다 git 에 올리지 않는다.

## 두 종류

| `source` | 모델 대신 | 확인할 수 있는 것 |
|---|---|---|
| `script` | 사람이 짠 말(`talk`). 슬롯 안에 온전히 들어온 단어만 받아 적고, 문장 끝 뒤 쉼을 0.3초 들으면 `<SEG>` 를 쓴다 | 리셋·커밋 로직 — 단어가 빠지거나 겹치는지, 슬롯이 얼마나 커지는지 |
| `replay` | bench 실행에서 녹화한 진짜 모델 출력 | 진짜 출력(할루시네이션 포함)을 로직이 어떻게 커밋하는지 |

`replay` 는 로직을 고치면 슬롯 위치나 앞부분 텍스트가 바뀌어 녹화에 없는 디코딩이 생긴다. 그러면 그
시나리오는 `STALE` 로 찍히고 검사하지 않는다 — 녹화가 더는 모델이 할 일을 말해 주지 않으니, 고친 코드로
bench 를 다시 돌려 같은 구간을 다시 잘라 와야 한다. 녹화 밖 디코딩 수는 `misses` 로 남는다.

## 검사

- `checks` 는 지금 반드시 맞아야 한다. 하나라도 틀리면 종료 코드 1.
- `known_issues` 는 찾았지만 아직 안 고친 문제다. 틀리는 게 정상이고, 맞기 시작하면 `FIXED` 로 찍힌다 — 그때 `checks` 로 옮긴다.
- 모든 시나리오에 두 검사가 저절로 붙는다: 엔진이 어긋나지 않았는지, 두 번 돌려도 같은지.

검사 이름과 받는 값은 `core/tests/test_zero_qwen3.py` 의 `CHECKS` 에 있다.

## script 시나리오 쓰기

```yaml
what: 무슨 상황인지 한 줄
why: 왜 만들었는지 — 어디서 찾은 문제인지, 언제
source: script
talk:
  - sentence: {first: 1, count: 5, start: 0.2}            # 단어1..단어5, 0.4초 단어 + 0.1초 쉼
  - sentence: {first: 6, count: 3, start: 4.0, repeat: 4, every: 2.0}   # 같은 모양 4번, 2초 간격
  - sentence: {first: 20, count: 70, start: 9.0, closed: false}         # 문장 끝 없이
  - sentence: {first: 1, count: 5, start: 12.0, lang: en}             # word1..word5, 영어
  - word: {text: 단어99, start: 30.0, end: 30.4}
model:
  empty_at: [6.0]   # 이 시각에 끝나는 디코딩은 빈 출력 (선택)
  guess: true       # 0.5초 안 되는 슬롯에 단어 앞소리만 있으면 ~<단어> 로 짐작해 적는다 (선택)
vad_ends: [4.2]     # 중간 VAD flush 시각 (선택)
until: 8.0          # 여기까지 100ms 조각으로 흘려 넣고 VAD flush
checks: [every-word-once]
known_issues:
  - longest-slot-under: {sec: 10}
```

가짜 모델은 앞 150ms 가 슬롯 밖에 있는 단어까지는 듣는다. 언어 머리말은 슬롯 첫 단어의 언어로 한 번만 쓴다
(진짜 모델처럼).

디코딩은 2초마다 온다. 디코딩 시각(2, 4, 6초 …)과 단어·쉼의 시각이 어떻게 겹치는지가 시나리오의 전부다 —
의도한 상황이 맞는지 `slot-never-restarted` 같은 전제 검사를 같이 넣는다.

## replay 시나리오 만들기

zero 파이프라인을 `record_decodes: true` 로 bench 에서 돌리면 events.jsonl 에 디코딩마다 `decode` 이벤트가 남는다.

```bash
python -m core.tests.zero_cut scan bench/runs/<데이터셋>/<파이프라인>          # 반복 루프·긴 슬롯 찾기
python -m core.tests.zero_cut cut  bench/runs/<데이터셋>/<파이프라인> <항목> <이름> --from 120 --to 150 \
    --what "..." --why "..."
```

구간은 VAD flush 자리로 맞춰진다(그 자리에서는 슬롯이 비어 있어 앞 기록이 필요 없다). 만든 뒤 yml 의
`what`·`why` 를 채우고 `known_issues` 에 그 구간에서 본 문제(`must-commit`, `never-commits`, `allowed-languages`,
`language-of`)를 적는다.

yml 은 git 에 올린다. 재생 자료 `fixtures/<이름>.jsonl` 은 데이터셋 원문이 들어 있어 올리지 않는다 — 자료가
없는 머신에서는 그 시나리오를 건너뛴다. bench 는 `core/` 아래 커밋 안 된 파일이 있으면 돌지 않으므로, 새 yml 은
bench 를 돌리기 전에 커밋한다.
