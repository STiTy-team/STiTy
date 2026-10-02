# 실행 — 스모크 테스트, tmux, 지켜보기

## 스모크 테스트

본 실행 전에 앞의 5개 항목만 돌려 모델 로딩·데이터 경로·설정 오류를 미리 잡는다. 몇 분이면 끝난다.

- 데이터셋 설정은 `<데이터셋>_first5`. 부분 칸이 이미 있으면(`_top50`) 그 칸을 `first5` 로 바꾼다
  (`fleurs_en-ko_top50` → `fleurs_en-ko_first5`). 조건 칸은 그대로 둔다 (`fleurs_ko-en_cafe_first5`).
- 파일이 없으면 원래 데이터셋 설정을 복사하고 `dataset.limit: 5` 를 넣고 `pick` 은 뺀다. `meta.description` 에
  "first 5 items, smoke test" 를 덧붙이고 `meta.tags` 에 `smoke` 를 더한다. 만들었다고 사용자에게 알린다.
- 파이프라인은 답에 따라 본 실행과 같은 것, 또는 `mock` (모델 없이 bench 자체만 확인).
- 스모크 테스트는 이 세션에서 바로 돌려도 된다: `make bench CONFIG=<p> DATASET=<d>_first5`.

## tmux 로 띄우기

몇 분 넘게 도는 실행은 반드시 tmux 로 띄운다 — 에이전트의 백그라운드 실행은 세션이 끝나면 같이 죽는다
(`.claude/rules/long-jobs-in-tmux.md`). 스크립트·로그·표시 파일은 모두 `logs/bench/` 에 둔다 (git 이 무시하는 곳).

`<tag>` 는 `<YYYYmmdd-HHMM>-<데이터셋>__<파이프라인>` 으로 짓는다. 여러 번 실행이면 `<YYYYmmdd-HHMM>-batch`.

`logs/bench/<tag>.sh`:

```bash
#!/usr/bin/env bash
set -o pipefail
cd <저장소 절대 경로>
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONUNBUFFERED=1
tag=logs/bench/<tag>
rm -f "$tag.done" "$tag.failed"
if make bench CONFIG=<파이프라인> DATASET=<데이터셋>; then touch "$tag.done"; else touch "$tag.failed"; fi
```

- 여러 번 실행이면 `make bench` 줄 대신 `make bench-batch LIST=<목록 파일>` ([batch.md](batch.md)).
- `.env` 는 tmux 가 기존 환경을 물고 있어서 스크립트 안에서 직접 읽는다.
- `PYTHONUNBUFFERED=1` 이 없으면 출력이 버퍼에 갇혀 로그가 비어 보인다.
- 끝났는지는 PID 가 아니라 `.done`·`.failed` 표시 파일로 안다.

띄우기 (tmux 세션 이름에는 `.` `:` 를 쓸 수 없으니 `bench-<YYYYmmdd-HHMM>` 으로):

```bash
mkdir -p logs/bench
tmux new-session -d -s bench-<YYYYmmdd-HHMM> -c <저장소 절대 경로> "bash logs/bench/<tag>.sh > logs/bench/<tag>.log 2>&1"
```

## 지켜보기

**"에이전트가 지켜봄"** 이면 `logs/bench/<tag>.done` 이나 `.failed` 가 생길 때까지 로그를 따라간다.
`Monitor` 로 로그를 따라가고, 표시 파일이 생기면 멈춘다.

| 로그에서 | 뜻 | 할 일 |
|---|---|---|
| `[ITEM] 12/270 en_1660 wer=0.083 segments=2` | 진행. 몇 번째 항목인지, 그 항목의 WER, 번역된 문장 수 | 10% 정도마다 진행률과 남은 시간 추정을 알린다 |
| `segments=0` 이 이어짐 | 번역이 안 나옴 | 알린다. 이유는 `events.jsonl` 의 `[DROP]` 로그 줄에 있다 |
| `item_error` · `Traceback` · `ConfigError` · `DataError` | 오류 | 바로 알린다 |
| `[INTERRUPTED]` | 중간에 끊김. 그때까지만 채점됨 | 알린다 |
| bench 가 설정이 바뀌었다며 멈춤 | 이름의 뜻이 바뀜 | [configs.md](configs.md) 의 버전 표대로 고칠 방법을 제안한다 |

`make bench` 는 bench 가 끝난 뒤 COMET 채점을 한 번 더 돈다. COMET 이 끝나야 `.done` 이 생긴다.

**"넘김"** 이면 아래를 알려 주고 끝낸다.

```bash
tmux attach -t bench-<YYYYmmdd-HHMM>      # 붙기 (떼기: Ctrl-b d)
tail -f logs/bench/<tag>.log              # 로그
ls logs/bench/<tag>.done logs/bench/<tag>.failed   # 끝났는지
```

## 결과 보고

`bench/runs/<데이터셋>/<파이프라인>/summary.json` 에서:

- `status` (`ok`·`degraded`·`failed`)와 `failure`
- `metrics` 의 `wer`, `cer`, `bleu`, `comet`(있으면), `laal_ms` (번역이 얼마나 늦게 나오나), `avg_fsl_sec`
- `counts` 의 `items`, `errored`, `empty_transcription_output`, `realtime_factor` (1 보다 작으면 실시간보다 빠름)
- 같은 데이터셋에 다른 실행이 있으면 같은 지표를 나란히 보여 준다.
- 마지막에 `make replay RUN=<데이터셋>/<파이프라인>` 을 알려 준다.
