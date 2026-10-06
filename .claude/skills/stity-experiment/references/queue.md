# 머신의 queue 에 넣기

GPU 머신마다 queue 가 하나다. 그 머신의 worker 가 자기 시간표 안에서 자기 queue 의 job 을 `make bench` 로 돌린다. 결과는 S3 에 올라가고
manager 의 Queue 페이지(`make bench-manager` → http://localhost:9140/queue/<머신>)에서 보인다. 구조는
[bench/README.md](../../../../bench/README.md) "여럿이 함께 쓰기".

## 넣기 전에

- **job 은 설정 내용을 직접 들고 간다.** 4단계에서 만든 `configs/` 의 YAML 을 그대로 보낸다 — S3 에 올리기 전이어도
  된다. 코드(`core/`·`bench/` 등)를 고쳐야 하는 실험이면 그 코드는 origin 의 branch 에 있어야 한다.
  이 스킬은 git 을 쓰지 않으므로, 그럴 때는 사용자에게 push 하고 branch 이름을 알려 달라고 하고 기다린다.
- branch 는 `/api/branches` 의 `branches[].name` 중에서 고른다. job 은 시도할 때마다 그 branch 의 최신 commit 으로 돈다.
- 넣은 뒤 끝났는지·왜 실패했는지는 `/api/machines` 의 `machines[].history` 에 머신마다 최근 100개까지 남는다.
- 설정 이름이 이미 다른 내용으로 쓰였으면 넣을 때 막힌다. 그때는 새 이름을 주거나 `meta.version` 을 올린다.

## 머신 고르기

job 은 머신 하나에 들어간다. `/api/machines` 의 `machines[]` 에서 `host`, `connected`(5분 안에 worker 소식이 있었나),
`notes`(팀이 남긴 그 머신의 설명), `jobs`(이미 쌓인 job) 를 보고, `AskUserQuestion` 으로 어느 머신에 넣을지 묻는다.
연결되어 있고 queue 가 짧은 머신을 첫 선택지로 둔다. 설명에 "쓰지 말 것" 같은 말이 있으면 선택지 설명에 그대로 옮긴다.

## 넣기

사람은 Queue 페이지의 그 머신 탭에서 "Add run" 으로 넣는다. 에이전트는 같은 검사를 거치는 API 를 부른다 —
`make bench-manager` 가 떠 있어야 한다(:9130 이 API 서버):

```bash
python3 - <<'PY' | curl -s -XPOST localhost:9130/api/machines/<머신>/jobs -d @-
import json, pathlib
print(json.dumps({
    "branch": "<branch>",
    "pipeline": {"name": "<파이프라인>", "yaml": pathlib.Path("configs/pipelines/<파이프라인>.yml").read_text()},
    "dataset": {"name": "<데이터셋>", "yaml": pathlib.Path("configs/datasets/<데이터셋>.yml").read_text()},
}))
PY
```

떠 있지 않으면 bench 환경에서 직접 넣는다 (`.env` 의 S3 변수를 읽는다). 이 길은 이름 검사를 거치지 않으니 API 를 먼저 쓴다:

```bash
uv run --project bench python -c "
from core.utils import env; env.load()
from bench.settings import BenchSettings; from bench.machines.machine import Machine
queue = Machine(BenchSettings.load().bucket(), '<머신>').queue
from pathlib import Path
print(queue.submit(branch='<branch>',
                   pipeline='<파이프라인>', pipeline_yaml=Path('configs/pipelines/<파이프라인>.yml').read_text(),
                   dataset='<데이터셋>', dataset_yaml=Path('configs/datasets/<데이터셋>.yml').read_text()).id)"
```

여러 조합이면 조합마다 하나씩 넣는다. queue 는 넣은 순서(FIFO)로 돈다.

## 보고

job id, 넣은 머신, 그 머신이 연결되어 있는지, 그리고 그 머신 탭 주소 `http://localhost:9140/queue/<머신>` 을
알린다. 끝나면 Runs 페이지(`http://localhost:9140/runs`)의 "리플레이" 나 그 탭 기록의 "리플레이 열기" 로 결과를 받아 연다.
