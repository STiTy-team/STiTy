# 설정 — 이름, 버전, 설명하는 법

기준 문서는 [configs/README.md](../../../../configs/README.md) 다. 여기는 스킬이 자주 쓰는 부분의 요약이고,
둘이 다르면 그 문서가 맞다.

## 이름

| 종류 | 모양 | 예 |
|---|---|---|
| 데이터셋 | `코퍼스_방향[_조건][_부분]` | `fleurs_ko-en`, `fleurs_ko-en_cafe`, `fleurs_en-ko_top50` |
| 파이프라인 | `역할.모델[+역할.모델…][_변형]` | `asr.qwen-seg-en+mt.qwen3.5-4b`, `mock` |

- 소문자, 숫자, `. + -` 만 쓴다. `_` 는 칸을 나눌 때만.
- 언어는 두 글자(`ko`, `en`). 기본값인 칸은 두지 않는다.
- 조건·부분 칸의 말은 configs/README.md 의 "칸에 쓰는 말" 표에서 먼저 찾는다. 새 말을 만들면 그 표에 한 줄 더한다.
- **부품 이름은 레지스트리 이름 그대로 쓴다** (`@transcribers.register("qwen-seg")` 의 `qwen-seg`).
  레지스트리 이름은 `core/components/` 와 `core/pipeline/` 에서 `.register(` 로 찾는다.
- **부품 버전 `:v<N>` 은 이름에서 `:` 를 빼고 `-v<N>` 으로 쓴다.** `qwen-seg:v2` 를 쓰는 영어 파이프라인은
  `asr.qwen-seg-v2-en+mt.qwen3.5-4b`. 파일 안의 `name:` 에는 레지스트리 이름 `qwen-seg:v2` 를 그대로 쓴다.

## 버전 — 새 이름인가, `meta.version` 을 올리는가

**실행이 있는 이름은 뜻이 바뀌면 안 된다.** bench 는 설정 내용(`meta` 제외)과 데이터 manifest 의 해시를 지난 실행과
비교해서 다르면 시작 전에 멈춘다.

| 바뀐 것 | 할 일 |
|---|---|
| 같은 조건을 고쳐 다시 잼 (값 수정, 모델 파일 교체) | 같은 파일에서 `meta.version` 을 올린다. 결과는 `<이름>@v<N>` 으로 따로 쌓인다 |
| 비교하려는 조건 자체가 새것 (소음 종류, 부분 집합, 다른 부품) | 새 이름의 파일을 만든다 |
| 부품 버전을 바꿈 (`qwen-seg` → `qwen-seg:v2`) | 다른 부품이므로 새 이름 (`qwen-seg-v2`) |
| `meta.description`·`meta.tags` 만 | 아무것도 안 올린다. 해시에 안 들어간다 |

- `@v<N>` 은 파일 이름에 쓰지 않는다. `meta.version` 에서 붙는다.
- `limit:` 은 반드시 따로 된 파일에 둔다(`first<N>`, `top<N>`). 같은 파일에 넣었다 빼면 결과가 섞인다.
- 새 파일에는 `meta.description` 을 쓴다 — 이름에 없는 것(소음 세기, 어떤 모델 파일인지)을 적는다.

## 설명하는 법

실행마다 아래를 한국어로 짧게 쓴다. 설정 키 이름은 그대로 두고 뜻을 한 마디 붙인다.

1. **무엇을 재나** — `pipeline.transcription.name`·`model` (음성 인식), `pipeline.translation.name`·`model` (번역),
   `commit` (언제 문장을 확정하나: `seg`·`punct`·`always`), `pipeline.vad` (말 끝 감지, `min_silence_ms`).
   설정에 없는 값은 "기본값" 이라고 쓴다.
2. **무엇을 먹이나** — `dataset.name`, `target` (번역할 언어), `limit`·`pick` (일부만), `augment` (소음·공간), `longform`.
3. **결과 폴더** (이 머신에서 돌릴 때) — 아래 명령으로 돌리지 않고 확인한다. 그 폴더에 `summary.json` 이나 `events.jsonl` 이 있으면
   "다시 돌리면 지난 결과를 덮어씀" 이라고 경고한다.
   ```bash
   uv run --project bench python -m bench --config <파이프라인> --dataset <데이터셋> --print-run-dir
   ```
4. **걸리는 시간** — 오디오를 실시간 속도로 흘리므로 오디오 길이보다 오래 걸린다. 지난 실행에서 벽시계 시간은
   오디오 길이의 약 1.4배였다. 오디오 길이는 `$STITY_DATA_ROOT/<dataset.name>/manifest.jsonl` 의 `duration` 합이다
   (`limit` 이 있으면 그만큼만). `make env` 가 `STITY_DATA_ROOT` 를 보여 준다. 모델 로딩 몇 분을 더한다.
5. **GPU** — 파이프라인의 `gpu_memory_utilization` 과 번역 부품의 `gpu_memory_utilization` (GPU 메모리 중 쓸 몫).
   이 머신에서 돌리면 `nvidia-smi` 로 지금 남은 메모리를, `tmux ls` 로 이미 도는 실행을 확인해 같이 적는다.
   머신의 queue 면 그 머신의 queue 에 쌓인 job 수를 적는다.
6. **비용** — 외부 유료 API 를 부르는 부품이 있으면 적고, `.claude/rules/cost-watch.md` 대로 사용량 기록 수단이
   있는지 확인한다. 없으면 5단계 확인에서 실행을 권하지 않는다. 로컬 모델과 `mock` 은 비용이 없다.
