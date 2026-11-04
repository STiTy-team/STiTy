# LongContextMT — 앞 번역을 몇 문장까지 문맥으로 줄 것인가

en→ko 번역기에 **자기가 앞 문장들에 낸 번역 N개**를 문맥으로 줄 때 품질(1.1)과 지연(1.2)이
N 에 따라 어떻게 바뀌는지 잰다. 원문·정답 번역·화자 정보는 문맥에 넣지 않는다.

- 데이터: IWSLT17 en-ko train 의 TED 강연 한 편 — Frank Gehry, "My days as a young rebel"
  (talkid 231, 398문장, 256문장 이상 29편 중 가장 김). 화자 1명의 독백이다.
- 모델: Qwen3.5-4B **bf16** (로컬, 텍스트 전용 `Qwen3_5ForCausalLM`), gpt-6-luna (`reasoning_effort: none`)
- N: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256. N 마다 강연을 처음부터 차례로 번역해 체인(번역 로그)을 따로 만든다.
- 채점: 257번째 문장부터(142문장) — 모든 N 에서 문맥이 꽉 찬 문장만 N 끼리 짝지어 비교한다.
  고정 지표만 쓴다: COMET-DA, XCOMET-XL, CometKiwi-DA, chrF++, spBLEU(flores200). LLM 판정자는 쓰지 않는다.

## 디렉터리

| 경로 | 내용 |
|---|---|
| `configs/experiment.yml` | 모델, N 격자, 채점 시작 위치, 시드, 단가, 예산, 1.2 스윕 설정 |
| `data/talk231.jsonl` | 강연 원문·정답 (`scripts/build_data.py` 가 HF 의 IWSLT17 zip 에서 만든다) |
| `scripts/lcmt/prompt.py` | 두 모델이 같이 받는 system / user 문면 |
| `scripts/lcmt/backends.py` | 로컬 bf16 로더와 gpt-6-luna 호출 (DialogueContext 의 `dctx` 를 재사용) |
| `scripts/run_translate.py` | 1.1 번역. 문장마다 N 조건 순서를 섞어 돌린다 |
| `scripts/latency_sweep.py` | 1.2. 마지막 20문장을 고정하고 문맥 길이만 바꿔 3회 반복 |
| `scripts/score.py` | 고정 지표 채점 (COMET·XCOMET·CometKiwi·MetricX·chrF++·spBLEU) |
| `scripts/aggregate.py` | `summary.json`, `summary.md` |
| `scripts/run_translate.sh`, `run_post.sh` | tmux 체인. 순서는 `logs/markers/*.done` 으로 잡는다 |
| `scripts/build_ref_sol.py`, `run_ref_sol.sh` | 다시 채점용 정답. gpt-6-sol 이 원문 한 줄에 정답 한 줄이 정확히 대응하게 다시 번역한다 → `data/talk231_ref_sol.jsonl` (idx 240~397) |
| `scripts/rescore.py`, `aggregate_rescore.py`, `run_rescore_gpu.sh` | 정답 두 벌(IWSLT, sol) × 단위 세 가지(문장, 4문장 묶음, 강연 전체)로 1.1 번역을 다시 채점 → `results/<run_id>/rescore/` |
| `results/<run_id>/` | 모델별 `translations.jsonl`·`latency_sweep.jsonl`, `api_usage.jsonl`(호출마다 비용), 채점 캐시, 요약 |
| `logs/gpu_samples.csv` | 로컬 실행 중 GPU 를 같이 쓴 프로세스. 집계가 그 시각의 로컬 지연을 뺀다 |

## 환경

파이썬 환경이 둘이다. 번역과 채점이 요구하는 transformers 버전이 달라서다.

| 용도 | 필요한 것 | 체인 스크립트에서 |
|---|---|---|
| 번역 (`run_translate.py`, `latency_sweep.py`) | transformers 5.x 이상 (Qwen3.5 의 `qwen3_5` 구조), torch, httpx, PyYAML | `PATH` 의 `python` |
| 채점·집계 (`score.py`, `aggregate.py`) | transformers 4.57, `unbabel-comet` 2.2.x, sacrebleu, `metricx24` | `METRICS_PY` (기본 `python`) |

`metricx24` 는 pip 로 설치되지 않는다. 저장소를 받아 채점 환경의 site-packages 에 경로 파일을 넣는다.

```bash
git clone --depth 1 https://github.com/google-research/metricx.git ~/.cache/metricx-src
echo ~/.cache/metricx-src > "$($METRICS_PY -c 'import site; print(site.getsitepackages()[0])')/metricx-src.pth"
```

XCOMET-XL·CometKiwi 는 HF 게이트 모델이라 라이선스 동의와 `hf auth login` 이 먼저다.
`.env` 에 `OPENAI_API_KEY_TRANS` 를 넣는다. `logs/` 는 git 에 올라가지 않으므로 `gpu_samples.csv` 도 없다 —
다른 환경에서 다시 집계하면 GPU 를 같이 쓴 구간을 빼지 못한다. 올라간 `summary.json` 은 뺀 뒤의 값이다.

## 다시 돌리기

저장소 루트에서. `OPENAI_API_KEY_TRANS` 는 루트 `.env` 에서 읽는다.

```bash
python evaluation/LongContextMT/scripts/build_data.py
tmux new-session -d -s lcmt-api   -c "$PWD" "bash evaluation/LongContextMT/scripts/run_translate.sh gpt-6-luna"
tmux new-session -d -s lcmt-local -c "$PWD" "bash evaluation/LongContextMT/scripts/run_translate.sh qwen3.5-4b-bf16"
tmux new-session -d -s lcmt-post  -c "$PWD" "METRICS_PY=<채점 환경>/bin/python bash evaluation/LongContextMT/scripts/run_post.sh >> evaluation/LongContextMT/logs/post.log 2>&1"
```

새로 돌리려면 `configs/experiment.yml` 의 `run_id` 를 바꾸고 `logs/markers/` 를 비운다.
`VAR=x tmux new-session …` 처럼 앞에 붙인 환경 변수는 tmux 서버에 넘어가지 않는다. `METRICS_PY` 는 명령 문자열 안에 넣는다.

## 다시 채점 (정답·채점 단위 바꾸기)

IWSLT 정답은 자막 번역이라 의역·누락·이웃 줄로 밀린 내용이 있어, 문장 단위로 채점하면 맞는 번역도 감점된다.
그래서 정답을 하나 더 만들고(sol) 채점 단위도 넓혀 1.1 번역을 다시 잰다. 번역은 다시 돌리지 않는다.

- **묶음**: idx 256 부터 4문장씩 35개(끝 2문장 버림). Doc-COMET 은 앞 문맥을 512 토큰 안에서 최대로 채우고(14~24문장)
  점수는 묶음에서만 낸다. 번역 쪽 문맥도 정답 앞 문장을 써서 모든 N 조건의 문맥이 같다. MetricX 는 묶음만 넣는다.
  4 로 정한 근거: 묶음이 최대 약 210 토큰이라 문맥 자리가 넉넉하고, 묶음 35개로 부트스트랩 구간을 낼 수 있다.
  8 이면 17개, 16 이면 COMET·MetricX 입력 상한을 넘는다.
- **강연 전체**: 142문장을 문자열 하나로 이어 chrF++·spBLEU 만 계산한다.
- sol 정답은 gpt-6-luna 와 같은 회사 모델이라 luna 쪽으로 기울 수 있다. 모델끼리 비교보다 N 에 따른 변화를 보는 데 쓴다.

```bash
python evaluation/LongContextMT/scripts/build_ref_sol.py            # 정답 (API, 약 $0.2)
$METRICS_PY -u evaluation/LongContextMT/scripts/rescore.py --stage cpu   # 묶음 정의, chrF++·spBLEU
$METRICS_PY -u evaluation/LongContextMT/scripts/rescore.py --stage gpu   # COMET·Doc-COMET·MetricX·XCOMET (16GB 이상)
$METRICS_PY evaluation/LongContextMT/scripts/aggregate_rescore.py
```

GPU 를 다른 작업과 나눠 쓰는 머신이면 `run_rescore_gpu.sh` 가 GPU 가 비고 3분 뒤에도 비어 있을 때 GPU 단계를 시작한다.

## 알아 둘 것

- 로컬 Qwen3.5 는 선형 어텐션 층의 빠른 커널(`flash-linear-attention`, `causal-conv1d`)이 없어
  torch 기본 구현으로 돈다. 품질에는 영향이 없고 지연 절대값에는 있다.
- 매 요청을 처음부터 계산한다(KV 캐시 재사용 없음). 문맥이 한 문장씩 늘어나는 실서비스에서
  캐시를 재사용하면 지연 곡선이 달라진다.
- 강연 한 편·화자 1명·주제 하나라 결론은 이 강연에 한정된다. IWSLT train 은 공개 데이터라
  두 모델이 학습 때 봤을 수 있다.
