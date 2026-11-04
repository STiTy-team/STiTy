# TranslatorPrompt — 번역기 시스템 프롬프트 실험

번역기 system prompt 를 두 축으로 비교한다. 모델은 Qwen3.5-4B bf16(로컬)과 gpt-6-luna(추론 끔).

- **2.4 출력 형식**: 조건 F0~F4 마다 형식 위반률과 품질. Gehry TED 강연 398문장 + 스트레스 입력 90개
  (잘린 조각, ASR 잡음, 입력 속 지시 등 형식이 깨지기 쉽게 만든 입력). 문맥 N=0 / 16.
- **2.3 번역 품질**: 사전 정보 없이 쓸 수 있는 문구(S1 역할, S2 말투, S3 상황별 말투 표, S4 일반 예시, S5 조합)가
  품질을 올리는지. TED 398문장 + WMT24 Chat en-ko test 1,982문장. 문맥 없음.
- **2.1 동시통역**: Gehry 문장을 Qwen3-ASR `<SEG>` 디코더로 조각낸 것(`data/talk231_seg_t{20,30}.jsonl`, 실험은 t20)을
  차례로 번역하고 낸 번역은 고치지 않는다. 조건 P0(조각을 문장처럼)~P3(동시통역 지시, 예시, 빈 출력 허용)마다
  모순률(xlmr-anli, 전제는 같은 모델의 2.3 S0 문장 전체 번역), 조각 충실도(CometKiwi), 조기 완성률.
  H1~H3 은 통역사식 부분 보류(뒤 정보가 필요한 원문을 남겼다가 다음 조각과 함께 번역)이고, 조각 번역을 이어 붙인
  최종 번역의 COMET 을 같이 잰다. H2F 는 H2 기록에서 마지막 호출만 문장 끝맺기 문면으로 다시 한다(`stream_finish.py`).

공유 문서: [5차] 번역기 시스템 프롬프트 형식·품질 실험 (Confluence, 번역 콘텍스트 폴더)

## 디렉터리

| 경로 | 내용 |
|---|---|
| `configs/format.yml`, `configs/quality.yml` | 2.4 / 2.3 설정 (조건, 모델, 단가, 예산) |
| `data/stress.jsonl` | 스트레스 입력 90개 (`scripts/build_stress.py`) |
| `data/quality_eval.jsonl`, `data/fewshot.json` | 2.3 평가 문장과 S4 예시 (`scripts/build_quality_data.py`) |
| `data/talk231_seg_t*.jsonl` | 2.1 입력 조각 (`scripts/build_seg.py`) |
| `scripts/tp/format_prompts.py`, `scripts/tp/quality_prompts.py` | 조건별 system / user 문면 |
| `scripts/tp/violations.py` | 형식 위반 판정 규칙 (후처리 전 원시 출력 기준) |
| `scripts/format_run.py`, `scripts/quality_run.py` | 실행기. 요청마다 결과를 붙이고, 다시 돌리면 끝난 것을 건너뛴다 |
| `scripts/format_aggregate.py`, `scripts/quality_aggregate.py` | 집계. 저장된 판정 대신 **지금 규칙으로 다시 판정**한다 |
| `configs/stream.yml`, `scripts/tp/stream_prompts.py` | 2.1 설정과 조건별 문면 |
| `scripts/stream_run.py`, `stream_score.py`, `stream_aggregate.py` | 2.1 실행·채점·집계. 문장이 끝날 때마다 한 줄씩 붙인다 |
| `scripts/stream_finish.py` | 2.1 H2F — H2 결과의 마지막 호출만 다시 |
| `scripts/check_batch_equiv.py` | 로컬 배치 생성이 순차 생성과 같은 출력을 내는지 확인 |
| `scripts/run_*.sh` | tmux 체인 |
| `results/fmt-20260930/`, `results/qual-20260930/` | 원시 결과, 호출별 비용(`api_usage.jsonl`), 채점 캐시, 요약(`*_summary.md`) |

2.4 의 N=16 문맥은 `../LongContextMT/results/lcmt-20260930/<모델>/translations.jsonl` (1.1 의 N=16 체인)에서 읽는다.
2.3 의 로컬 결과는 배치 16개로 생성했다. 배치와 순차는 64문장 중 53개만 같은 출력을 내므로 한 런 안에서 섞지 않는다.

## 환경

`../LongContextMT/README.md` 의 "환경" 절과 같다 — 번역은 transformers 5.x 이상, 채점·집계는 `METRICS_PY`
(transformers 4.57 + unbabel-comet + metricx24). `.env` 에 `OPENAI_API_KEY_TRANS`.
`build_seg.py` 만 transformers 4.57 환경과 `autoseg-judge` 브랜치의 분절 코드가 필요하다 (파일 머리말 참고).

## 다시 돌리기

저장소 루트에서. 환경 변수는 tmux 명령 문자열 안에 넣는다.

```bash
python evaluation/TranslatorPrompt/scripts/build_stress.py
python evaluation/TranslatorPrompt/scripts/build_quality_data.py
M=<채점 환경>/bin/python
# 2.4
tmux new-session -d -s fmt-api   -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_format.sh gpt-6-luna"
tmux new-session -d -s fmt-local -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_format.sh qwen3.5-4b-bf16"
tmux new-session -d -s fmt-post  -c "$PWD" "METRICS_PY=$M bash evaluation/TranslatorPrompt/scripts/run_format_post.sh"
# 2.3 (S0~S4), 조건 추가는 run_quality_extra.sh S5
tmux new-session -d -s qual-api   -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_quality.sh translate gpt-6-luna"
tmux new-session -d -s qual-local -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_quality.sh translate qwen3.5-4b-bf16 --batch 16"
tmux new-session -d -s qual-post  -c "$PWD" "METRICS_PY=$M bash evaluation/TranslatorPrompt/scripts/run_quality.sh post"
# 2.1 (전제로 2.3 S0 결과를 읽으므로 그 뒤에)
tmux new-session -d -s stream-api   -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_stream.sh translate gpt-6-luna"
tmux new-session -d -s stream-local -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_stream.sh translate qwen3.5-4b-bf16"
tmux new-session -d -s stream-post  -c "$PWD" "METRICS_PY=$M bash evaluation/TranslatorPrompt/scripts/run_stream.sh post"
```

## 알아 둘 것

- 위반 판정은 규칙 기반이라 놓치는 것이 있을 수 있다. 스트레스 범주는 6~20개씩이라 한 건에 10%p 넘게 움직인다.
- gpt-6-luna 는 `temperature 0` 이어도 같은 프롬프트에 약간 다른 출력을 낸다 (같은 프롬프트인 S4·S5 한→영의 COMET 차이 0.0005).
- WMT24 Chat 은 CC-BY-NC-4.0 (연구용). 개인정보가 `NAME-F`, `PRS-ORG` 같은 자리표시로 가려져 있어 모든 조건에 "자리표시는 그대로" 규칙을 넣었다.
