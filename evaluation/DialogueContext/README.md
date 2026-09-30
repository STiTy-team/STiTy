# DialogueContext — 대화 문맥이 번역 품질과 지연에 주는 영향

번역기에 앞 대화를 **어떤 형태로**(원문 `SRC`, 번역 `TGT`, 둘 다 `SRC_TGT`, 화자 표시까지 `SPK_SRC_TGT`),
**몇 턴이나**(n=1, 3, 5) 넘겨야 하는지 재는 ko→en 실험이다. 문맥 없음(`NONE`)이 기준선이다.
모델은 Qwen3.5-4B, Gemma-3-4B(로컬 4bit), DeepL quality, gpt-6-luna 넷이다.

- 보고서: [report/REPORT.md](report/REPORT.md)
- 설계 계약(스키마·프롬프트·결과 행 필드): [DESIGN.md](DESIGN.md)

## 디렉터리

| 경로 | 내용 |
|---|---|
| `data/` | 대화 5개(`dialogues.jsonl`, 88턴)와 평가 인스턴스 20개(`instances.jsonl`), 훑어보기 표 `preview.md` |
| `scripts/author_dialogues.py` | 데이터를 코드로 적어 둔 것. 돌리면 검사를 통과할 때만 `data/` 를 다시 쓴다 |
| `scripts/run_translations.py`, `scripts/dctx/` | 번역 실행기. 정준 요청 → 프롬프트/DeepL context → 순차 실행·지연 측정 |
| `scripts/score_reference.py` | COMET, XCOMET-XL, chrF++, BLEU |
| `scripts/judge_context.py` | 문맥 판정기(gpt-6-sol). 6개 점수, 오류 라벨, 인스턴스별 check 통과 여부 |
| `scripts/aggregate.py`, `scripts/figures.py` | 합치기, 표 T0~T7, 그래프 |
| `scripts/run_all.sh` | 전체 체인. tmux 로 띄운다 |
| `scripts/gpu_sampler.sh` | 실행 중 GPU 를 같이 쓰는 프로세스를 5초마다 기록 |
| `configs/experiment.yml` | 모델, 조건, 시드(20260925), 단가, 예산, 파이썬 환경 경로 |
| `results/dctx-20260925/` | 원시 결과, 비용 기록, 채점, 집계 표, 그래프 |
| `logs/` | 단계별 로그와 완료 표시(`logs/markers/`) |
| `report/` | 최종 보고서와 작성 메모 |

## 다시 돌리기

저장소 루트에서 실행한다. API 키(`OPENAI_API_KEY`, `DEEPL_API_KEY`, `HF_TOKEN`)는 루트 `.env` 에서 읽는다.

```bash
python evaluation/DialogueContext/scripts/author_dialogues.py        # 데이터 (선택)
tmux new-session -d -s dctx -c "$PWD" "bash evaluation/DialogueContext/scripts/run_all.sh"
tail -f evaluation/DialogueContext/logs/run_all.log
```

`run_all.sh` 는 main 번역 → S1 번역과 판정을 함께 → 참조 채점 → 집계 순으로 돈다. 끝난 단계는
`logs/markers/*.done` 으로 건너뛰고, 단계 안에서도 끝난 작업은 건너뛴다. 새로 돌리려면 `configs/experiment.yml`
의 `run_id` 를 바꾼다. 단계별 명령과 환경 변수(`JUDGE=no`, `SCORING=no`, `S1_PARALLEL=no`, `JUDGE_BUDGET`)는
보고서의 Reproduction 절에 있다.

파이썬 환경 두 개(`venv-tf5`: 번역·판정, `venv-metrics`: COMET·집계)의 경로는 `configs/experiment.yml` 의
`envs` 에 있다. 둘 다 예전 세션의 임시 폴더에 있어 재부팅하면 사라질 수 있다.

`make bench` 는 쓰지 않는다. bench 의 문맥 경로는 모델 자신의 앞 번역만 넘기고 화자 정보를 번역기에 주지
않아서, 정답 앞 번역과 화자 조건을 만들 수 없다.
