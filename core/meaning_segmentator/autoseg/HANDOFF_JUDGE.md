# judge10 을 다른 환경에서 이어 돌리기

판단형 루프(`loop_judge`)의 현재 런 `judge10` 을 다른 기계에서 `--resume` 으로 잇는 절차.
설계와 근거는 [LOOP_JUDGE.md](LOOP_JUDGE.md), 논문용 정리는 [PAPER_JUDGE.md](PAPER_JUDGE.md).

## 지금 상태 (2026-09-15 13:00)

| | 상태 |
|---|---|
| 브랜치 | `autoseg-judge` |
| 소스 런 | `run26` — 3분할: train 200 (사례·실측 예시) / test-A 200 (채택 판정) / test-B 200 (체크포인트·유망 확인·최종 표). 최종 홀드아웃 test 는 없다 |
| judge10 | v0 후보 2개를 train 에서 잼 (0.5952 / 0.5923 → 후보 1 채택, 7,322자). **iter 0 을 test-A 에서 채점하던 중 멈췄다.** `state.json` 은 아직 없다 (iter 1 시작 때 처음 써진다) |
| 지출 | judge10 $6.80 (`iter_00/metrics.json`). judge08 $27.86, judge09 $27.97 은 끝났거나 멈췄다 |
| 캐시 | `run24/cache/` 의 분절 캐시(`segment*.json`)와 번역 캐시(`translate_*.json`)를 저장소에 넣어 뒀다. v0 후보 둘의 train 채점과 test-A 채점 일부가 적중한다 |
| 테스트 | `test_loop_judge`·`test_agents_judge`·`test_hset`·`test_agents_distill` 전부 통과 |

## 준비물

- `.venv-autoseg` (COMET·langcodes 가 여기 있다. 없으면 `.venv`)
- GPU 9 GB 이상 — madlad-3b 와 CometKiwi
- CometKiwi 는 HF 게이트 모델: 라이선스 동의 + `hf auth login`
- `.env` 에 `OPENAI_API_KEY`
- tmux

## 재개

```bash
git fetch && git checkout autoseg-judge && git pull
tmux new-session -d -s judge10 -c <저장소> \
  "FROM=en2x/en-multi/run26 PY=.venv-autoseg/bin/python RUN=judge10 RESUME=1 BUDGET=50 \
   EXTRA='--pe-candidates 3 --screen-n 50 --screen-skip --labeled-examples \
          --candidate-roles free,rewrite,examples_only --iterations 7 --confirm-dev-b' \
   bash core/meaning_segmentator/tools/autoseg_en2x/run_judge01.sh"
tail -f core/meaning_segmentator/experiment/artifacts/en2x/logs/judge10.log
```

`RESUME=1` 이면 `prompt_v0.txt` 를 그대로 쓰고(Writer 를 다시 부르지 않는다) 앞선 지출을
예산에서 뺀다. 첫 줄에 `[data] 3분할 — train 200 (사례·예시) / test-A 200 (판정) / test-B 200`
이 찍혀야 한다. 그 뒤 `[iter 0]` 채점(캐시 적중이면 몇 분, 아니면 40분·$4)과 `[iter 1] 후보 0/1/2`
가 이어진다.

플래그 뜻 (자세한 근거는 LOOP_JUDGE.md):

| 플래그 | 뜻 |
|---|---|
| `--pe-candidates 3 --screen-n 50` | 이터마다 개정 후보 3개를 받아 test-A 에서 돌려 뽑은 50문장으로 선별, 최선만 200 전체로 본채점 |
| `--screen-skip` | 선별 최고 Δ 가 0 이하면 본채점 없이 기각 |
| `--labeled-examples` | train 사례 문장을 실측 라벨 순위로 채운 Input/Output 예시로 PE 에 준다 |
| `--candidate-roles free,rewrite,examples_only` | 후보 0 은 자유 편집, 1 은 Writer 가 원칙·예시를 새로 씀, 2 는 예시만 편집 |
| `--confirm-dev-b` | test-A 에서 평균 > 0.005·하한 > −0.01 로 기각된 개정은 test-B 를 더 재서 합산 하한으로 재판정 |

## 무엇을 보나

- `[iter N] 후보 j 선별 Δ …` 셋 → `후보 j 선택` → `[iter N] Δ H_set … → accept/confirm/reject`.
  유망하면 `유망(…) — dev-B 로 확인` 과 `합산 Δ … → accept/reject`.
- `[iter N] 부검 …` 한 줄이 그 개정이 어디서 무너졌는지 요약한다. 상세는 `iter_NN/regression.json`.
- 채택되면 `best_prompt.txt` 가 갱신되고 다음 이터의 사례는 train 을 새 프롬프트로 다시 재서 뽑는다.
- `[끝]` 뒤 `final_report.md` — test-B 에서 채택본과 v0 를 잰 표.

## 이전 런에서 알아 둘 것

- **judge05~09 의 dev-B·test 값은 무효다.** H_set 채점기의 조각 번역 캐시가 문장 번호로 키를
  잡아, dev-A 뒤에 잰 집합이 앞 문장의 번역을 받았다. 원문 키로 고쳤다(`hset.py` `piece()`).
  dev-A 값과 이터별 Δ 는 유효하다.
- judge09 iter 2 의 채택(+0.0080)은 실측 예시 문장이 판정 집합에 있던 누수와 위 오염이 겹친
  것이라 근거가 없다. 그래서 3분할(run26)로 갈랐다.
- 프롬프트 골격은 다섯 섹션이다([Decision Procedure] 제거). 옛 v0 를 `--prompt` 로 주면 골격
  검사에 걸리므로 새 런은 `--generate-v0` 로 시작한다.

## 멈추고 다시 잇기

```bash
tmux kill-session -t judge10
echo "== $(date '+%F %T') judge10 stopped manually: <사유>" >> core/meaning_segmentator/experiment/artifacts/en2x/logs/judge10.log
# 코드를 고친 뒤 위 재개 명령 그대로
```

이터 경계에서 멈추면 잃는 것이 없다(`state.json` 은 이터 시작 때 저장). 이터 중간에 멈추면 그
이터의 에이전트 호출(센트 단위)과 아직 캐시에 안 들어간 채점만 다시 한다.
