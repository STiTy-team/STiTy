#!/bin/bash
# judge19 를 이터 1 이 닫히는 순간 멈추고, rewrite 재료를 넣어 --resume 으로 잇는다.
#
#   tmux new-session -d -s judge19-swap -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/resume_judge19_rewrite.sh"
#
# 왜: judge19 는 v0 을 --prompt 로 받아서 rewrite 역할이 free 로 떨어졌다(재료가 없다고 판단). 그
# 분기를 고쳤지만 돌고 있는 프로세스는 이미 모듈을 올려서 반영이 안 된다. 이터 경계에서 멈추면
# 손실이 없다 — state.json 이 채택본·이력·누적 비용을 들고 있고, 분절 캐시도 그대로 쓴다.
# 멈추는 지점은 이터 2 의 첫 줄(구간별 손실 몫)이다. 그 뒤 Critic 이 돌기 전이라 돈이 안 든다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
A=core/meaning_segmentator/experiment/artifacts/en2x
LOG=$A/logs/judge19.log
SWAP=$A/logs/judge19_swap.log

echo "== $(date '+%F %T') 이터 2 진입을 기다린다 (이터 1 이 닫히면 state.json 이 쓰인다)" >> "$SWAP"
while ! grep -aq "\[iter 2\] 구간별 손실 몫" "$LOG"; do
  tmux has-session -t judge19 2>/dev/null || { echo "== $(date '+%F %T') judge19 이 이미 끝났다 — 그대로 재개한다" >> "$SWAP"; break; }
  sleep 20
done
tmux kill-session -t judge19 2>/dev/null
sleep 3
echo "== $(date '+%F %T') judge19 중단 — rewrite 재료를 넣고 --resume" >> "$SWAP"

# v0 을 쓸 때와 **같은 재료**가 되도록 judge17 의 language_profile 을 복사한다. 없으면 Profiler 를
# 한 번 더 부르게 되고($0.2) 프로파일이 v0 때와 달라진다.
cp -n $A/en-multi/judge17/language_profile.json $A/en-multi/judge19/language_profile.json

cat >> "$LOG" <<'NOTE'

== rewrite 재료를 넣어 이어 돌린다 (이터 2 부터)
==   이터 1 은 rewrite 없이 돌았다 — v0 을 --prompt 로 받으면 writer_material() 을 호출하지 않아
==   rewrite 후보가 free 로 대체됐다. 분기를 고쳤고, judge17 의 language_profile 을 복사해
==   v0 을 쓸 때와 같은 재료를 준다. 덤으로 Writer 가 Critic 요약 대신 **실패 사례 8개**(현재 절단
==   vs 오라클 절단)를 직접 본다. 같은 런 안에서 이터마다 후보 구성이 다르므로 판독 시 주의.
NOTE

V0PATH="$PWD/$A/en-multi/judge17/prompt_v0.txt"
EXTRA="--type-select --type-rank-only --full-score-max 2 --findings-max 2 --type-holdout-max 25 --candidates-cross --candidates-cap 12 --candidate-roles rewrite,prune,narrow_rule,single_small --labeled-examples --growth-per-iter 0.15 --n-cases 60 --case-exclude bin --case-alloc loss --adopt-rule mean --guard-bin 0.01 --max-k 99 --checkpoint-every 3 --workers 128 --extra-key-envs OPENAI_API_KEY_2 --iterations 3"
tmux new-session -d -s judge19 -c "$PWD" \
  "RESUME=1 V0=$V0PATH FROM=en2x/en-multi/run27 PY=.venv-autoseg/bin/python RUN=judge19 BUDGET=60 EXTRA='$EXTRA' bash core/meaning_segmentator/tools/autoseg_en2x/run_judge01.sh"
echo "== $(date '+%F %T') judge19 재개 — 역할 rewrite,prune,narrow_rule,single_small" >> "$SWAP"
