#!/bin/bash
# judge18 이 끝나면 judge19 를 띄운다.
#
#   tmux new-session -d -s judge-queue -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/queue_judge19.sh"
#
# 대기는 tmux 세션 유무로 본다 — PID 로 기다리면 세션이 죽을 때 의미가 없어진다.
# **judge 런은 하나씩만 돌린다.** judge16·17·18 의 cache/segment.json 은 같은 inode 를 가리키는
# 하드링크라, 두 런이 동시에 돌면 JsonCache 가 파일 전체를 다시 쓰면서 서로의 항목을 지운다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
WAIT_FOR=${WAIT_FOR:-judge18}
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/queue_judge19.log
mkdir -p "$(dirname "$LOG")"

echo "== $(date '+%F %T') 대기 시작 — $WAIT_FOR 세션이 사라지면 judge19 를 띄운다" >> "$LOG"
while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 30; done
echo "== $(date '+%F %T') $WAIT_FOR 종료 확인" >> "$LOG"

# judge19 — 유형 Δ 는 진단·순위로만 쓰고 판정은 test-A 200문장 본채점으로 되돌린다.
# 후보 역할에 prune(원칙 하나 삭제)과 rewrite(갈아끼우기)를 넣는다 — judge17·18 의 후보 33개는
# 전부 규칙을 **추가**했고 전부 음수였다. 빼기 방향은 한 번도 시험하지 않았다.
# v0 는 judge17 것으로 고정해 세 런의 출발점을 같게 둔다.
# 대조군 채점(--type-control)은 끈다 — 순위만 쓰는 데는 필요 없고, judge18 여섯 쌍이 방향 없음으로
# 끝났다. 시간만 이터당 25분 든다. finding 2개(후보 8개)로 줄여 이터를 1.5시간에 맞춘다.
V0PATH="$PWD/core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge17/prompt_v0.txt"
EXTRA="--type-select --type-rank-only --full-score-max 2 --findings-max 2 --type-holdout-max 25 --candidates-cross --candidates-cap 12 --candidate-roles prune,single_small,narrow_rule,rewrite --labeled-examples --growth-per-iter 0.15 --n-cases 60 --case-exclude bin --case-alloc loss --adopt-rule mean --guard-bin 0.01 --max-k 99 --checkpoint-every 3 --workers 128 --extra-key-envs OPENAI_API_KEY_2 --iterations 3"

tmux new-session -d -s judge19 -c "$PWD" \
  "V0=$V0PATH FROM=en2x/en-multi/run27 PY=.venv-autoseg/bin/python RUN=judge19 BUDGET=60 EXTRA='$EXTRA' bash core/meaning_segmentator/tools/autoseg_en2x/run_judge01.sh"
echo "== $(date '+%F %T') judge19 시작 (budget \$60, 이터 3, 본채점 2개/이터)" >> "$LOG"
