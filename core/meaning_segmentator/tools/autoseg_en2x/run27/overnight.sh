#!/bin/bash
# 밤샘 — 채택(best_prompt != v0)이 나올 때까지 judge 런을 잇는다. 런당 BUDGET, 최대 MAX_RUNS 런.
#   tmux new-session -d -s overnight -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/run27/overnight.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
FIRST=${FIRST:-13}; MAX_RUNS=${MAX_RUNS:-2}; BUDGET=${BUDGET:-60}
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/overnight.log
EXTRA='--pe-candidates 4 --screen-n 50 --screen-skip --labeled-examples
       --candidate-roles free,single_small,examples_only,single_small --iterations 4 --confirm-dev-b
       --max-k 99 --workers 128 --extra-key-envs OPENAI_API_KEY_2 --checkpoint-every 0
       --case-alloc loss --full-score-max 3'
for ((n=0; n<MAX_RUNS; n++)); do
  RUN=judge$((FIRST + n))
  echo "== $(date '+%F %T') $RUN start (budget $BUDGET)" >> "$LOG"
  FROM=en2x/en-multi/run27 PY=.venv/bin/python RUN=$RUN RESUME=1 BUDGET=$BUDGET EXTRA="$EXTRA" \
    bash core/meaning_segmentator/tools/autoseg_en2x/run_judge01.sh
  D=$A/$RUN
  if [ -f "$D/best_prompt.txt" ] && [ -f "$D/prompt_v0.txt" ] && ! cmp -s "$D/best_prompt.txt" "$D/prompt_v0.txt"; then
    echo "== $(date '+%F %T') $RUN ADOPTED — best_prompt != v0, 멈춘다" >> "$LOG"; exit 0
  fi
  echo "== $(date '+%F %T') $RUN 채택 없음" >> "$LOG"
done
echo "== $(date '+%F %T') MAX_RUNS 도달, 채택 없음" >> "$LOG"
