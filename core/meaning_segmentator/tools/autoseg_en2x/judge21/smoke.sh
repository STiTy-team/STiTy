#!/bin/bash
# judge21 의 새 기능이 **배선까지 도는지** 24문장으로 확인한다. 값이 아니라 경로를 본다.
#   fallback 역할 / 역할 위반 재시도 / examples_only / 깊이 배수 / 역전 쌍 / --no-near-miss
#   / sep_cases (사례가 train 에서 나오는가) / 부검 깊이 변화
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge21_smoke.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') smoke start" >> $LOG
.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run28_smoke --run-id en2x/en-multi/judge21_smoke \
    --generate-v0 --v0-candidates 1 \
    --candidate-roles fallback,narrow_rule,examples_only,prune --pe-candidates 4 \
    --candidates-cap 4 --findings-max 1 --full-score-max 4 \
    --adopt-rule lo --adopt-strong 0 --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 12 --inversions-max 5 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 1 \
    --workers 32 --extra-key-envs OPENAI_API_KEY_2 \
    --provider openai --model gpt-5-mini --budget 4 >> $LOG 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> $LOG
