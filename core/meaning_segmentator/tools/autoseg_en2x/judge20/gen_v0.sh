#!/bin/bash
# judge20 의 v0 를 만든다 — 이터 0 이라 Writer 호출과 프로파일까지만 돌고 끝난다.
#   tmux new-session -d -s judge20v0 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge20/gen_v0.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge20_v0.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') v0 생성 start" >> $LOG
.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run28 --run-id en2x/en-multi/judge20 \
    --generate-v0 --v0-candidates 1 \
    --min-gap 1 --min-chunk 2 --max-k 99 --iterations 0 \
    --labeled-examples --workers 128 --extra-key-envs OPENAI_API_KEY_2 \
    --provider openai --model gpt-5-mini --budget 5 >> $LOG 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> $LOG
