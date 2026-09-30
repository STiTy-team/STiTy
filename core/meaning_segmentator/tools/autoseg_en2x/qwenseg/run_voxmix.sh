#!/bin/bash
# qwenseg_voxmix — vox-mix-t91-c320 디코더의 텍스트 전용 <SEG> 점수를 judge13 최종표와 같은 자로 잰다.
# sweep_voxmix.sh 가 여기서 나오는 qwen_scores_test_<모델>.json 을 읽으므로 이쪽이 먼저다.
#   tmux new-session -d -s qwenprob -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/run_voxmix.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=${RUN:-qwenseg_voxmix}
SPLIT=${SPLIT:-test}
MODEL=${MODEL:-models/Qwen3-ASR-1.7B-en-covost2-vox-mix-t91-c320-merged}
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/$RUN.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') $RUN $SPLIT $MODEL start" >> "$LOG"
${PY:-.venv-autoseg/bin/python} -u -m core.meaning_segmentator.autoseg.gates.qwen_seg_prob \
    --from-run en2x/en-multi/run27 --run-id en2x/en-multi/$RUN --split "$SPLIT" \
    --model "$MODEL" --prompt-run en2x/en-multi/judge13 --prompt-budget 0.5 >> "$LOG" 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> "$LOG"
