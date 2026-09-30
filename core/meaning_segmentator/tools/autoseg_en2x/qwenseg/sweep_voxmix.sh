#!/bin/bash
# qwenseg_voxmix 스윕 — vox-mix-t91-c320 디코더의 텍스트 전용 <SEG> 스트리밍 절단을
# run27 test 에서 judge13 최종표와 같은 자(H_set)로 잰다.
#   tmux new-session -d -s qwensweep -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/sweep_voxmix.sh"
# LLM 호출 0 이 정상이다 (judge13 분절은 캐시 재생, 빗나가면 --prompt-budget 안에서만).
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=${RUN:-qwenseg_voxmix}
MODEL=${MODEL:-models/Qwen3-ASR-1.7B-en-covost2-vox-mix-t91-c320-merged}
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/${RUN}_sweep.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') sweep start $MODEL" >> "$LOG"
${PY:-.venv-autoseg/bin/python} -u -m core.meaning_segmentator.autoseg.gates.qwen_seg_sweep \
    --from-run en2x/en-multi/run27 --run-id en2x/en-multi/$RUN \
    --model "$MODEL" \
    --prompt-run en2x/en-multi/judge13 --prompt-budget 0.5 >> "$LOG" 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> "$LOG"
