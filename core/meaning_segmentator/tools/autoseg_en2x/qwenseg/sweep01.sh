#!/bin/bash
# qwenseg01 스윕 — 스트리밍 임계값 <SEG> 절단의 지연 대비 H_set 곡선 (run27 test, judge13 과 같은 자).
#   tmux new-session -d -s qwensweep -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/sweep01.sh"
# LLM 호출 0 이 정상이다 (judge13 분절은 캐시 재생, 빗나가면 --prompt-budget 안에서만).
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=${RUN:-qwenseg01}
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/${RUN}_sweep.log
echo "== $(date '+%F %T') sweep start" >> "$LOG"
.venv/bin/python -u -m core.meaning_segmentator.autoseg.gates.qwen_seg_sweep \
    --from-run en2x/en-multi/run27 --run-id en2x/en-multi/$RUN \
    --prompt-run en2x/en-multi/judge13 --prompt-budget 0.5 >> "$LOG" 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> "$LOG"
