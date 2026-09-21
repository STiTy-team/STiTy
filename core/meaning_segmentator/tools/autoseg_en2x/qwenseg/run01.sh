#!/bin/bash
# qwenseg01 — ASR 파인튜닝 모델의 텍스트 전용 <SEG> 확률을 judge13 최종표와 같은 자로 잰다.
#   tmux new-session -d -s qwenseg -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/run01.sh"
# LLM 호출 0 이 정상이다. judge13 분절은 캐시 재생이고, 빗나가면 --prompt-budget 안에서만 쓴다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=${RUN:-qwenseg01}
SPLIT=${SPLIT:-test}
MODEL=${MODEL:-models/Qwen3-ASR-1.7B-en-covost2-dailytalk-mix-c200-merged}
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/$RUN.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') $RUN $SPLIT $MODEL start" >> "$LOG"
.venv/bin/python -u -m core.meaning_segmentator.autoseg.gates.qwen_seg_prob \
    --from-run en2x/en-multi/run27 --run-id en2x/en-multi/$RUN --split "$SPLIT" \
    --model "$MODEL" --prompt-run en2x/en-multi/judge13 --prompt-budget 0.5 >> "$LOG" 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> "$LOG"
