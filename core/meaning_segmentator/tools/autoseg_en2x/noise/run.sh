#!/bin/bash
# 분절 비결정론을 H_set 단위로 재는 체인 — API 0, GPU 만.
#   tmux new-session -d -s segnoise -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/noise/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/sample_noise.log
mkdir -p "$(dirname "$LOG")"
{
  echo "== $(date '+%F %T') start"
  .venv-autoseg/bin/python -u -m core.meaning_segmentator.tools.autoseg_en2x.noise.sample_noise
  echo "== $(date '+%F %T') exit=$?"
} >> "$LOG" 2>&1
