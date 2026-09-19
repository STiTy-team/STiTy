#!/bin/bash
# 배치 크기가 추론 토큰·벽시계에 무엇을 하는지 재는 통제 비교.
#   tmux new-session -d -s probebatch -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/probe_batch/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/probe_batch.log
mkdir -p "$(dirname "$LOG")"
{
  echo "== $(date '+%F %T') start"
  .venv-autoseg/bin/python -u -m core.meaning_segmentator.tools.autoseg_en2x.probe_batch.probe_batch
  echo "== $(date '+%F %T') exit=$?"
} >> "$LOG" 2>&1
