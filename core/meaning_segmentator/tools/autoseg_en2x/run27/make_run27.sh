#!/bin/bash
# run27 라벨 체인 — tmux 로 띄운다: tmux new-session -d -s run27 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/run27/make_run27.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv/bin/python}
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run27_labels.log
mkdir -p "$(dirname "$LOG")"
{
  echo "== $(date '+%F %T') run27 start"
  $PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run27.make_run27 --phase build || exit 1
  $PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run27.make_run27 --phase labels || exit 1
  $PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split test \
      --run-id en2x/en-multi/run27 --mt-cache-from en2x/en-multi/run27 || exit 1
  $PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run27.make_run27 --phase merge || exit 1
  echo "== $(date '+%F %T') run27 DONE"
} >> "$LOG" 2>&1
echo "== $(date '+%F %T') exit=$? " >> "$LOG"
