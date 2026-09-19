#!/bin/bash
# run28 라벨 체인 — API 0, GPU 만.
#   tmux new-session -d -s run28 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/run28/make_run28.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-autoseg/bin/python}
M=core.meaning_segmentator.tools.autoseg_en2x.run28.make_run28
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run28_labels.log
mkdir -p "$(dirname "$LOG")"
{
  echo "== $(date '+%F %T') run28 start"
  $PY -u -m $M --phase build || exit 1
  for S in sel test; do
    echo "== $(date '+%F %T') $S labels"
    $PY -u -m $M --phase labels --split $S || exit 1
    echo "== $(date '+%F %T') $S pseudoref"
    $PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split $S \
        --run-id en2x/en-multi/run28 --mt-cache-from en2x/en-multi/run28 || exit 1
    echo "== $(date '+%F %T') $S merge"
    $PY -u -m $M --phase merge --split $S || exit 1
  done
  echo "== $(date '+%F %T') run28 DONE"
} >> "$LOG" 2>&1
echo "== $(date '+%F %T') exit=$?" >> "$LOG"
