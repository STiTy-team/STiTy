#!/bin/bash
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=.
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run23_src_nli.log"
echo "== $(date '+%F %T') src_nli start" >> "$LOG"
.venv/bin/python -u core/meaning_segmentator/tools/autoseg_en2x/run23/probe/src_nli.py >> "$LOG" 2>&1
echo "== $(date '+%F %T') src_nli exit=$? DONE" >> "$LOG"
