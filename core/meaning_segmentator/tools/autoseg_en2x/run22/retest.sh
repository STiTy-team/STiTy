#!/bin/bash
# run22 v0 를 dev 에서 한 번 더 채점 — 같은 프롬프트 test-retest 잡음 바닥.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv/bin/python
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run22_retest.log"
echo "== $(date '+%F %T') retest v0a start" >> "$LOG"
$PY -u -m core.meaning_segmentator.tools.autoseg_en2x.retest \
    --run-id en2x/en-multi/run22 --tag v0a \
    --provider openai --model gpt-5-mini --budget 2 >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') retest exit=$rc" >> "$LOG"
[ $rc -eq 0 ] && touch "core/meaning_segmentator/tools/autoseg_en2x/run22/retest.done"
