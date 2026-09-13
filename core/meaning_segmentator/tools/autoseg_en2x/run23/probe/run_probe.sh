#!/bin/bash
# 문안 교란 측정 — dev 215, k=3, 기준 v0(k=3, 캐시 적중).
#   rules   : iter2 관문 통과 규칙 2개를 PE 없이 그대로 [Scoring Rules] 끝에 붙임 → 규칙의 몫
#   nullswap: 내용 변화 0 — 저점수 항목 두 줄 순서만 바꿈 → 글이 바뀌기만 해도 얼마나 흔들리나
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv/bin/python
P="core/meaning_segmentator/tools/autoseg_en2x/run23/probe"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run23_probe.log"
echo "== $(date '+%F %T') probe start" >> "$LOG"
$PY -u -m core.meaning_segmentator.tools.autoseg_en2x.rule_probe \
    --run-id en2x/en-multi/run23 --prompt core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run23/prompt_v0.txt \
    --rule-file "$P/rules_iter2.txt" --split dev --k-samples 3 --tag rules_k3 \
    --provider openai --model gpt-5-mini --budget 4 >> "$LOG" 2>&1
echo "== $(date '+%F %T') rules exit=$?" >> "$LOG"
$PY -u -m core.meaning_segmentator.tools.autoseg_en2x.rule_probe \
    --run-id en2x/en-multi/run23 --prompt core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run23/prompt_v0.txt \
    --variant-prompt "$P/v0_null_swap.txt" --split dev --k-samples 3 --tag nullswap_k3 \
    --provider openai --model gpt-5-mini --budget 4 >> "$LOG" 2>&1
echo "== $(date '+%F %T') nullswap exit=$?" >> "$LOG"
touch "$P/probe.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
