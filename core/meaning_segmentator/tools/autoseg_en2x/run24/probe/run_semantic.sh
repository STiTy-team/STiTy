#!/bin/bash
# (b) 실험 — 표면형 제약을 풀고 "뒤가 앞을 뒤집나 / 앞·뒤 조각이 혼자 서나" 를 LLM 이 직접 판단.
# 기준 run24 v0 (라벨 B, k=3, 캐시), dev 215. 변형 v0_semantic.txt (바뀐 줄 3).
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv/bin/python; S=core/meaning_segmentator/tools/autoseg_en2x/run24/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run24_semantic.log
echo "== $(date '+%F %T') semantic probe start" >> $LOG
$PY -u -m core.meaning_segmentator.tools.autoseg_en2x.rule_probe \
    --run-id en2x/en-multi/run24 --prompt core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run24/prompt_v0.txt \
    --variant-prompt $S/v0_semantic.txt --split dev --k-samples 3 --tag semantic_k3 --save-rows \
    --provider openai --model gpt-5-mini --budget 4 >> $LOG 2>&1
echo "== $(date '+%F %T') rule_probe exit=$?" >> $LOG
$PY -u $S/component_eval.py semantic_k3 >> $LOG 2>&1
echo "== $(date '+%F %T') DONE" >> $LOG
