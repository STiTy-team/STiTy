#!/bin/bash
# 새 오라클(cohesion × (1−contra), run25) 에서 프롬프트를 잰다.
#   기준: v0_minimal_tgt (판단 셋 — 뒤집힘/앞조각/뒷조각). 캐시 적중이라 $0
#   변형: v0_coh        (판단 둘 — 뒤집힘/이어붙인 번역이 원문이 되나)
# dev 215, k=3. 변형만 든다 (~$2.7).
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv/bin/python; S=core/meaning_segmentator/tools/autoseg_en2x/run24/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run25_coh_prompt.log
echo "== $(date '+%F %T') coh 프롬프트 탐침 시작" >> $LOG
$PY -u -m core.meaning_segmentator.tools.autoseg_en2x.rule_probe \
    --run-id en2x/en-multi/run25 --prompt $S/v0_minimal_tgt.txt \
    --variant-prompt $S/v0_coh.txt --split dev --k-samples 3 --tag coh_k3 --save-rows \
    --provider openai --model gpt-5-mini --budget 4 >> $LOG 2>&1
echo "== $(date '+%F %T') rule_probe exit=$?" >> $LOG
echo "== $(date '+%F %T') DONE" >> $LOG
