#!/bin/bash
# minimal_tgt 프롬프트의 test gold — dev overlap 이득이 실제 번역 품질로 바뀌나.
# 1) test 100 채점 (k=3, ~$1.3)  2) madlad 번역 + BLEU  3) COMET  4) run24 v0 대비 부트스트랩
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv/bin/python; S=core/meaning_segmentator/tools/autoseg_en2x/run24/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run24_gold_minimal.log
TAG=min_tgt_k3
echo "== $(date '+%F %T') score test" >> $LOG
$PY -u $S/gold_minimal.py --prompt $S/v0_minimal_tgt.txt --tag $TAG \
    --provider openai --model gpt-5-mini --budget 3 >> $LOG 2>&1 || { echo "SCORE FAILED" >> $LOG; exit 1; }
echo "== $(date '+%F %T') bleu" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id en2x/en-multi/run24_gold_$TAG --label $TAG --split test \
    --dataset fleurs --manifest-tag multi_loop405 --targets de ja zh \
    --translate-engine local --wordtimes interp --no-auto-greedy --no-sentence-bleu \
    --workers 16 >> $LOG 2>&1 || { echo "BLEU FAILED" >> $LOG; exit 1; }
echo "== $(date '+%F %T') comet" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.scoring.comet_eval \
    --run-id en2x/en-multi/run24_gold_$TAG --label $TAG --split test \
    --manifest-tag multi_loop405 --targets de ja zh >> $LOG 2>&1 || { echo "COMET FAILED" >> $LOG; exit 1; }
echo "== $(date '+%F %T') compare" >> $LOG
$PY -u $S/compare_gold.py $TAG >> $LOG 2>&1
echo "== $(date '+%F %T') DONE" >> $LOG
