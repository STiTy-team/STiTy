#!/bin/bash
# 라벨 후보 F(참조 COMET)/G(무참조 QE) — dev 분석 → test 오라클 절단 → gold.
# LLM 호출 0. GPU 만 든다. 남의 프로세스는 안 죽이고 VRAM 이 날 때까지 기다린다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; S=core/meaning_segmentator/tools/autoseg_en2x/run24/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run24_pseudoref.log
NEED=9000
wait_gpu() {
  local ok=0
  for i in $(seq 1 720); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$NEED" ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 2 ] && return 0
    sleep 30
  done
  echo "GPU WAIT TIMEOUT" >> $LOG; return 1
}
wait_gpu || exit 1
echo "== $(date '+%F %T') dev 라벨 F/G" >> $LOG
$PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split dev >> $LOG 2>&1 \
  || { echo "DEV FAILED" >> $LOG; exit 1; }
echo "== $(date '+%F %T') test 라벨 F/G + emit" >> $LOG
wait_gpu || exit 1
$PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split test --emit >> $LOG 2>&1 \
  || { echo "TEST FAILED" >> $LOG; exit 1; }
for TAG in F_pseudoref G_qeconcat; do
  echo "== $(date '+%F %T') $TAG bleu" >> $LOG
  wait_gpu || exit 1
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
      --run-id en2x/en-multi/run24_gold_$TAG --label $TAG --split test \
      --dataset fleurs --manifest-tag multi_loop405 --targets de ja zh \
      --translate-engine local --wordtimes interp --no-auto-greedy --no-sentence-bleu \
      --workers 16 >> $LOG 2>&1 || { echo "$TAG BLEU FAILED" >> $LOG; continue; }
  echo "== $(date '+%F %T') $TAG comet" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.scoring.comet_eval \
      --run-id en2x/en-multi/run24_gold_$TAG --label $TAG --split test \
      --manifest-tag multi_loop405 --targets de ja zh >> $LOG 2>&1 || { echo "$TAG COMET FAILED" >> $LOG; continue; }
  $PY -u $S/compare_gold.py $TAG >> $LOG 2>&1
done
echo "== $(date '+%F %T') DONE" >> $LOG
