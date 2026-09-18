#!/bin/bash
# MU n=50 을 n=2 뒤에 이어 돈다 (ja -> zh). 진행분은 5행뿐이라 사실상 처음부터다.
#
# n=50 은 빔이 배치 안에서 곱해져 한 호출에 도는 문장 수가 BEAMS/50 이다.
#   BEAMS=128 -> 2문장/호출, 빔 KV 3GB. 회차 KV 0.75GB + 가중치 6.5GB 로 합계 11GB 대.
#   BEAMS=64  -> 1문장/호출이 되어 배치 이득이 사라지므로 128 이 하한이다.
# POOL 은 256 으로 줄인다 — 후보 생성이 느려 pool 이 크면 진행분이 늦게 쌓인다.
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
LOG=$F/logs/baselines/mu_n50_4090.log
while [ ! -f "$F/baselines/mu_n2_4090.done" ]; do sleep 60; done
B=".venv/bin/python -u -m core.meaning_segmentator.autoseg.baselines.build --run-id $RUN --dataset covost2 --manifest-tag full --resume --no-token-cap --policy mu_prefix --n-cands 50 --out-name mu_prefix_mad_n50 --batch-size 32 --pool 256 --max-beams 128"

for t in ja zh; do
  if [ -s "$F/baselines/mu_prefix_mad_n50_${t}_test.json" ]; then
    echo "== $(date '+%F %T') $t skip (최종본 있음)" >> $LOG
    continue
  fi
  echo "== $(date '+%F %T') $t start" >> $LOG
  $B --targets $t >> $F/logs/baselines/mu_prefix_mad_n50_$t.log 2>&1
  rc=$?
  echo "== $(date '+%F %T') $t exit=$rc" >> $LOG
  [ $rc -ne 0 ] && fail=1
done
[ "${fail:-0}" = 0 ] && touch $F/baselines/mu_n50_4090.done
echo "== $(date '+%F %T') ALL DONE" >> $LOG
