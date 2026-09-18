#!/bin/bash
# MU n=2 를 n=10 뒤에 이어 돈다 (ja -> zh). 진행분은 GB10 이 넘긴 것을 --resume 이 문다.
# 설정은 27_mu_n10_4090.sh 와 같다. n=2 는 빔이 적어 BEAMS=64 면 한 번에 32문장이다.
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
LOG=$F/logs/baselines/mu_n2_4090.log
# n=10 이 GPU 를 비울 때까지 기다린다. PID 가 아니라 마커로 잡는다.
while [ ! -f "$F/baselines/mu_n10_4090.done" ]; do sleep 60; done
B=".venv/bin/python -u -m core.meaning_segmentator.autoseg.baselines.build --run-id $RUN --dataset covost2 --manifest-tag full --resume --no-token-cap --policy mu_prefix --n-cands 2 --out-name mu_prefix_mad_n2 --batch-size 32 --pool 1024 --max-beams 64"

for t in ja zh; do
  if [ -s "$F/baselines/mu_prefix_mad_n2_${t}_test.json" ]; then
    echo "== $(date '+%F %T') $t skip (최종본 있음)" >> $LOG
    continue
  fi
  echo "== $(date '+%F %T') $t start" >> $LOG
  $B --targets $t >> $F/logs/baselines/mu_prefix_mad_n2_$t.log 2>&1
  rc=$?
  echo "== $(date '+%F %T') $t exit=$rc" >> $LOG
  [ $rc -ne 0 ] && fail=1
done
[ "${fail:-0}" = 0 ] && touch $F/baselines/mu_n2_4090.done
echo "== $(date '+%F %T') ALL DONE" >> $LOG
