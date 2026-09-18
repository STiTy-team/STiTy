#!/bin/bash
# MU n=10 을 4090 에서 ja -> zh 순으로 돈다. n=2 는 GB10 이 맡고 있어 건드리지 않는다.
#
# 24_native_madlad.sh 를 안 쓰는 이유: STAGE=mu 는 n 을 2·10·50 전부 돌아 GB10 과 겹친다.
#
# 4090 24GB 설정. madlad-3b 는 디코더 32층 × 헤드 16 × d_kv 128 이라 KV 캐시가 시퀀스당
# 토큰 하나에 256KB 고, 이게 메모리를 지배한다. 가중치 fp16 6.5GB + 크로스어텐션 KV 가
# 깔린 위에 빔 KV 와 회차 KV 가 얹힌다. BEAMS=512 BATCH=128 은 오늘 낮에 OOM 났다.
#   BEAMS=64  n=10 이면 한 번에 6문장, 빔 KV 1.5GB
#   BATCH=32  회차 KV 0.75GB
#   POOL=1024 후보를 1024문장씩만 만들어 진행분이 일찍 쌓인다
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
LOG=$F/logs/baselines/mu_n10_4090.log
B=".venv/bin/python -u -m core.meaning_segmentator.autoseg.baselines.build --run-id $RUN --dataset covost2 --manifest-tag full --resume --no-token-cap --policy mu_prefix --n-cands 10 --out-name mu_prefix_mad_n10 --batch-size 32 --pool 1024 --max-beams 64"

for t in ja zh; do
  if [ -s "$F/baselines/mu_prefix_mad_n10_${t}_test.json" ]; then
    echo "== $(date '+%F %T') $t skip (최종본 있음)" >> $LOG
    continue
  fi
  echo "== $(date '+%F %T') $t start" >> $LOG
  $B --targets $t >> $F/logs/baselines/mu_prefix_mad_n10_$t.log 2>&1
  rc=$?
  echo "== $(date '+%F %T') $t exit=$rc" >> $LOG
  [ $rc -ne 0 ] && fail=1
done
[ "${fail:-0}" = 0 ] && touch $F/baselines/mu_n10_4090.done
echo "== $(date '+%F %T') ALL DONE" >> $LOG
