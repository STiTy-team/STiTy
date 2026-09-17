#!/bin/bash
# 상한+재시도 등가 대조 — 기존 nocap 산출과 같은 분절이 나와야 한다.
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
L=$F/logs/baselines
B=".venv/bin/python -u -m core.meaning_segmentator.autoseg.baselines.build --run-id $RUN --dataset covost2 --manifest-tag full --targets ja --batch-size 32 --pool 1024 --max-beams 128"

# 긴문장 300 (--label caplong) 과 앞 300 (--limit) 둘 다 본다.
$B --label caplong --policy alignatt  --f 2        --out-name capchkR_aa_long > $L/capchkR_aa_long_ja.log 2>&1
$B --label caplong --policy mu_prefix --n-cands 10 --out-name capchkR_mu_long > $L/capchkR_mu_long_ja.log 2>&1
$B --limit 300     --policy alignatt  --f 2        --out-name capchkR_aa_head > $L/capchkR_aa_head_ja.log 2>&1
$B --limit 300     --policy mu_prefix --n-cands 10 --out-name capchkR_mu_head > $L/capchkR_mu_head_ja.log 2>&1
touch $F/baselines/capchkR.done
