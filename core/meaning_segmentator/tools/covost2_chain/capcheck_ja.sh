#!/bin/bash
# max_new_tokens 상한 등가 대조 — ja 300문장, 상한 on/off 가 같은 분절을 내는지.
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
L=$F/logs/baselines
B="$PY -u -m core.meaning_segmentator.autoseg.baselines.build --run-id $RUN --dataset covost2 --manifest-tag full --targets ja --limit 300 --batch-size 32 --pool 1024 --max-beams 128"

# 4090 은 BEAMS=512 에서 OOM 났다. 128 로 낮춰 돈다 — 배치 크기는 결과를 안 바꾼다.
$B --policy mu_prefix --n-cands 10 --out-name capchk_mu_cap   > $L/capchk_mu_cap_ja.log 2>&1
echo "mu cap rc=$?"
$B --policy mu_prefix --n-cands 10 --out-name capchk_mu_nocap --no-token-cap > $L/capchk_mu_nocap_ja.log 2>&1
echo "mu nocap rc=$?"
$B --policy alignatt --f 2 --out-name capchk_aa_cap   > $L/capchk_aa_cap_ja.log 2>&1
echo "aa cap rc=$?"
$B --policy alignatt --f 2 --out-name capchk_aa_nocap --no-token-cap > $L/capchk_aa_nocap_ja.log 2>&1
echo "aa nocap rc=$?"
touch $F/baselines/capchk.done
