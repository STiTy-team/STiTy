#!/bin/bash
# max_new_tokens 상한 등가 대조 2차 — ja 에서 어절수 상위 300문장.
# 1차는 매니페스트 앞 300문장이었다. 폭주 행 비율이 접두사가 길수록 오르므로 긴 쪽을 따로 본다.
# 문장 목록은 prompt_eval/caplong_test.json 에 박아 두고 --label caplong 으로 읽힌다.
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
L=$F/logs/baselines
# 1차 대조가 GPU 를 비울 때까지 기다린다. 마커는 앞서 지워 두었다.
while [ ! -f "$F/baselines/capchk.done" ]; do sleep 30; done
B=".venv/bin/python -u -m core.meaning_segmentator.autoseg.baselines.build --run-id $RUN --dataset covost2 --manifest-tag full --targets ja --label caplong --batch-size 32 --pool 1024 --max-beams 128"

$B --policy mu_prefix --n-cands 10 --out-name capchkL_mu_cap   > $L/capchkL_mu_cap_ja.log 2>&1
$B --policy mu_prefix --n-cands 10 --out-name capchkL_mu_nocap --no-token-cap > $L/capchkL_mu_nocap_ja.log 2>&1
$B --policy alignatt --f 2 --out-name capchkL_aa_cap   > $L/capchkL_aa_cap_ja.log 2>&1
$B --policy alignatt --f 2 --out-name capchkL_aa_nocap --no-token-cap > $L/capchkL_aa_nocap_ja.log 2>&1
touch $F/baselines/capchkL.done
