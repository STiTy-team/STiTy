#!/bin/bash
# AlignAtt ja 를 4090 에서 돈다 — de·zh 가 GPU 를 비운 뒤에 시작한다.
# f=2 는 GB10 이 넘긴 5,177/15,530 지점부터 --resume 이 이어가고, f=4·6·8 은 새로 돈다.
# 상한은 끈다 (24_native_madlad.sh 의 TOKEN_CAP 기본값). ja 는 폭주 행이 섞여 상한이
# 벌어 주는 몫이 가장 큰 타깃이지만, 상한을 켜면 AlignAtt 결과가 달라지고 닿은 행을
# 다시 돌려 등가를 맞추면 오히려 느렸다 (ja 300문장 159초 vs 135초).
set -u
cd /home/skkai/STiTy || exit 1
F=core/meaning_segmentator/experiment/artifacts/en2x/covost2/full
while [ ! -f "$F/baselines/madlad_alignatt_zh.done" ]; do sleep 60; done
export PY=.venv/bin/python STAGE=alignatt BATCH=32 POOL=1024 BEAMS=128
TGT=ja bash core/meaning_segmentator/tools/covost2_chain/24_native_madlad.sh
