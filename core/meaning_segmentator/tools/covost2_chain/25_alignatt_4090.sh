#!/bin/bash
# AlignAtt 스윕을 4090 에서 이어 돈다 (MU 는 GB10 으로 갔다).
# de f=4 는 13,165/15,530, zh f=4 는 7,200/15,530 지점의 진행분을 --resume 이 물고 이어간다.
# 그 진행분은 토큰 상한 적용 전에 난 것이라 상한 없이 이어야 한 파일 안에서 체제가 안 섞인다
# (24_native_madlad.sh 의 TOKEN_CAP 기본값이 off 다).
# 4090 은 BEAMS=512 에서 OOM 났다 — 배치 계열은 결과를 안 바꾸므로 낮춰 잡는다.
set -u
cd /home/skkai/STiTy || exit 1
S=core/meaning_segmentator/tools/covost2_chain/24_native_madlad.sh
export PY=.venv/bin/python STAGE=alignatt BATCH=32 POOL=1024 BEAMS=128
TGT=de bash $S
TGT=zh bash $S
