#!/bin/bash
# 비교군 네이티브 노브 스윕 — 내부 NMT 를 평가 번역기와 같은 madlad 로 통일해서 다시 만든다.
#
# 왜 다시 만드나. AlignAtt 의 어텐션과 MU 의 접두사 일치는 둘 다 "이 모델이 지금 무엇을
# 아는가" 를 묻는다. 그 모델이 실제로 번역을 내놓는 모델과 다르면 엉뚱한 모델에 대해
# 판정하게 된다. 원논문들은 자기 시스템 하나로 판정과 출력을 같이 하므로, 그 결합을
# 맞추려면 평가 번역기(madlad)와 같아야 한다. 기존 라벨(alignatt f=2, mu n=10)은 NLLB
# 산이라 새 점과 한 곡선에 못 섞는다 — f=2, n=10 도 목록에 있는 이유다.
#
# SASST·Causal Align·구두점은 NMT 를 안 써서 대상이 아니다 (각각 spaCy 파서 / SimAlign +
# 참조 번역 / 정규식).
#
#   for t in de ja zh; do
#     tmux new-session -d -s madlad-$t -c <저장소> \
#       "TGT=$t bash core/meaning_segmentator/tools/covost2_chain/24_native_madlad.sh"
#   done
#
# 프로세스당 VRAM 6.5GB (NLLB 의 1.7GB 대비) 라 셋이면 19.5GB 다. judge 루프가 돌고 있으면
# 자리가 없다 — 시작 전에 `nvidia-smi` 로 20GB 이상 비어 있는지 확인할 것.
# STAGE=alignatt|mu|both (기본 both).
#
# BATCH/POOL 은 문장 단위 배치의 크기다. 한 문장 안에서는 직전 회차가 정한 강제 접두사
# 때문에 순차지만 문장끼리는 독립이라, 여러 문장의 같은 회차를 한 배치로 묶는다. 배치 1
# 디코드는 스텝마다 디코더 가중치를 통째로 읽으므로 대역폭이 좁은 기계에서 특히 느리다
# (GB10 실측: 단건 4.33초/문장 -> 배치 32 에서 1초 남짓).
# **POOL 은 BATCH 의 몇 배여야 한다.** 묶음은 강제 접두사 길이가 같은 것끼리만 만들어지고
# 한 회차에 그 길이가 20~40가지로 갈리므로, 동시 진행 문장이 적으면 배치가 안 찬다.
# PY 로 인터프리터를 갈아끼울 수 있다 (예: conda 환경의 python).
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-autoseg/bin/python}
TGT=${TGT:?TGT=de|ja|zh}
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
LOG=$F/logs/baselines/madlad_$TGT.log
mkdir -p "$(dirname "$LOG")"
# --nmt-model 을 안 주면 nmt.MODEL 기본값(madlad)이다. 이름에 mad 를 박아 NLLB 산출과 가른다.
BATCH=${BATCH:-128}
POOL=${POOL:-4096}
BEAMS=${BEAMS:-512}
# 토큰 상한은 기본으로 끈다. 상한을 켜면 AlignAtt 결과가 달라지고(폭주 생성을 자른 위치가
# 그대로 forced 커밋 길이가 되어 이후 회차가 전부 갈린다), 상한에 닿은 행을 다시 돌려
# 등가를 맞추면 ja 300문장 실측에서 상한 없는 쪽보다 오히려 느렸다 (159초 vs 135초).
TOKEN_CAP=${TOKEN_CAP:-off}
[ "$TOKEN_CAP" = off ] && CAPFLAG=--no-token-cap || CAPFLAG=
B="$PY -u -m core.meaning_segmentator.autoseg.baselines.build --run-id $RUN --dataset covost2 --manifest-tag full --targets $TGT --resume --batch-size $BATCH --pool $POOL --max-beams $BEAMS $CAPFLAG"

# 이미 나온 최종본은 건너뛴다. 최종본이 생기면 진행분(.partial.jsonl)은 지워지므로
# --resume 이 물 게 없어, 건너뛰지 않으면 끝난 config 를 처음부터 다시 돈다.
done_already() {
  [ -s "$F/baselines/$1_${TGT}_test.json" ]
}

if [ "${STAGE:-both}" != mu ]; then
for f in 2 4 6 8; do
  if done_already alignatt_mad_f$f; then
    echo "== $(date '+%F %T') alignatt f=$f skip (최종본 있음)" >> $LOG
    continue
  fi
  echo "== $(date '+%F %T') alignatt f=$f start" >> $LOG
  $B --policy alignatt --f $f --out-name alignatt_mad_f$f \
     >> $F/logs/baselines/alignatt_mad_f${f}_$TGT.log 2>&1
  rc=$?
  echo "== $(date '+%F %T') alignatt f=$f exit=$rc" >> $LOG
  [ $rc -ne 0 ] && fail=1
done
[ "${fail:-0}" = 0 ] && touch $F/baselines/madlad_alignatt_$TGT.done
fi

if [ "${STAGE:-both}" != alignatt ]; then
# n 이 작을수록 빔이 적어 싸다. 싼 것부터 돌려야 중간에 멈춰도 산출이 더 많이 남는다.
for n in 2 10 50; do
  if done_already mu_prefix_mad_n$n; then
    echo "== $(date '+%F %T') mu_prefix n_cands=$n skip (최종본 있음)" >> $LOG
    continue
  fi
  echo "== $(date '+%F %T') mu_prefix n_cands=$n start" >> $LOG
  $B --policy mu_prefix --n-cands $n --out-name mu_prefix_mad_n$n \
     >> $F/logs/baselines/mu_prefix_mad_n${n}_$TGT.log 2>&1
  rc=$?
  echo "== $(date '+%F %T') mu_prefix n_cands=$n exit=$rc" >> $LOG
  [ $rc -ne 0 ] && fail=1
done
[ "${fail:-0}" = 0 ] && touch $F/baselines/madlad_mu_$TGT.done
fi
echo "== $(date '+%F %T') ALL DONE" >> $LOG
