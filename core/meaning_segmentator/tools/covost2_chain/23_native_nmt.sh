#!/bin/bash
# NLLB 를 쓰는 두 정책의 네이티브 노브 스윕 — AlignAtt `f`, prefix-match MU `n_cands`.
#   f 는 최근 f 어절 안에 어텐션이 붙어 있으면 방출을 미루는 노브이고,
#   n_cands 는 접두사 판정을 beam top-N 후보 집합으로 완화하는 정도다 (N 이 크면 더 이르게 자른다).
#   둘 다 바꾸면 강제 디코딩 경로가 통째로 달라져 사후 재사용이 안 되므로 값마다 라벨을 새로 만든다.
#
#   **타깃 하나를 맡아 돈다 (TGT).** AlignAtt 은 어절을 하나씩 늘리며 문장당 8번쯤 모델을
#   부르는데 배치가 1 이라 GPU 사용률이 30% 에 그친다 (VRAM 은 1.7GB 뿐). 한 프로세스로
#   타깃 셋을 순차로 돌면 19시간인데, 타깃별로 쪼개 띄우면 노는 틈이 메워져 7~9시간이 된다.
#   진행분은 100건마다 `<라벨>.partial.jsonl` 에 흘려 쓰고 --resume 이 이어받는다.
#
#   STAGE=alignatt|mu|both (기본 both) 로 단계를 고른다 — MU 를 다른 기계에서 돌릴 때 쓴다.
#
#   for t in de ja zh; do
#     tmux new-session -d -s sweep-nmt-$t -c <저장소> \
#       "TGT=$t bash core/meaning_segmentator/tools/covost2_chain/23_native_nmt.sh"
#   done
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-autoseg/bin/python}
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
LOG=$F/logs/baselines/native_nmt.log
mkdir -p "$(dirname "$LOG")"
TGT=${TGT:?TGT=de|ja|zh}
LOG=$F/logs/baselines/native_nmt_$TGT.log
B="$PY -u -m core.meaning_segmentator.autoseg.baselines.build --run-id $RUN --dataset covost2 --manifest-tag full --targets $TGT --resume"

if [ "${STAGE:-both}" != mu ]; then
for f in 4 6 8; do
  echo "== $(date '+%F %T') alignatt f=$f start" >> $LOG
  $B --policy alignatt --f $f --out-name alignatt_f$f >> $F/logs/baselines/alignatt_f${f}_$TGT.log 2>&1
  echo "== $(date '+%F %T') alignatt f=$f exit=$?" >> $LOG
done
touch $F/baselines/native_alignatt_$TGT.done
fi

if [ "${STAGE:-both}" != alignatt ]; then
for n in 50 2; do
  echo "== $(date '+%F %T') mu_prefix n_cands=$n start" >> $LOG
  $B --policy mu_prefix --n-cands $n --out-name mu_prefix_n$n >> $F/logs/baselines/mu_prefix_n${n}_$TGT.log 2>&1
  echo "== $(date '+%F %T') mu_prefix n_cands=$n exit=$?" >> $LOG
done
touch $F/baselines/native_mu_$TGT.done
fi
echo "== $(date '+%F %T') ALL DONE" >> $LOG
