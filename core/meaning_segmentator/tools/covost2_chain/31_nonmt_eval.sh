#!/bin/bash
# NMT 를 안 쓰는 비교군 평가 — SASST(구문), Causal Align, 구두점.
# 판정 근거가 각각 spaCy 파서 / SimAlign+참조번역 / 정규식이라 madlad 재작업 대상이 아니다.
# 라벨은 15,530행으로 madlad 라벨과 ID 집합이 같다.
#
# 축이 갈린다:
#   syntax_mc3·mc5·mc10  자기 노브(max_chunk) 점 3개 -> --baselines-native
#   syntax(기본)·causal_align·punct  노브가 없으므로 우리 T 격자로 곡선
#     --conditions 로 좁히면 T 변이(causal_align_T2 등)까지 걸러진다. 안 준다 —
#     mechanical_8(하한 기준선)도 그래야 들어온다.
#     **이 T 곡선은 그 정책의 노브가 아니라 우리가 얹은 축이다.** 논문에 그렇게 적을 것.
#
# bleu_eval 은 이번 실행의 조건만 담아 파일을 덮어쓰므로, madlad 23조건 + COMET 을
# 백업해 두고 끝나면 merge_conditions.py 로 되얹는다.
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
BK=$F/bleu_backup_premadlad_nonmt
L=$F/logs
mkdir -p $BK $L
cp $F/bleu/de.json $F/bleu/ja.json $F/bleu/zh.json $BK/
echo "백업 -> $BK"

B="syntax syntax_mc3 syntax_mc5 syntax_mc10 causal_align punct"
NAT="syntax_mc3 syntax_mc5 syntax_mc10"

for t in de ja zh; do
  echo "===== bleu_eval $t $(date '+%F %T') ====="
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id $RUN --label none_use_manifest --split test \
    --dataset covost2 --manifest-tag full --targets $t \
    --t-grid 2 3 4 6 --src-spaced 1 \
    --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
    --workers 24 --baselines $B --baselines-native $NAT \
    --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
    > $L/bleu_eval_nonmt_$t.log 2>&1
  rc=$?
  echo "  $t exit=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "!! 실패 — 백업 되돌림"; cp $BK/*.json $F/bleu/; exit 1; }
done

echo "===== madlad 조건과 병합 $(date '+%F %T') ====="
$PY core/meaning_segmentator/tools/covost2_chain/merge_conditions.py $BK $F/bleu de ja zh || {
  echo "!! 병합 실패 — 백업 되돌림"; cp $BK/*.json $F/bleu/; exit 1; }
touch $F/baselines/nonmt_bleu.done
echo "===== 완료 $(date '+%F %T') ====="
