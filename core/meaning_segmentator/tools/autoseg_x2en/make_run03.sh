#!/bin/bash
# x2en run03 라벨 체인 (judge 루프용 4분할 + 오라클 라벨) — tmux 로 띄운다:
#   tmux new-session -d -s x2en-labels -c <저장소> "bash core/meaning_segmentator/tools/autoseg_x2en/make_run03.sh"
# LANGS 로 언어(기본 de zh ja, 순차), SPLITS 로 분할(기본 train test_a test_b)을 고른다.
# 언어마다 끝나면 <run>/labels.done 에 분할 목록을 한 줄 남긴다. 두 언어를 동시에 띄우려면 세션 둘:
#   tmux new-session -d -s x2en-labels-de -c <저장소> "LANGS=de bash .../make_run03.sh"
#   tmux new-session -d -s x2en-labels-zh -c <저장소> "LANGS=zh bash .../make_run03.sh"
#   (환경변수는 명령 문자열 안에 — 서버가 떠 있으면 클라이언트 환경이 세션에 안 넘어간다)
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# **`.venv-autoseg` 를 쓴다.** `.venv` 에는 `langcodes` 가 없어 라벨 단계가 import 에서 죽는다
# (2026-09-20 실측). 라벨링에 필요한 것(langcodes·comet·transformers·sentencepiece)이 다 있는
# 환경은 그쪽이고, judge 루프도 같은 환경으로 돈다.
PY=${PY:-.venv-autoseg/bin/python}
RUN=${RUN:-run03}
LANGS=${LANGS:-de zh ja}
SPLITS=${SPLITS:-train test_a test_b}   # test(최종 홀드아웃)는 나중에 SPLITS=test 로
M=core.meaning_segmentator.tools.autoseg_x2en.make_run03
A=core/meaning_segmentator/experiment/artifacts/x2en
mkdir -p "$A/logs"
for l in $LANGS; do
  LOG=$A/logs/${RUN}_labels_${l}.log
  D=$A/${l}-multi/$RUN
  {
    echo "== $(date '+%F %T') $l $RUN start"
    $PY -u -m $M --lang $l --run $RUN --phase build || { echo "== $(date '+%F %T') $l build FAIL"; exit 1; }
    $PY -u -m $M --lang $l --run $RUN --phase labels --splits $(echo $SPLITS | tr ' ' ,) || { echo "== $(date '+%F %T') $l labels FAIL"; exit 1; }
    for s in $SPLITS; do
      $PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split $s \
          --run-id x2en/${l}-multi/$RUN --mt-cache-from x2en/${l}-multi/$RUN \
          || { echo "== $(date '+%F %T') $l pseudoref $s FAIL"; exit 1; }
    done
    $PY -u -m $M --lang $l --run $RUN --phase merge --splits $(echo $SPLITS | tr ' ' ,) || { echo "== $(date '+%F %T') $l merge FAIL"; exit 1; }
    echo "$(date '+%F %T') $SPLITS" >> "$D/labels.done"
    echo "== $(date '+%F %T') $l $RUN DONE"
  } >> "$LOG" 2>&1 || exit 1
done
echo "== $(date '+%F %T') ALL DONE ($LANGS)" >> "$A/logs/${RUN}_labels_chain.log"
