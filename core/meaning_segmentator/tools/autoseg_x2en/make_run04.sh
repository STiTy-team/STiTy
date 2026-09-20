#!/bin/bash
# x2en run04 라벨 체인 — run03 의 905문장 라벨을 옮겨 담고 **새 문장만** 계산한다.
#
#   de  655문장 / zh 655 / ja 475   (전부 다시 돌리면 언어당 2~3시간이 더 든다)
#
# 선행: run03 의 `spare` 라벨링이 끝나야 한다 — run04 build 가 `data/spare.json` 을 읽고
# compose 가 `oracle_labels_spare.json` 을 읽는다. **PID 가 아니라 체인 로그의 마커**로 기다린다.
#
#   tmux new-session -d -s x2en-run04 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_x2en/make_run04.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-autoseg/bin/python}
RUN=${RUN:-run04}
FROM=${FROM:-run03}
LANGS=${LANGS:-de zh ja}
M=core.meaning_segmentator.tools.autoseg_x2en.make_run04
A=core/meaning_segmentator/experiment/artifacts/x2en
CHAIN=$A/logs/${FROM}_labels_chain.log
WAIT_FOR=${WAIT_FOR:-"ALL DONE (de zh ja)"}

if [ -n "$WAIT_FOR" ]; then
  echo "== $(date '+%F %T') run04 대기: '$WAIT_FOR'" >> "$A/logs/${RUN}_labels_chain.log"
  until grep -qF "$WAIT_FOR" "$CHAIN" 2>/dev/null; do sleep 30; done
fi

for l in $LANGS; do
  LOG=$A/logs/${RUN}_labels_${l}.log
  {
    echo "== $(date '+%F %T') $l $RUN start"
    $PY -u -m $M --lang $l --phase build  --from-run $FROM --run $RUN || { echo "build FAIL"; exit 1; }
    $PY -u -m $M --lang $l --phase labels --from-run $FROM --run $RUN || { echo "labels FAIL"; exit 1; }
    $PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split new \
        --run-id x2en/${l}-multi/$RUN --mt-cache-from x2en/${l}-multi/$RUN || { echo "pseudoref FAIL"; exit 1; }
    $PY -u -m $M --lang $l --phase merge   --from-run $FROM --run $RUN || { echo "merge FAIL"; exit 1; }
    $PY -u -m $M --lang $l --phase compose --from-run $FROM --run $RUN || { echo "compose FAIL"; exit 1; }
    echo "== $(date '+%F %T') $l $RUN DONE"
  } >> "$LOG" 2>&1 || exit 1
done
echo "== $(date '+%F %T') ALL DONE ($LANGS)" >> "$A/logs/${RUN}_labels_chain.log"
