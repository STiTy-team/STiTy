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
# 무엇을 기다릴지는 호출하는 쪽이 정한다 — 앞 단계가 run03 체인일 수도, 먼저 띄운 run04
# 세션일 수도 있다. **병렬은 메모리로 안전하지 않다**: de·zh 를 같이 돌리다 zh 가 타깃을
# 넘어가며 12GB 로 커져 de 가 OOM 으로 죽었다(2026-09-20 21:04). 타깃마다 메모리가 커지므로
# 첫 타깃에서 잰 17.9GB 로 판단하면 안 된다. 언어는 순차로 돌린다.
CHAIN=${CHAIN:-$A/logs/${FROM}_labels_chain.log}
WAIT_FOR=${WAIT_FOR:-"ALL DONE (de zh ja)"}

if [ -n "$WAIT_FOR" ]; then
  # **대기 메시지를 감시 대상 로그에 쓰지 않는다.** 처음에 체인 로그에 "대기: 'ALL DONE (zh)'"
  # 를 적고 같은 파일을 grep 했더니 자기 줄에 걸려 즉시 통과했다 — de 가 zh 와 병렬로 떠서
  # 다시 OOM 위험에 들어갔다(2026-09-20 21:06). pgrep 이 자기 셸을 잡는 것과 같은 함정이다.
  # 대기 표시는 따로 두고, 매칭도 **줄 시작에 고정**한다("== ... ALL DONE (zh)" 꼴만 센다).
  echo "== $(date '+%F %T') run04 대기 시작 ($LANGS)" >> "$A/logs/${RUN}_wait.log"
  until grep -qE "^== [0-9-]+ [0-9:]+ ALL DONE \\($(echo "$WAIT_FOR" | sed 's/.*(\(.*\))/\1/')\\)$" \
        "$CHAIN" 2>/dev/null; do sleep 30; done
  echo "== $(date '+%F %T') run04 대기 끝 ($LANGS)" >> "$A/logs/${RUN}_wait.log"
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
  # **언어마다 자기 마커를 남긴다.** 종전에는 루프가 다 끝난 뒤 체인 로그에 한 줄을 쓰고
  # 다음 단계가 그 줄을 grep 했는데, zh 가 정상 종료(로그에 DONE)했는데도 그 줄이 안 남아
  # 뒤 단계가 한 시간을 헛기다렸다(2026-09-20 22:06). 원인은 못 특정했고, 특정할 필요도 없다 —
  # 산출물 옆에 파일로 남기면 공유 로그 한 줄에 기대지 않는다.
  date '+%F %T' >> "$A/${l}-multi/$RUN/labels.done"
done
echo "== $(date '+%F %T') ALL DONE ($LANGS)" >> "$A/logs/${RUN}_labels_chain.log"
