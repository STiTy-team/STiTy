#!/bin/bash
# x2en judge 루프 (de/ja/zh → 4타깃). 설정은 en2x/en-multi/judge13 과 같다:
#   run03 4분할(train 사례·예시 / test_a 판정 / test_b 확인·최종) + v0 자동 생성(후보 2)
#   --max-k 99 --checkpoint-every 0 --iterations 4
#   --pe-candidates 4 --screen-n 50 --screen-skip --labeled-examples --candidate-roles free,free,examples_only --confirm-dev-b
#   judge13 산출물에서 역산한 값: 판정 짝 수 1851/437 은 max-k 99 에서만 맞는다(10 이면 1651/399);
#   checkpoint.json 이 없어 체크포인트 0; iter_05 가 없어 이터 4; 후보 4 중 후보 2 만 예시 편집, rewrite 없음.
#   --screen-skip 은 judge13 에서 발동 조건(전 후보 Δ≤0)이 한 번도 없어 흔적으로 판별 불가 — judge10 계보를 따라 켠다.
#   tmux new-session -d -s judge-de -c <저장소> "SRC=de bash core/meaning_segmentator/tools/autoseg_x2en/run_judge.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-autoseg/bin/python}
LANG_=${SRC:?SRC=de|ja|zh}
RUN=${RUN:-judge01}
BUDGET=${BUDGET:-60}
FROM=x2en/${LANG_}-multi/run03
RID=x2en/${LANG_}-multi/$RUN
EXTRA=${EXTRA:---pe-candidates 4 --screen-n 50 --screen-skip --labeled-examples --candidate-roles free,free,examples_only --confirm-dev-b}
LOG=core/meaning_segmentator/experiment/artifacts/x2en/logs/${RUN}_${LANG_}.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') $RID start (from $FROM, budget \$$BUDGET, extra: $EXTRA)" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run $FROM --run-id $RID \
    --generate-v0 --v0-candidates 2 \
    --min-gap 1 --min-chunk 2 --max-k ${MAXK:-99} \
    --iterations ${ITER:-4} --n-cases 12 --checkpoint-every ${CKPT:-0} \
    --provider openai --model gpt-5-mini --budget $BUDGET ${RESUME:+--resume} $EXTRA >> $LOG 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> $LOG
