#!/bin/bash
# judge02 — en2x judge44 의 구성을 de/ja/zh 소스에 옮긴다.
#
# judge44 와 같게 두는 것
#   --candidates-cross 로 역할과 Critic 발견을 곱해 후보를 만든다. --pe-verbatim,
#   --critic-both-kinds, --screen-n 0, --case-exclude bin, --case-alloc loss,
#   --labeled-examples, --growth-per-iter 0.15, --inversions-max 15,
#   --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1, --workers 720 --score-workers 8.
#
#   --candidates-cap 6 은 짝 넷보다 크다. cap 은 `pairs[:cap]` 이라 기준 없이 앞에서 자르는
#   비용 천장이고, Critic 이 order 발견 둘을 내면 짝이 여섯이 되면서 맨 뒤의 fallback 이 조용히
#   죽는다. 천장을 짝 수보다 높게 둬서 자르는 일이 아예 없게 한다. 역할을 줄일 때는 cap 이 아니라
#   --candidate-roles 에서 뺀다 — 의도가 플래그에 적히고 로그에도 남는다.
#
# judge44 와 다르게 두는 것
#   v0     judge44 는 --prompt 로 영어 v0 를 읽고 --draw-tag 로 judge42 의 기준선 3벌을 물려받았다.
#          둘 다 영어 전용이다. 기존 x2en/*/judge01/prompt_v0.txt 는 [Order Principles] 칸이 없고
#          [Core Principles] 가 산문 문단인 **옛 골격**이라 이 역할들이 얹히지 않는다. 그래서
#          --generate-v0 로 언어마다 새로 쓴다 — 지금 코드는 Writer 출력에 scoring_rules 와
#          output_rules 와 예시를 덮어쓰고 check_skeleton 으로 칸을 검사하므로 judge44 골격이 나온다.
#   역할    narrow_rule 을 뺐다. judge44 에서 네 이터 모두 음수였고(−0.0169 / −0.0094 / −0.0208 /
#          −0.0153, 평균 −0.016) 그 역할이 붙는 check 발견은 측정 일곱이 전부 음수였다. 남은 셋이
#          이터당 후보 넷을 만든다: severity×order, replace×order, replace×check, fallback×order.
#          --score-workers 8 이라 후보 여섯도 한 웨이브로 도므로 벽시계는 거의 안 줄고 비용만 준다
#          (언어당 −$12).
#   벌 수   baseline 1 / confirm 1 / final 3. confirm 이 1 이면 loop_judge 의 2단 관문이 통째로
#          꺼지고(multi = confirm_draws > 1) 채택은 decide(평균 > 0 + 퇴행 가드) 한 벌 판정으로
#          간다. --gate-rule 과 --gate-min 은 그 갈래에서 읽히지 않으므로 아예 주지 않는다.
#          최종만 3벌로 두는 이유는, 이터 판정을 한 벌에 맡긴 대가를 런 끝 홀드아웃 비교에서
#          받아내야 하는데 그 비교까지 한 벌이면 결론이 ±0.011 잡음에 묻히기 때문이다.
#   이터    2. judge44 는 5이터를 돌고 홀드아웃 기여가 iter 1·2 에 +0.0123 으로 쏠렸다
#          (3스텝 +0.0123 / 4스텝 +0.0179 / 5스텝 +0.0187). 새 언어에서는 앞 두 스텝만 본다.
#   사례 수  train 의 20% 로 맞춘다 (아래 NCASES). judge44 가 train 500 에 100 을 썼다.
#
# 분할은 run04 가 정한다. dev.json 과 test.json 이 있고 test_a.json 이 없으므로 scheme_sel
# 갈래로 가서 dev 전량이 채택 판정, train 이 사례·예시, test 가 최종 홀드아웃이 된다. dev-B 가
# 없는 구성이라 체크포인트와 유망확인은 코드가 알아서 끈다. --dev-a 는 이 갈래에서 안 읽힌다.
#
# 비용 — 채점 한 벌이 dev/train 500문장 × 타깃 4 에 $3.0, test 560 에 $3.4 다(judge44 2차
# 구간에서 역산). 언어당 v0 생성·선별 $6.5 + 기준선 1벌 $3.0 + 이터 2 × (train 1벌 + 후보 4벌 +
# 에이전트 $1.3) $30.2 + 최종 6벌 $20.4 = 약 $60. ja 는 분할이 10% 작아 $55 쯤이다. 예산
# 가드는 런을 죽이므로 재시도와 크래시를 덮는 값으로 잡는다. 끊기면 --resume 이 이어받고
# 이미 쓴 돈을 차감한다.
#
#   tmux new-session -d -s judge02-de -c <저장소> \
#       "SRC=de bash core/meaning_segmentator/tools/autoseg_x2en/judge02/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
L=${SRC:?SRC=de|ja|zh}
RUN=${RUN:-judge02}
LOG=core/meaning_segmentator/experiment/artifacts/x2en/logs/${RUN}_${L}.log
mkdir -p "$(dirname "$LOG")"

# 사례 수를 train 크기에 비례시킨다. 고정값으로 두면 분할이 작은 ja(train 450)만 같은 사례 수를
# 더 좁은 풀에서 긁어 세 언어를 같은 자로 못 읽는다. de·zh 는 100, ja 는 90 이 된다.
NCASES=${NCASES:-$(.venv-autoseg/bin/python -c "
import json
p='core/meaning_segmentator/experiment/artifacts/x2en/${L}-multi/run04/data/train.json'
print(round(len(json.load(open(p))) * 0.2))
")}

echo "== $(date '+%F %T') x2en/${L}-multi/$RUN 시작 (from run04, 사례 $NCASES, 예산 \$${BUDGET:-100})" >> "$LOG"
.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run "x2en/${L}-multi/run04" --run-id "x2en/${L}-multi/$RUN" \
    --generate-v0 --v0-candidates 2 --resume \
    --candidate-roles severity,replace,fallback --candidates-cross \
    --candidates-cap 6 --findings-max 2 --full-score-max 6 \
    --pe-verbatim --critic-both-kinds --screen-n 0 \
    --baseline-draws 1 --confirm-draws 1 --final-draws 3 \
    --adopt-rule mean --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases "$NCASES" --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations "${ITER:-2}" \
    --seg-reasoning-effort "${SEG_EFFORT:-medium}" \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-100}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') $RUN $L exit=$rc" >> "$LOG"
exit $rc
