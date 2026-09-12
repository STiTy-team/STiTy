#!/bin/bash
# run16 — 증류 루프 재실행. run15 에서 갈라낸 원인 셋을 고친 프롬프트/판정으로 다시 돈다.
#
#   1) 채택 기준 achv -> overlap (`--objective overlap`, 이제 기본값)
#   2) SCORE_MEANING 이 측정 절차를 그대로 서술 — 문법 완결성 지시 제거
#   3) 문장 내 순위 + 동점 금지, tie_rate / tie_at_cut_rate 추적
#
# 라벨은 run15 것을 그대로 쓴다 (`--labels-from`) — 같은 분할·같은 정의라 재계산할 이유가
# 없고, 준비 단계 약 15분이 빠진다.
#
# 나머지 하이퍼파라미터는 전부 loop_distill 기본값이고, 그 기본값이 run15/config.json 과
# 같다 (train 30 / train_pool 90 / dev 215 / test 100 / seed 20260806 / iterations 5 /
# patience 3 / v0 3 / revision 3 / adopt_se_mult 1.0 / batch 6 / workers 16 /
# reasoning medium / comet_batch 64). 다른 것은 objective 와 provider 둘이다.
#
# **프로바이더가 run15 의 letsur 이 아니라 openai 다.** 모델 이름(gpt-5-mini)은 같지만
# 엔드포인트가 다르므로 run15 대비 차이에 이 축이 하나 더 끼어 있다. 비용 집계 경로도
# 다르다 — letsur 은 응답에 estimated_cost 를 싣지만 openai 는 안 주므로 gateway 의
# _PRICES 표(gpt-5-mini = 입력 0.25 / 캐시 0.025 / 출력 2.00 USD per 1M)로 계산한다.
# 그 표가 `--budget` 의 유일한 근거다.
set -u

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

# tmux 서버는 기존 셸 환경을 안 물고 있다. 키는 여기서 직접 읽는다.
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.

# 기계마다 venv 이름이 다르다 — mobility 는 .venv-autoseg, skkai 는 .venv.
if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi

S="core/meaning_segmentator/tools/autoseg_en2x/run16"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run16.log"
mkdir -p "$(dirname "$LOG")"

# 크래시 후 재실행에서도 이전 로그를 지우지 않는다 — 지우면 그 실행의 지출이 복원 불가다.
echo "== $(date '+%F %T') run16 start (PY=$PY)" >> "$LOG"

$PY -u -m core.meaning_segmentator.autoseg.loop_distill \
    --pair-id en2x/en-multi --run-id run16 \
    --labels-from en2x/en-multi/run15 \
    --provider openai --model gpt-5-mini \
    --iterations 5 --budget 12 >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') loop_distill exit=$rc" >> "$LOG"
[ $rc -eq 0 ] && touch "$S/loop.done" || touch "$S/loop.failed"

# 비용 집계는 산출물이 사라져도 로그로 복원되게 여기서 한 번 더 찍는다.
echo "== $(date '+%F %T') cost_report" >> "$LOG"
$PY -u -m core.meaning_segmentator.autoseg.infra.cost_report \
    --run-id en2x/en-multi/run16 --budget 12 >> "$LOG" 2>&1

touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
