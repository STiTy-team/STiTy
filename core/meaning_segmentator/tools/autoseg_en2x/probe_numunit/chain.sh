#!/bin/bash
# 숫자–단위 감점 규칙 통제 실험 — run22 가 끝난 뒤에 돈다.
#
# 부검이 네 번 같은 진단을 냈다 (run19 iter2, run21 iter1, run22 iter1, 그리고 그 전).
# 기전 이름은 `removed_safe_boundary` 지만 `truncate` 는 점수 상위 k−1 개만 남기고
# k 는 문장 길이와 T 로 정해진다 — 감점은 경계를 없앨 수 없고 자리만 바꾼다.
#
# 그런데 run21 에서 **채택된** 유일한 개정(iter3, dev +0.0368±0.0179)에도 같은 부류의
# 숫자–단위 감점이 들어 있다. 차이는 문구다:
#
#   run21 채택본  "moderate/soft ordering bias ... toward mid-range relative ranks"
#                 + 예외 세 가지 (단위 뒤 구두점 / 자릿수 구분 숫자 / 대문자 단위)
#   run22 기각본  "reduce the score substantially ... place into a lower band (0–40)"
#                 예외 없음
#
# 그래서 **같은 부류를 문구만 바꿔** v0 에 하나씩 넣고 잰다. 가르려는 것:
#
#   완충본 ≈ 0, 일괄본 < 0   -> 방향이 아니라 세기·예외가 문제다
#   둘 다 ≈ 0                -> run21 의 +0.0368 은 감점 규칙이 아니라 같이 들어간
#                               precedence 블록이나 바뀐 예시에서 나왔다
#   둘 다 < 0                -> 감점 규칙을 넣는 것 자체가 해롭다 (맞바꿈 폐기)
#
# 가설 둘 다 사전에 정해 놓았다. 최저를 고르는 게 아니라 각각을 2 se 로 판정한다.
#
# **잴 수 있는 크기.** dev 215문장에서 Δ 의 표준오차가 ±0.019 이므로 |Δ| > 0.038 이면
# 갈린다. 그보다 작은 효과는 이 실험으로 못 본다 — 부검이 주장하는 크기(문장 다수가
# 1.0 에서 0.0 으로)는 이 범위 안이다.
#
# 기준 프롬프트는 run22 의 v0 이고 dev 채점이 캐시에 있으므로 추가 비용은 변형 둘뿐이다.
# 변형당 215문장 / 배치 6 = 36콜 × $0.024 ≈ $0.9, 합쳐 $2 안팎.
set -u

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.

if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi

S="core/meaning_segmentator/tools/autoseg_en2x/probe_numunit"
PREV="core/meaning_segmentator/tools/autoseg_en2x/run22/chain.done"
RUN="en2x/en-multi/run22"
V0="core/meaning_segmentator/experiment/artifacts/$RUN/prompt_v0.txt"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/probe_numunit.log"
mkdir -p "$(dirname "$LOG")"

echo "== $(date '+%F %T') run22 종료 대기 ($PREV)" >> "$LOG"
# **PID 가 아니라 마커 파일로 기다린다** — 세션이 죽으면 PID 는 무의미하다.
waited=0
while [ ! -f "$PREV" ]; do
    sleep 30
    waited=$((waited + 30))
    if [ $waited -ge 14400 ]; then
        echo "== $(date '+%F %T') 4시간 대기 초과, 포기" >> "$LOG"
        touch "$S/chain.failed"
        exit 1
    fi
done
echo "== $(date '+%F %T') probe start (PY=$PY, 대기 ${waited}s)" >> "$LOG"

rc_all=0
for V in hedged blanket; do
    echo "== $(date '+%F %T') --- $V ---" >> "$LOG"
    $PY -u -m core.meaning_segmentator.tools.autoseg_en2x.rule_probe \
        --run-id "$RUN" \
        --prompt "$V0" \
        --rule-file "$S/$V.txt" \
        --split dev --tag "numunit_$V" \
        --provider openai --model gpt-5-mini \
        --budget 3 >> "$LOG" 2>&1
    rc=$?
    echo "== $(date '+%F %T') $V exit=$rc" >> "$LOG"
    [ $rc -eq 0 ] || rc_all=1
done

[ $rc_all -eq 0 ] && touch "$S/probe.done" || touch "$S/probe.failed"
touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
