#!/bin/bash
# judge41 — 편집 표면을 옮긴 뒤 처음 도는 루프 런. **루프가 무언가를 벌 수 있는지** 본다.
#
# 여기까지의 사정
#   홀드아웃 560문장에서 잰 이득 둘은 **둘 다 사람이 손으로 쓴 것**이다 — combo 의 등급 + 반복 선택
#   (+0.0084)과 원칙의 정도 축(+0.0066). 루프가 낸 편집 서른 몇 개는 전부 ≤0 이다. judge37 이
#   목적함수 탓은 아니라고 했고(max→mean 으로 SNR 0.62배), judge38 이 이유를 짚었다: 에이전트가
#   만질 수 있는 표면이 이득이 있는 자리에 붙어 있지 않았다.
#     [Core Principles] 의 질문은 등급 **배정**을 정하고 그쪽은 이미 작동한다(1위 6.16배).
#     비어 있던 것은 등급 **안** 서열이고(8위 1.27배, 채택 문턱이 40–69 등급 안에 떨어진다),
#     그것을 정하는 것이 같은 줄의 **정도 축**이다.
#     [Order Principles] 의 옛 계약은 그 서열을 **덮어쓰는** 것이어서 이득을 깎았다
#     (같은 편집이 옛 v0 위 −0.0015, 정도 축을 붙인 판 위 −0.0053).
#
#   그래서 골격·계약·역할을 한 이야기로 맞췄다. 이 런이 그것을 처음 태운다.
#
# 역할 넷이 세 일에 갈린다 (이터당 후보 5개)
#   severity    ×order   원칙 하나의 **정도 축**만 다시 쓴다. 질문은 첫 물음표까지 바이트 보존.
#                        이항 발견을 "어느 원칙의 축이 둘을 못 가르는가" 의 증거로 읽는다.
#   fallback    ×order   [Order Principles] 에 **좁은 예외**를 넣는다 — 정도로 읽으면 틀리는 짝 하나.
#   narrow_rule ×check   [Core Principles] 에 새 질문을 넣는다 (등급 배정 쪽).
#   replace     ×둘      원칙 하나를 발견으로 교체한다. 비교 팔이다 — 지금까지 가장 깨끗했던 역할.
#
#   C 를 건드리는 편집은 역할과 무관하게 **질문 + 정도 축 두 부분**이어야 한다(code 가 거부한다).
#   표지는 실제 생성물로 맞췄다 — 정도 축 31/31 통과, 방향만 말하는 절 23개 오탐 0.
#
# v0 는 이미 생성해 감사했다
#   후보 셋 전부 여덟 원칙에 정도 축을 썼고, [Order Principles] 는 좁은 예외였고, 느낌의 정도·줄 중간
#   헤더·환경 의존 상수가 없었고 골격 검사를 통과했다. `--resume` 이 그 파일을 그대로 쓴다.
#
# 볼 것
#   (a) `severity` 후보의 Δ. **이 런의 유일한 질문이다** — 루프가 등급 안 서열을 개선할 수 있는가.
#   (b) `fallback` 이 이제 좁은 예외를 쓰는가, 아니면 다시 일반 선호를 쓰는가.
#   (c) `narrow_rule` 이 정도 축까지 쓰는가. 안 쓰면 코드가 거부하고 재시도가 사유를 준다.
#   (d) 세 이터가 같은 규칙을 되풀이하는가(judge34 는 "자기완결 진술 / 술어 완성 / 주동사구 완성"
#       으로 사실상 같은 것을 세 번 냈다).
#
# 예산 $150 의 근거
#   v0 3후보 train 채점 $9 + 기준선 3벌 $12 + 후보 15개 $45 + 2차(이터당 1개 통과 가정) $18 = $84.
#   채택이 나면 기준선 재측정 $12 과 최종 test 3벌 두 프롬프트 $26 이 붙어 $122 다. 예산 가드는
#   `BudgetExceeded` 로 런을 죽이므로 최악을 덮는 값으로 잡는다 — 상한을 올리는 것이 지출을 올리지는
#   않는다. 앞선 실행(v0 생성만)은 호출 5개로 사실상 0 이었고 `--resume` 이 그것을 차감한다.
#
#   tmux new-session -d -s judge41 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge41/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge41.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge41 \
    --generate-v0 --v0-candidates 3 --resume \
    --candidate-roles severity,replace,fallback,narrow_rule --candidates-cross \
    --candidates-cap 6 --findings-max 2 --full-score-max 6 \
    --pe-verbatim --critic-both-kinds --screen-n 0 \
    --baseline-draws 3 --confirm-draws 3 --final-draws 3 \
    --gate-rule mean --gate-min 0.003 --adopt-rule lo --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 3 \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-150}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge41 exit=$rc" >> "$LOG"
