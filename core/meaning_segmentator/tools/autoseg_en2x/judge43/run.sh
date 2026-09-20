#!/bin/bash
# judge43 — **고친 길이 지시**로 Writer 를 다시 뽑아 잰다.
#
# 무엇이 달라졌나
#   종전 지시문은 "9500자 이하, 주어진 두 블록이 약 3,000" 이었다. 그 숫자는 원칙에 정도 축을
#   요구하기 **전**에 쓴 것이다. 실측으로 고정분이 7,164자다(골격 3,253 + 출력 규약 1,456 +
#   예시 2,455). 9,500 중 2,336자만 남는데 거기에 [Role] 과 원칙 여덟(질문 + 정도 축)과 예외
#   서넛을 넣으라는 것은 **불가능한 요구**였다.
#
#   그래서 judge42 의 후보 여덟이 전부 10,433~11,037자였고(지시를 1,000~1,500자 초과) 그중 셋은
#   **뒤쪽 원칙**(C8·C6·C3)의 정도 축을 빠뜨렸다 — 길이가 차서 끝을 줄인 것이다.
#
#   `writer_budget()` 이 고정분을 세어 상한을 계산한다(지금 11,364 = 7,164 + Writer 몫 4,200).
#   그리고 "정도 축을 먼저 예산에 넣고 질문을 써라 — 여덟 개를 쓰고 나서 마지막을 매길 자리가
#   없다는 걸 발견하지 마라" 를 명시했다.
#
# 무엇을 비교하나
#   judge42 의 여덟 개(옛 지시)와 이 여섯 개(새 지시)를 **같은 train 분할·같은 벌**에서 비교한다.
#   `--from-run` 과 `--draw-tag` 가 같으므로 분절 캐시가 공유되고, 프롬프트가 다르니 값은 새로 뽑힌다.
#
#   볼 것 셋
#     1. 정도 축 누락이 사라지는가 (judge42 는 3/8 이 1~2줄 빠뜨렸다)
#     2. 길이가 상한 안에 들어오는가
#     3. train 점수 분포가 올라가는가. judge42 는 0.5377~0.5542 로 **폭이 컸다** — 후보 3 이
#        0.5542 로 judge34 의 0.5588 에 가까웠다. 그러니 "Writer 가 못 쓴다" 가 아니라 "폭이 크다"
#        였고, 지시를 고쳤으면 그 폭의 아래쪽이 올라와야 한다.
#
#   판정은 **최고값이 아니라 분포**로 한다. 최고값은 후보를 많이 뽑으면 어차피 올라간다
#   (`--v0-candidates` 와 `V0_TIE_BAND` 가 그 일을 한다). 알고 싶은 것은 평균과 아래쪽이다.
#
# 비용: Writer 6회는 약 $0.1, train 채점 6벌 × 약 $3 = $18. 예산 $25 가 기준선 진입에서 끊는다.
#
#   tmux new-session -d -s judge43 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge43/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge43.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge43 \
    --generate-v0 --v0-candidates 6 --resume \
    --candidate-roles severity,replace,fallback,narrow_rule --candidates-cross \
    --candidates-cap 6 --findings-max 2 --full-score-max 6 \
    --pe-verbatim --critic-both-kinds --screen-n 0 \
    --baseline-draws 3 --confirm-draws 3 --final-draws 3 \
    --gate-rule mean --gate-min 0.003 --adopt-rule lo --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 3 \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-25}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge43 exit=$rc" >> "$LOG"
