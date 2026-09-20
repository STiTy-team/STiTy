#!/bin/bash
# judge40 — 설계한 것이 **실측 최고(combo)** 를 넘는가.
#
# 왜 기준선이 combo 인가
#   지금까지 홀드아웃 560문장에서 가장 높은 프롬프트가 combo(0.5630)다. 그 뒤의 변경들은 각자
#   자기 기준선 대비로는 양수였지만 절대 수준은 combo 를 못 넘었다:
#     combo 0.5630 → 판단 칸 둘로 쪼갬(skel2) 0.5591 → 그 골격 위 새 v0 0.5518
#     → 정도 절(graded_clean) 0.5584
#   **정도 절의 +0.0066 이 칸을 쪼개며 잃은 것을 되찾은 것인지, 그 이상인지** 가려야 한다.
#   기준선을 combo 로 잡으면 그 질문에 바로 답한다.
#
# 무엇을 재나
#
#   graded2  정도 절을 붙이고 C7 의 빠진 가벼운 쪽까지 채운 판. 다만 **골격의 O 계약이 아직 옛것**
#            이고("overrides that default") O1·O3 이 새 지시문이 금지한 일반 선호다.
#   design   설계 그대로 손으로 조립한 판. 골격을 커밋된 `scoring_rules()` 로 갈아 끼웠고(문자 그대로
#            같다) 금지된 일반 선호 두 줄을 뺐다. 남은 O 는 구체적 사례 한 줄이다.
#              Core 가 주 척도   — 질문이 등급을 정하고 같은 줄의 정도가 등급 안 서열을 정한다
#              Order 는 예외뿐  — 정도로 읽으면 틀리는 짝을 하나씩 집는다
#              combo 의 기제    — 다섯 등급 + 반복 선택은 그대로다
#
#   combo 자체에 정도 절을 붙이는 판은 **만들지 않았다.** combo 의 골격에는 등급 안 서열 기준이
#   아예 없다("choose the one that is the best cut of them" — 무엇이 best 인지 말하지 않는다).
#   거기에 정도 절만 붙이면 골격이 그것을 읽으라고 말하지 않으므로 쓰이지 않는다. 잘못된 시험이다.
#
# 읽는 법
#   design > 0          설계가 실측 최고를 넘었다. 채택하고 `--generate-v0` 로 넘어간다.
#   design − graded2    계약과 일반 선호 정리가 얼마인가. 양수면 O 칸이 이득을 깎고 있었다는
#                       judge38 의 관찰(같은 편집 v0 위 −0.0015 → graded 위 −0.0053)이 굳는다.
#   둘 다 0 근처        정도 절은 칸을 쪼갠 손해의 보상이었을 뿐이고, 계보를 combo 로 되돌려
#                       거기서 다시 쌓아야 한다.
#
#   구간 모양도 본다. 정도 절의 이득은 **자리를 여러 개 고르는 구간**에서 나와야 한다 — dev 에서
#   ≤3 +0.0143, 홀드아웃에서 ≤5 +0.0094 / ≤99 −0.0007 이었다. ≤99(k=1.2)에서 벌면 딴 것이다.
#
# 한 벌 sd 가 test 560 에서 0.0049 라 3벌 대 3벌 Δ 의 sd 가 0.0040, 하한 > 0 문턱이 약 +0.0078 이다.
# 비용: test 560 한 벌 약 $4.4 × 3프롬프트 × 3벌 = $40. 예산 $60.
#
#   tmux new-session -d -s judge40 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge40/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge40.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge40 \
    --score-only --final-split test --final-draws 3 --draw-tag judge40t \
    --prompt "$A/judge27/prompt_v0_combo.txt" \
    --score-baseline "$A/judge27/prompt_v0_combo.txt" \
    --score-prompts "$A/judge34/prompt_v0_graded2.txt,$A/judge34/prompt_v0_design.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 9 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-60}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge40 exit=$rc" >> "$LOG"
