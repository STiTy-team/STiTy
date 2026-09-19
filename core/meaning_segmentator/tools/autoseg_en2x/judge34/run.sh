#!/bin/bash
# judge34 — 루프가 **점수를 만드는 절차**를 고칠 수 있게 된 첫 런.
#
# 왜 이 런인가
#
#   지금까지 잰 것을 한자리에 놓으면 방향이 하나로 모인다. 판단 문장을 더하거나 바꾼 편집은
#   스물세 번 측정됐고 가장 좋은 것이 0 이다 — judge27·31 에서는 −0.006~−0.011 로 믿을 만하게
#   해로웠고, 판단 칸을 둘로 가른 뒤(judge33b iter 1)에야 겨우 0 이 됐다(+0.0011 / +0.0002 /
#   +0.0001, 셋 다 신뢰구간이 0 을 감싼다). 반면 `[Scoring Rules]` 의 절차를 선언형("결과가
#   순위여야 한다")에서 절차형(다섯 등급으로 가른 뒤 등급 안에서 하나씩 뽑는다)으로 바꾼 것은
#   dev 500 에서 +0.0068, 홀드아웃 560문장에서 **+0.0084** 였고 이득이 구간의 절단 수 순서를
#   따랐다(≤3 +0.0087 / ≤5 +0.0129 / ≤99 −0.0000).
#
#   그런데 루프는 그 절을 건드릴 수 없었다. `[Scoring Rules]` 는 `FROZEN` 이었고 `UNIT_TAGS` 에도
#   없어서 어떤 편집도 닿지 못했다. **이득이 있는 자리를 빼고 없는 자리만 뒤지고 있었다는 뜻이다.**
#
# 이번에 처음 켜는 것
#
#   procedure 역할    `[Scoring Rules]` 의 절차 줄 하나를 다시 쓴다(`replace`, 순증감 300자).
#                     발견과 짝짓지 않으므로 이터당 하나다 — 고치는 것이 개별 사례가 아니라
#                     순위를 만드는 절차라서 Critic 의 단항·이항 발견에 담기지 않는다.
#                     못 고치는 것: 위의 정의 세 줄(cohesion·contra·target). 정의가 바뀌면
#                     프롬프트가 측정과 다른 말을 하고, 그때 오르는 점수는 판단이 나아진 것이
#                     아니다. `- ` 로 시작하지 않는 것으로 가린다(코드가 주입하는 절이라 모양이
#                     고정이다). 줄을 늘리거나 지우는 것도 막는다 — 줄 수가 변하면 S 번호가 밀려
#                     다음 이터의 S1~S3 이 정의가 아니게 된다.
#
#   --gate-min 0.003  1차 문턱을 부호에서 값으로 올린다. `> 0` 은 **효과가 없는 후보의 절반을
#                     통과시킨다** — 잡음이 양수로 떨어진 쪽이 전부 올라간다. judge33b iter 1 이
#                     그것이다: 후보 셋이 +0.0011 / +0.0002 / +0.0001 로 3/3 통과해 2차에 $15 를
#                     썼는데, 채택 문턱은 3벌 대 3벌에서 +0.0062 이라 닿을 길이 없었다. 2차가
#                     채택할 수 있는 하한의 절반쯤을 요구하면 그 돈을 쓰지 않는다.
#
#   --findings-max 2  후보가 이터당 여섯에서 다섯으로 준다(fallback×order, single_small×check,
#                     replace×둘, procedure). 부담의 주된 몫은 벌 수가 아니라 후보 수다 —
#                     후보 본채점이 총액의 69% 이고 기준선 3벌은 14% 였다.
#
#   --baseline-draws 2  1차 문턱을 거의 안 깎는다(Δ 의 sd 0.0045 → 0.0047, +7%). 후보가 1벌이라
#                     그쪽 잡음이 지배하기 때문이다. 2차는 3벌을 유지한다 — 거기서 2벌로 내리면
#                     채택 문턱이 +0.0062 → +0.0076 으로 22% 올라 통과가 사실상 막힌다.
#
#   v0 재생성          후보 둘의 H_set 차이가 한 벌 sd(0.004) 안이면 점수로 고르지 않고 **1차 포맷
#                     통과율**로 고른다(`V0_TIE_BAND`). judge33b 는 0.5589 대 0.5588 로 갈렸다 —
#                     sd 의 1/40 이라 동전 던지기였다. 그 자리에서는 결정론적인 자가 낫고, 포맷
#                     통과율이 그 자를 한다(재시도를 덜 부르고 출력 규약을 스스로 지킨다).
#                     **재정렬이 Δ 를 편향시키지는 않는다** — `realign_tags` 는 옮긴 자리가 입력
#                     마커 자리와 하나라도 다르면 포기하므로 성공한 재정렬의 절단 자리와 점수는
#                     원래 출력과 같고, 고치는 것은 주변 글자뿐이다.
#
# 볼 것
#   (a) `procedure` 후보의 Δ 가 판단 편집 후보들보다 큰가. 이 런의 유일한 질문이다.
#   (b) 1차 통과가 이터당 몇 개인가. `--gate-min` 이 듣는지는 그것으로 본다.
#   (c) v0 채택이 포맷 통과율로 갈렸는가(로그에 동점대와 1차 값이 찍힌다).
#   (d) `procedure` 가 절차의 어느 줄을 고르는가 — S7(등급과 반복 선택)이 가장 길고 가장 최근에
#       바뀐 줄이다.
#
# 예산 $140 의 근거 — judge33b 실측 단가로 다시 잡은 값이다
#   dev 500 한 벌이 $3.9 (judge33b iter 0 이 3벌에 $11.80), 후보 하나 본채점이 $3, 2차 한 벌이 $3,
#   test 560 한 벌이 약 $4.4 다. v0 2후보 $8 + 기준선 2벌 $8 + 후보 15개 $45 + 2차(이터당 1개
#   통과 가정) $18 = **$79**. 채택이 나면 기준선 재측정 $8 과 최종 test 3벌 두 프롬프트 $26 이
#   붙어 $113 이다. 예산 가드는 `BudgetExceeded` 로 런을 **죽이므로** 최악을 덮는 값으로 잡는다 —
#   상한을 올리는 것이 지출을 올리지는 않는다.
#
#   지출의 98.4% 가 분절 채점이다(judge33b: $38.43 / $39.13). 에이전트 호출은 전부 합쳐 $0.07 이다.
#   비용 레버는 후보 수와 벌 수뿐이고, 모델·사고량을 빼면 그 둘밖에 없다.
#
#   tmux new-session -d -s judge34 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge34/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 추적은 끈다 — 월 한도가 소진돼 429 만 돌아오고 로그가 그 오류로 가득 찬다(judge33b 는 23분에
# 152KB 였고 대부분 소음이었다). 비용 집계는 `Usage.by_purpose` 가 맡아 추적과 무관하다.
export LANGSMITH_TRACING=0
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge34.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge34 \
    --generate-v0 --v0-candidates 2 --resume \
    --candidate-roles fallback,single_small,replace,procedure --candidates-cross \
    --candidates-cap 8 --findings-max 2 --full-score-max 8 \
    --pe-verbatim --critic-both-kinds \
    --screen-n 0 \
    --baseline-draws 2 --confirm-draws 3 --final-draws 3 \
    --gate-rule mean --gate-min 0.003 --adopt-rule lo --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations "${ITER:-3}" \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-140}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge34 exit=$rc" >> "$LOG"
