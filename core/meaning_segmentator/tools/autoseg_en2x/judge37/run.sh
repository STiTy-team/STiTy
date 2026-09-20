#!/bin/bash
# judge37 — 목적함수가 개정을 **못 본 것인지** 가른다. 같은 개정, 두 집계.
#
# 무엇을 묻나
#
#   judge31~34 에서 후보 서른 몇 개가 전부 ≤0 이었다. 그런데 부검을 보면 매번 **나빠진 짝과
#   좋아진 짝이 반반**이다(2126 대 2126 / 1985 대 2071 / 1938 대 2137). 절단이 바뀌는데 방향이
#   없고 합만 음수다. 그 모양은 개정이 나쁘다는 증거도 되지만, **목적함수가 개정을 못 본다는
#   증거도 된다.**
#
#   `H_set = q × (1 − max contra)` 는 채택 집합의 **최악 하나**에 지배된다. 그래서 여러 자리를
#   조금씩 낫게 만든 개정은 최악을 못 고치면 눈금이 안 움직이고, 반대로 방향 없는 재배열이
#   나쁜 자리 하나를 집합에 들이면 문장 전체가 깎인다. **기울기가 희소하다.**
#   judge36 이 곁증거를 줬다 — 그 구조를 프롬프트에 말해 줘도 −0.0026 이다. 모델은 자리별로
#   점수를 쓰고 어느 자리가 함께 채택될지 모르므로 max 를 알아도 쓸 레버가 없다.
#
#   mean 은 모델이 실제로 통제하는 것과 모양이 같다. 그래서 묻는다:
#   **같은 개정 열다섯 개가 mean 에서는 갈리는가?**
#
# 무엇을 재나
#
#   judge34 의 후보 15개 전부. `pe_edits.json` 의 편집을 v0 에 다시 적용해 복원했고 **열다섯 개
#   모두 기록된 글자 수와 정확히 일치**한다. iter 1 의 `procedure` 후보도 들어 있다 — 그때
#   `frozen_intact` 에 막혀 채점되지 않았지만 편집은 남아 있었다.
#
#   `--final-draws 1` 이라 분절이 전부 공용 캐시(`segment.json`)에 있다. judge34 가 같은 프롬프트
#   문자열로 이미 뽑아 두었으므로 **API 비용이 0 이다.** 번역도 캐시되어 있고 QE 는 로컬 GPU 다.
#   드는 것은 GPU 시간뿐이다.
#
# 읽는 법 — 이 표는 **판정이 아니다**
#
#   `--contra-agg mean` 으로 잰 값은 지금까지의 어떤 값과도 비교하면 안 된다. 다른 눈금이다.
#   런 안에서 **후보끼리** 비교하고, 두 집계에서 **순위와 분리도가 어떻게 달라지는지**만 본다.
#
#   (a) mean 에서 후보 Δ 의 폭이 max 보다 넓은가 — 넓으면 목적함수가 보기 시작한 것이다.
#   (b) 후보 순위가 두 집계에서 같은가 — 같으면 개정의 문제고, 뒤집히면 목적함수의 문제다.
#   (c) max 에서 −0.002~−0.016 으로 뭉쳐 있던 것들이 mean 에서 양수·음수로 갈리는가.
#
#   max 쪽을 먼저 돌려 judge34 가 기록한 값이 재현되는지 확인한다 — 재현되지 않으면 복원이
#   틀린 것이고 mean 쪽 표도 믿을 수 없다.
#
#   tmux new-session -d -s judge37 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge37/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge34
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge37.log
CANDS=$(ls "$A"/cands/*.txt | tr '\n' ',' | sed 's/,$//')

for AGG in max mean; do
  echo "== $(date '+%F %T') judge37 contra-agg=$AGG 시작" >> "$LOG"
  .venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
      --from-run en2x/en-multi/run30 --run-id "en2x/en-multi/judge37_$AGG" \
      --score-only --final-split dev --final-draws 1 --contra-agg "$AGG" \
      --prompt "$A/prompt_v0.txt" \
      --score-baseline "$A/prompt_v0.txt" \
      --score-prompts "$CANDS" \
      --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
      --workers 720 --score-workers 8 \
      --provider openai --model gpt-5-mini --budget "${BUDGET:-12}" >> "$LOG" 2>&1
  echo "== $(date '+%F %T') judge37 contra-agg=$AGG exit=$?" >> "$LOG"
done
