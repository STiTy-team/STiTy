#!/bin/bash
# run13 프롬프트의 내용을 현재 형식으로 옮겨 홀드아웃에서 한 번씩 잰다 — 무엇이 값을 하는지 가른다.
#
#   tmux new-session -d -s probe-port -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/probe_port.sh"
#
# 구조(출력 형식·H_set·min_gap 1·판정 방식)는 현재 것으로 고정하고 **내용만** 바꾼다. 후보를 고르지
# 않고 미리 정한 프롬프트를 test 200문장에서 한 번 재므로 judge19 를 망친 선별 편향이 없다.
#   p1 전부 이식 / p2 어휘 목록 뺌 / p3 등급표 뺌 / p4 2단 절차 뺌 / p5 v0 을 길이만 늘림(대조군)
# 다섯을 **순서대로** 돈다 — 분절 캐시가 하드링크로 묶여 있어 동시 실행은 서로를 덮는다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-autoseg/bin/python}
RUN=${RUN:-probe_port}
FROM=${FROM:-en2x/en-multi/run27}
DIR=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/$RUN
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/$RUN.log
SUM=$DIR/summary.md
mkdir -p "$(dirname "$LOG")" "$DIR"

# 비교 기준선은 judge17~19 가 쓴 v0 그대로다 — 최종 블록이 run_dir/prompt_v0.txt 를 기준으로 읽는다.
cp -n $DIR/prompts/p0_v0.txt $DIR/prompt_v0.txt
[ -f "$SUM" ] || echo "# run13 내용 이식 — test 200문장 홀드아웃 (기준선 = judge v0)" > "$SUM"

for tag in p1_full p2_no_cues p3_no_ladder p4_no_procedure p5_v0_padded; do
  [ -f "$DIR/curve_$tag.json" ] && { echo "== $(date '+%F %T') $tag 이미 있음 — 건너뜀" >> "$LOG"; continue; }
  echo "== $(date '+%F %T') $tag 시작" >> "$LOG"
  $PY -u -m core.meaning_segmentator.autoseg.loop_judge \
      --from-run $FROM --run-id en2x/en-multi/$RUN \
      --prompt $DIR/prompts/$tag.txt --score-only --score-baseline $DIR/prompts/p0_v0.txt \
      --min-gap 1 --min-chunk 2 --max-k 99 --iterations 1 \
      --k-samples 3 --workers 128 --extra-key-envs OPENAI_API_KEY_2 \
      --provider openai --model gpt-5-mini --budget 60 >> "$LOG" 2>&1
  echo "== $(date '+%F %T') $tag exit=$?" >> "$LOG"
  for f in curve.json final_report.md; do
    [ -f "$DIR/$f" ] && cp "$DIR/$f" "$DIR/${f%.*}_$tag.${f##*.}"
  done
  # 요약 한 줄 — 산출물이 지워져도 남는다
  $PY - "$DIR/curve_$tag.json" "$tag" >> "$SUM" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1])); b = c.get("vs_v0") or {}
m = c.get("means", {})
print(f"- **{sys.argv[2]}** H_set {m.get('prompt')} vs v0 {m.get('v0')} "
      f"Δ {b.get('mean'):+.4f} [{b.get('lo'):+.4f}, {b.get('hi'):+.4f}] / {c.get('by_bin',{}).get('prompt')}")
PYEOF
done
echo "== $(date '+%F %T') 끝 — 요약은 $SUM" >> "$LOG"
