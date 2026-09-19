#!/bin/bash
# v0 갈래 A·B·C 를 sel 300 에서 잰다 — 최종 test 는 건드리지 않는다.
#   tmux new-session -d -s v0abc -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge20/score_v0.sh"
#
# 기준선은 A 로 고정하고 B·C 를 각각 쌍체 비교한다. 순서대로 돈다 — 분절 캐시가 한 파일이라
# 동시에 돌리면 JsonCache 가 서로의 항목을 지운다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge20
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge20_v0abc.log
SUM=$DIR/v0abc_summary.md
mkdir -p "$(dirname "$LOG")"
[ -f "$SUM" ] || echo "# v0 갈래 A·B·C — sel 300 (기준선 A)" > "$SUM"

for TAG in B C; do
  [ -f "$DIR/curve_v0$TAG.json" ] && { echo "== $(date '+%F %T') $TAG 이미 있음" >> "$LOG"; continue; }
  echo "== $(date '+%F %T') $TAG 시작" >> "$LOG"
  .venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
      --from-run en2x/en-multi/run28 --run-id en2x/en-multi/judge20 \
      --prompt $DIR/prompts/v0_$TAG.txt --score-only --score-baseline $DIR/prompts/v0_A.txt \
      --final-split sel --min-gap 1 --min-chunk 2 --max-k 99 --iterations 1 \
      --k-samples 3 --workers 128 --extra-key-envs OPENAI_API_KEY_2 \
      --provider openai --model gpt-5-mini --budget 30 >> "$LOG" 2>&1
  echo "== $(date '+%F %T') $TAG exit=$?" >> "$LOG"
  for f in curve.json final_report.md; do
    [ -f "$DIR/$f" ] && cp "$DIR/$f" "$DIR/${f%.*}_v0$TAG.${f##*.}"
  done
  .venv-autoseg/bin/python - "$DIR/curve_v0$TAG.json" "$TAG" >> "$SUM" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1])); m = c['means']; v = c.get('vs_v0') or {}
print(f"- **v0_{sys.argv[2]}** H_set {m['prompt']:.4f} vs A {m.get('v0', float('nan')):.4f} "
      f"Δ {v.get('mean', 0):+.4f} [{v.get('lo', 0):+.4f}, {v.get('hi', 0):+.4f}] / {c['by_bin']['prompt']}")
PYEOF
done
echo "== $(date '+%F %T') DONE" >> "$LOG"
