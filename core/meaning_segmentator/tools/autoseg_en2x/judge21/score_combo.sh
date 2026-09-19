#!/bin/bash
# judge21 에서 근소 기각된 편집 둘을 test 660 에서 잰다 — 기준선은 v0.
#   단일: 이터 2 fallback (순서 규칙 추가, dev Δ +0.0078 [-0.0013, +0.0171])
#   합본: 그 위에 이터 2 prune (delete C3, dev Δ +0.0068 [-0.0027, +0.0165])
# 가산성 확인이 목적이다 — 합본이 단일보다 높아야 두 편집이 더해진다.
#   tmux new-session -d -s combo -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge21/score_combo.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge21
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge21_combo.log
SUM=$DIR/combo/summary.md
[ -f "$SUM" ] || echo "# 근소 기각 편집 — test 660 (기준선 v0)" > "$SUM"

for TAG in single_fallback combo; do
  [ -f "$DIR/combo/curve_$TAG.json" ] && { echo "== $(date '+%F %T') $TAG 이미 있음" >> "$LOG"; continue; }
  echo "== $(date '+%F %T') $TAG 시작" >> "$LOG"
  .venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
      --from-run en2x/en-multi/run28 --run-id en2x/en-multi/judge21 \
      --prompt "$DIR/combo/$TAG.txt" --score-only --score-baseline "$DIR/prompt_v0.txt" \
      --final-split test --min-gap 1 --min-chunk 2 --max-k 99 --iterations 1 \
      --k-samples 1 --workers 128 --extra-key-envs OPENAI_API_KEY_2 \
      --provider openai --model gpt-5-mini --budget 20 >> "$LOG" 2>&1
  echo "== $(date '+%F %T') $TAG exit=$?" >> "$LOG"
  for f in curve.json final_report.md; do
    [ -f "$DIR/$f" ] && cp "$DIR/$f" "$DIR/combo/${f%.*}_$TAG.${f##*.}"
  done
  .venv-autoseg/bin/python - "$DIR/combo/curve_$TAG.json" "$TAG" >> "$SUM" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1])); m = c['means']; v = c.get('vs_v0') or {}
print(f"- **{sys.argv[2]}** H_set {m['prompt']:.4f} vs v0 {m.get('v0', float('nan')):.4f} "
      f"Δ {v.get('mean', 0):+.4f} [{v.get('lo', 0):+.4f}, {v.get('hi', 0):+.4f}] / {c['by_bin']['prompt']}")
PYEOF
done
echo "== $(date '+%F %T') DONE" >> "$LOG"
