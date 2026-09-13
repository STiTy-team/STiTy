#!/bin/bash
# minimal 프롬프트 둘 — 표면형 규칙 0, 판단 조건만. 기준 run24 v0 (라벨 B, k=3, 캐시), dev 215.
#   minimal_src : ① 뒤가 앞의 뜻을 뒤집나 (타깃 언급 없음)  ② ③ 앞·뒤 조각 독립 번역 가능한가
#   minimal_tgt : ① 을 "zh/ja/de 로 앞부분만 번역했을 때 전체 번역과 모순되나" 로
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv/bin/python; S=core/meaning_segmentator/tools/autoseg_en2x/run24/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run24_minimal.log
for v in minimal_src minimal_tgt; do
  echo "== $(date '+%F %T') $v start" >> $LOG
  $PY -u -m core.meaning_segmentator.tools.autoseg_en2x.rule_probe \
      --run-id en2x/en-multi/run24 --prompt core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run24/prompt_v0.txt \
      --variant-prompt $S/v0_$v.txt --split dev --k-samples 3 --tag ${v}_k3 --save-rows \
      --provider openai --model gpt-5-mini --budget 4 >> $LOG 2>&1
  echo "== $(date '+%F %T') $v rule_probe exit=$?" >> $LOG
  $PY -u $S/component_eval.py ${v}_k3 >> $LOG 2>&1
done
echo "== $(date '+%F %T') DONE" >> $LOG
