#!/bin/bash
# prefix13 GRPO: the adopted ceren-lat recipe (B) on top of merged/prefix13_full, rollout prompts
# carrying the source-language tag (--source_tag 1) so RL matches the SFT format. The reward already
# holds every adopted axis: COMET (cross-lang) / 1-CER (same-lang verbatim), latency (lam), shortness
# (mu), wrong-script language-consistency (nu - the adopted ㄷ), NLI contradiction (rho), silence-tail
# hallucination (kappa). After merge the LID head is refit AGAIN - GRPO's LoRA also moves audio_tower.
# Run after prefix13_chain.sh printed P13S_DONE. Markers P13G.
cd "$(git rev-parse --show-toplevel)" || exit 1
set -o pipefail
trap 'echo "P13G_EXIT $(date "+%F %T")"' EXIT
set -a; . ./.env; set +a
export PATH=$HOME/.local/bin:$PATH

roomy() { local ok=0; while [ $ok -lt 3 ]; do
  if [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -lt 3 ] && sleep 60; done; }

PY=".venv/bin/python"
CPY=".venv-tpm/bin/python"
[ -x "$CPY" ] || CPY="$PY"
GOUT=models/zprobe/grpo/prefix13_grpo_cerenlat
A=models/zprobe/adapters/prefix13_grpo_cerenlat
LOG=models/zprobe/logs
mkdir -p "$GOUT" "$LOG"

echo "$(date '+%F %T') P13G MARK grpo 400 (B recipe + source_tag)"
if [ ! -d "$A" ]; then
  for TRY in 1 2 3; do
    roomy 20000
    $CPY -u core/research/zprobe/train/comet_server.py --port 8777 > $LOG/comet_server_p13g.log 2>&1 & CS=$!
    until grep -q "ready" $LOG/comet_server_p13g.log || ! kill -0 $CS 2>/dev/null; do sleep 5; done
    kill -0 $CS 2>/dev/null || { echo "P13G_FAIL comet_server"; exit 1; }
    PYTHONPATH=Qwen3-ASR $PY -u core/research/zprobe/train/grpo_prefix.py \
      --policy_model models/zprobe/merged/prefix13_full --out "$GOUT" --source_tag 1 \
      --steps 400 --lam 0.1 --sil_frac 0.25 --mu 0.25 --anchor 1 --hard_frac 0.5 --nu 0.3 --rho 0.2 \
      --nli vicgalle/xlm-roberta-large-xnli-anli --pairs en-ko ko-en ko-ko en-en --utts 800 \
      --koko_cap 34 --enen_cap 33 --kspon_utts 33 --dailytalk_utts 33 \
      > $LOG/grpo_prefix13_cerenlat.log 2>&1
    kill $CS 2>/dev/null; wait $CS 2>/dev/null
    last=$(ls -d $GOUT/adapter_step* 2>/dev/null | sort -V | tail -1)
    grep -q GRPO_DONE $LOG/grpo_prefix13_cerenlat.log 2>/dev/null && { cp -r "$last" "$A"; break; }
    echo "GRPO try $TRY failed"; tail -3 $LOG/grpo_prefix13_cerenlat.log; sleep 30
  done
fi
[ -d "$A" ] || { echo "P13G_FAIL grpo"; exit 1; }

echo "$(date '+%F %T') P13G MARK merge"
[ -d models/zprobe/merged/prefix13_grpo_cerenlat_full ] || \
PYTHONPATH=Qwen3-ASR $PY core/research/zprobe/train/merge_adapter.py \
  --base models/zprobe/merged/prefix13_full \
  --adapter "$A" \
  --out models/zprobe/merged/prefix13_grpo_cerenlat_full || { echo "P13G_FAIL merge"; exit 1; }

echo "$(date '+%F %T') P13G MARK lid head refit (grpo merged)"
[ -f models/zprobe/data/prefix13/lid_probe_head_grpo.npz ] || \
PYTHONPATH=Qwen3-ASR $PY -u core/research/zprobe/lid/lid_probe.py \
  --model models/zprobe/merged/prefix13_grpo_cerenlat_full --skip-fused \
  --head-out models/zprobe/data/prefix13/lid_probe_head_grpo.npz || echo P13G_FAIL headrefit

for CFG in brig.koen+prefix13-grpo-lidreset brig.koen+prefix13-grpo-probelid; do
  KO=bench/runs/northstar_ko+en-ko@v2/$CFG
  EN=bench/runs/northstar_ko+en-en@v2/$CFG
  echo "$(date '+%F %T') P13G MARK bench $CFG"
  roomy 20000
  [ -f $KO/summary.json ] || uv run --project bench python -m bench --config $CFG --dataset northstar_ko+en-ko || { echo P13G_FAIL bench-$CFG; continue; }
  [ -f $EN/summary.json ] || uv run --project bench python scripts/brig_rescore.py \
    --pipeline $CFG --src-dataset northstar_ko+en-ko --dst-dataset northstar_ko+en-en || echo P13G_FAIL rescore-$CFG
  for R in $KO $EN; do
    [ -f $R/summary.json ] && { [ -f $R/comet_scores.jsonl ] || uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $R 2>&1 | tail -1; }
  done
done

echo "$(date '+%F %T') P13G MARK gate report"
$PY - <<'PYEOF'
import json
print("| run | COMET | LongYAAL |")
for label, cfg in [
    ("prefix13 SFT (태그)", "brig.koen+prefix13-lidreset"),
    ("prefix13+grpoB (태그)", "brig.koen+prefix13-grpo-lidreset"),
    ("prefix13+grpoB (태그+탐침LID)", "brig.koen+prefix13-grpo-probelid"),
    ("ceren+lid (기준선)", "brig.koen+grpoceren-lidreset"),
]:
    for tgt in ("ko", "en"):
        path = f"bench/runs/northstar_ko+en-{tgt}@v2/{cfg}"
        try:
            m = json.load(open(path + "/summary.json")).get("metrics", {})
            print(f"| {label} {tgt} | {m.get('comet', 0):.3f} | {m.get('longyaal_ms', 0):.0f} |")
        except FileNotFoundError:
            print(f"| {label} {tgt} | 없음 |")
print("관문: ko 0.750 / en 0.671, 지연 무후퇴(1990/1360). 추가 판독: 같은언어 WER vs qwen-seg WER(단일 돛대 거리)")
PYEOF
echo "P13G_DONE $(date '+%F %T')"
