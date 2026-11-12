#!/bin/bash
# prefix14: emission-schedule labels (design docs/superpowers/specs/2026-11-11-schedule-labels-design.md).
#   wait for the prefix13tr chain to exit -> GRPO smoke (schedule flags) -> wait for data fetch -> sentence
#   manifest -> forced alignment -> Hy word-prefix translations -> luna schedule labels -> completeness gate +
#   hygiene table -> auto_go verdict [GO file] -> training rows
#   -> SFT (lazy features, load_best) -> merge -> bench (manager queue) -> GRPO (schedule emit/wait terms,
#   latency floor; schedule_reward.py) -> merge -> bench.
# Idempotent by file existence; rerun the same command to resume. Markers P14 MARK/FAIL/STATS_READY/SFT_READY/DONE.
# Hygiene: after P14_STATS_READY auto_go.py decides; a failing verdict stops the chain, and `touch $OUT/GO`
# then rerunning overrides it. GO is cleared whenever the stats stage reruns.
cd "$(git rev-parse --show-toplevel)" || exit 1
set -o pipefail
trap 'echo "P14_EXIT $(date "+%F %T")"' EXIT
set -a; . /home/skkai/STiTy/.env; set +a
export PATH=$HOME/.local/bin:$PATH
export STITY_DATA_ROOT=/home/skkai/STiTy-team/datasets
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
DS=/home/skkai/STiTy-team/datasets
OUT=models/zprobe/data/prefix14
LOG=models/zprobe/logs; mkdir -p $LOG $OUT
LABEL_MODEL=${LABEL_MODEL:-gpt-6-luna}; LABEL_BUDGET=${LABEL_BUDGET:-35}; LABEL_REASONING=${LABEL_REASONING:-low}
# STOP_AFTER_ROWS=1 ends the chain once train/val rows exist (the training half runs elsewhere); BENCH=0 skips the
# manager bench stages (no manager reachable from that machine - submit from the 4090 with the merged models copied back)
SCHED=$OUT/schedule_${LABEL_MODEL}.jsonl

roomy() { local ok=0; while [ $ok -lt 3 ]; do
  if [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -lt 3 ] && sleep 60; done; }
stage() { echo "$(date '+%F %T') P14 MARK $*"; df -h / | tail -1 | awk '{print "  disk free " $4}'; }

stage "wait for prefix13tr chain exit"
# both /tmp-log waits are skipped once their downstream output exists (a rerun after a reboot has no /tmp logs)
[ -f $OUT/grpo_smoke.done ] || until grep -q "^P13T_EXIT" /tmp/p13t_chain.log 2>/dev/null; do sleep 120; done

stage "grpo smoke (schedule flags)"
[ -f $OUT/grpo_smoke.done ] || { roomy 16000; bash core/research/zprobe/train/prefix14_grpo_smoke.sh > $LOG/prefix14_grpo_smoke.log 2>&1 \
  && touch $OUT/grpo_smoke.done || { echo P14_FAIL grpo-smoke; tail -5 $LOG/prefix14_grpo_smoke.log; exit 1; }; }

stage "wait for data fetch"
fetch_dead() {  # FAIL marker, or EXIT without DONE (the fetch died without a FAIL marker; DONE prints before EXIT)
  grep -q '^P14D_FAIL' /tmp/p14d.log 2>/dev/null && return 0
  grep -q '^P14D_EXIT' /tmp/p14d.log 2>/dev/null && ! grep -q '^P14D_DONE' /tmp/p14d.log 2>/dev/null; }
[ -f $OUT/sentences.jsonl ] || until grep -q "^P14D_DONE" /tmp/p14d.log 2>/dev/null; do
  # p14d_retry.sh (tmux p14dr) rotates a dead fetch's log and reruns within ~60 s: confirm after a grace period
  if fetch_dead; then sleep 180; fetch_dead && { echo P14_FAIL fetch; tail -3 /tmp/p14d.log; exit 1; }; fi
  sleep 120
done

stage "sentence manifest"
[ -f $OUT/sentences.jsonl ] || $PY core/research/zprobe/labels/build_prefix14_manifest.py \
  --aligned models/zprobe/data/prefix13/aligned.jsonl --fleurs-root ~/datasets/fleurs/data \
  --aihub-manifest $DS/aihub71693/ko/manifest.jsonl --aihub-sentences 5000 \
  --aihub-en-manifest $DS/aihub71693/en/manifest.jsonl --aihub-en-sentences 2000 \
  --kosp2e-manifest $OUT/kosp2e_zeroth/manifest.jsonl --kosp2e-sentences 3000 \
  --out $OUT/sentences.jsonl --seed 14 || { echo P14_FAIL manifest; exit 1; }
echo "  sentences: $(wc -l < $OUT/sentences.jsonl)"

stage "forced alignment"
if [ ! -f $OUT/align.done ]; then
  roomy 4000
  PYTHONPATH=Qwen3-ASR $PY -u core/research/zprobe/lid/align_more.py \
    --manifest $OUT/sentences.jsonl --out $OUT/word_times.jsonl --reuse models/zprobe/data/prefix12/word_times.jsonl \
    || { echo P14_FAIL align; exit 1; }
  # align_more exits 0 even when some recs failed: check coverage before calling it done
  $PY - $OUT <<'PYEOF' || { echo P14_FAIL align-coverage; exit 1; }
import json, sys
out = sys.argv[1]
recs = {r["audio"] for l in open(f"{out}/sentences.jsonl", encoding="utf-8") if l.strip() for r in json.loads(l)["recs"]}
rows = sum(1 for l in open(f"{out}/word_times.jsonl", encoding="utf-8") if l.strip())
print(f"  align coverage: word_times rows {rows} / distinct audio in sentences {len(recs)}")
sys.exit(0 if rows >= 0.98 * len(recs) else 1)
PYEOF
  touch $OUT/align.done
fi

stage "hy word prefixes"
[ -f $OUT/hy.done ] || { roomy 6000; PYTHONPATH=. $PY -u core/research/zprobe/labels/hy_word_prefixes.py \
  --sentences $OUT/sentences.jsonl --out $OUT/hy_prefixes.jsonl && touch $OUT/hy.done \
  || { echo P14_FAIL hy; exit 1; }; }

stage "schedule labels ($LABEL_MODEL, budget $LABEL_BUDGET)"
label() { PYTHONPATH=. $PY -u core/research/zprobe/labels/label_schedule.py \
  --sentences $OUT/sentences.jsonl --hy $OUT/hy_prefixes.jsonl --model $LABEL_MODEL --workers 32 \
  --budget $LABEL_BUDGET --reasoning $LABEL_REASONING --out $SCHED; }
if [ ! -f $OUT/label.done ]; then
  label; rc=$?
  # exit 3 = API errors on >1% of rows (an outage): one more pass retries only the ERROR rows (ok/FAIL sids are kept)
  if [ $rc -eq 3 ]; then echo "  label pass 1 hit API errors; retrying the ERROR rows in 10 min"; sleep 600; label; rc=$?; fi
  [ $rc -eq 0 ] && touch $OUT/label.done || { echo P14_FAIL label rc=$rc; exit 1; }
fi
echo "  labeled: $(wc -l < $SCHED), cost: $($PY -c "
import json
t = 0.0
for l in open('$SCHED.usage.jsonl', encoding='utf-8'):
    try: t += json.loads(l)['est_usd']
    except Exception: pass
print(round(t, 2))")"

stage "completeness gate + hygiene"
GATED=${SCHED%.jsonl}.gated.jsonl
if [ ! -f $OUT/stats.done ]; then
  rm -f $OUT/GO   # a GO left from an earlier gated file would skip the hygiene decision on the new one
  roomy 6000
  if ! { PYTHONPATH=. $CPY -u core/research/zprobe/labels/schedule_stats.py \
      --schedule $SCHED --sentences $OUT/sentences.jsonl --hy $OUT/hy_prefixes.jsonl \
      --old models/zprobe/data/prefix13/aligned.jsonl --out-gated ${GATED}.tmp 2>&1 | tee $LOG/prefix14_stats.log; }; then
    echo P14_FAIL stats; exit 1
  fi
  mv ${GATED}.tmp ${GATED} && touch $OUT/stats.done
fi
echo "P14_STATS_READY $(date '+%F %T') - $LOG/prefix14_stats.log"
if [ ! -f $OUT/GO ]; then
  PYTHONPATH=. $PY core/research/zprobe/labels/auto_go.py --gated $GATED --labeled $SCHED --sentences $OUT/sentences.jsonl 2>&1 | tee -a $LOG/prefix14_stats.log
  [ ${PIPESTATUS[0]} -eq 0 ] && touch $OUT/GO || { echo "P14_FAIL hygiene - review $LOG/prefix14_stats.log; to override: touch $OUT/GO and rerun"; exit 1; }
fi

stage "training rows"
[ -f models/zprobe/data/train_prefix14.jsonl ] || $PY core/research/zprobe/labels/build_prefix14.py \
  --schedule $GATED --sentences $OUT/sentences.jsonl --word-times $OUT/word_times.jsonl \
  --sil-wav models/zprobe/data/silence_tail_4s.wav --out models/zprobe/data/train_prefix14.jsonl \
  --val-out models/zprobe/data/val_prefix14.jsonl --rows-per-rec 5 --catchup-per-rec 2 --val-cap 1500 \
  --sil-cap 800 --seg-cap 2500 --reset-rows 3000 --seed 14 || { echo P14_FAIL rows; exit 1; }
echo "  train rows: $(wc -l < models/zprobe/data/train_prefix14.jsonl), val rows: $(wc -l < models/zprobe/data/val_prefix14.jsonl)"

[ "${STOP_AFTER_ROWS:-0}" = 1 ] && { echo "P14_ROWS_READY $(date '+%F %T') - STOP_AFTER_ROWS=1, training half runs elsewhere"; exit 0; }

stage "sft"
sft() { $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py \
  --model_path models/Qwen3-ASR-1.7B-koen-seg-mix-merged \
  --train_file models/zprobe/data/train_prefix14.jsonl --eval_file models/zprobe/data/val_prefix14.jsonl \
  --output_dir models/zprobe/adapters/prefix14 --use_lora 1 --no_seg 1 --load_best 1 --lazy_features 1 \
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj \
  --lr 1e-4 --epochs 1 --batch_size 2 --grad_acc 8 --save_steps 200 --save_total_limit 2 --resume 1 "$@"; }
if [ ! -f models/zprobe/adapters/prefix14/final/adapter_model.safetensors ]; then
  roomy 20000
  sft --gradient_checkpointing 0 || sft --gradient_checkpointing 1 || { echo P14_FAIL sft; exit 1; }
fi

stage "merge"
[ -d models/zprobe/merged/prefix14_full ] || PYTHONPATH=Qwen3-ASR $PY core/research/zprobe/train/merge_adapter.py \
  --base models/Qwen3-ASR-1.7B-koen-seg-mix-merged --adapter models/zprobe/adapters/prefix14/final \
  --out models/zprobe/merged/prefix14_full || { echo P14_FAIL merge; exit 1; }
echo "P14_SFT_READY $(date '+%F %T')"

submit() {  # $1 pipeline  $2 dataset  -> job id in $JOB
  JOB=$(CFG=$1 DSN=$2 python3 - <<'PYEOF' | curl -s -XPOST localhost:29130/api/machines/4090/jobs -H 'Content-Type: application/json' -d @- | python3 -c "import json,sys; d=json.load(sys.stdin); x=d.get('data') or {}; print(x.get('id') or ''); sys.stderr.write('P14 JOB %s %s %s\n' % (d.get('ok'), x.get('id'), d.get('error','')[:200]))"
import json, os, pathlib
c, d = os.environ["CFG"], os.environ["DSN"]
print(json.dumps({"branch": "main", "pipeline": {"name": c, "yaml": pathlib.Path(f"configs/pipelines/{c}.yml").read_text()},
                  "dataset": {"name": d, "yaml": pathlib.Path(f"configs/datasets/{d}.yml").read_text()}}))
PYEOF
)
  [ -n "$JOB" ] && curl -s -XPOST localhost:29130/api/machines/4090/jobs/$JOB/run-now -H 'Content-Type: application/json' -d '{}' >/dev/null
}
wait_job() {  # block until the job id has cleanly left the manager's jobs list (done/failed move to history)
  if [ -z "$1" ]; then echo P14_FAIL bench-submit; exit 1; fi
  local n=0 rc
  while :; do
    curl -s --max-time 20 localhost:29130/api/machines | JID=$1 python3 -c "
import json, os, sys
try:
    d = json.load(sys.stdin)
    jobs = [x for m in d['machines'] if m['host'] == '4090' for x in m['jobs']]
    sys.exit(0 if any(str(x['id']) == os.environ['JID'] for x in jobs) else 1)
except Exception:
    sys.exit(2)
"
    rc=$?
    [ $rc -eq 1 ] && break
    n=$((n + 1)); [ $((n % 10)) -eq 0 ] && echo "$(date '+%F %T') P14 waiting for bench job $1 (rc=$rc)"
    sleep 60
  done
}
bench() {  # $1 marker name  $2 pipeline  $3 dataset: submit once (job id kept in $OUT/$1.job), then wait
  [ "${BENCH:-1}" = 0 ] && { echo "  BENCH=0: skipping $2 (submit from the 4090 later)"; return 0; }
  if [ -f $OUT/$1.job ]; then JOB=$(cat $OUT/$1.job); echo "  reuse bench job $JOB"
  else submit $2 $3; [ -n "$JOB" ] && echo "$JOB" > $OUT/$1.job; fi
  wait_job "$JOB"
}

stage "bench sft (manager queue)"
bench bench_sft brig.koen+prefix14-lidreset northstar_ko+en-ko

stage "grpo 400 (B recipe + schedule emit/wait terms + latency floor, en-ko/ko-en, no tag)"
GOUT=models/zprobe/grpo/prefix14_grpo_cerenlat; A=models/zprobe/adapters/prefix14_grpo_cerenlat; mkdir -p $GOUT
if [ ! -d "$A" ]; then
  for TRY in 1 2 3; do
    roomy 20000
    $CPY -u core/research/zprobe/train/comet_server.py --port 8777 > $LOG/comet_server_p14.log 2>&1 & CS=$!
    until grep -q "ready" $LOG/comet_server_p14.log || ! kill -0 $CS 2>/dev/null; do sleep 5; done
    kill -0 $CS 2>/dev/null || { echo P14_FAIL comet_server; exit 1; }
    PYTHONPATH=Qwen3-ASR $PY -u core/research/zprobe/train/grpo_prefix.py \
      --policy_model models/zprobe/merged/prefix14_full --out $GOUT --source_tag 0 \
      --steps 400 --lam 0.1 --sil_frac 0.25 --mu 0.25 --anchor 1 --hard_frac 0.5 --nu 0.3 --rho 0.2 \
      --nli vicgalle/xlm-roberta-large-xnli-anli --pairs en-ko ko-en --utts 400 \
      --schedule $GATED --sentences $OUT/sentences.jsonl --word_times $OUT/word_times.jsonl \
      --sigma_s 0.2 --sigma_l 0.1 --lat_floor 1 \
      > $LOG/grpo_prefix14_cerenlat.log 2>&1
    kill $CS 2>/dev/null; wait $CS 2>/dev/null
    last=$(ls -d $GOUT/adapter_step* 2>/dev/null | sort -V | tail -1)
    grep -q GRPO_DONE $LOG/grpo_prefix14_cerenlat.log 2>/dev/null && { cp -r "$last" "$A"; break; }
    echo "GRPO try $TRY failed"; tail -3 $LOG/grpo_prefix14_cerenlat.log; sleep 30
  done
fi
[ -d "$A" ] || { echo P14_FAIL grpo; exit 1; }

stage "merge grpo"
[ -d models/zprobe/merged/prefix14_grpo_cerenlat_full ] || PYTHONPATH=Qwen3-ASR $PY core/research/zprobe/train/merge_adapter.py \
  --base models/zprobe/merged/prefix14_full --adapter "$A" --out models/zprobe/merged/prefix14_grpo_cerenlat_full \
  || { echo P14_FAIL merge-grpo; exit 1; }

stage "bench grpo (manager queue)"
bench bench_grpo brig.koen+prefix14-grpo-lidreset northstar_ko+en-ko
echo "P14_DONE $(date '+%F %T')"
