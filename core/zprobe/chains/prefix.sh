#!/bin/bash
# Step 2 of "make the translation itself stream": prefix-to-prefix labels (build_prefix.py), one adapter (prefix1) for
# Korean and English targets, then append-only streaming (stream_bench.py --mode prefix) scored with YAAL and LAAL.
# Waits until the GPU is free (other projects share it); every stage skips when its output already exists.
#   tmux new-session -d -s prefix -c /home/mobility/STiTy "bash core/zprobe/chains/prefix.sh >> core/zprobe/chains/prefix.log 2>&1"
cd /home/mobility/STiTy
PY=${PY:-.venv/bin/python}; CPY=${CPY:-.venv-tpm/bin/python}
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
D=models/zprobe/data; K=$D/prefix; H=$D/hy; A=models/zprobe/adapters; LOG=models/zprobe/logs; OUT=models/zprobe/grpo
BENCH=${BENCH:-/home/mobility/STiTy-bench}; W=${W:-/home/mobility/zprobe_bench}
NAME=prefix1
mark() { echo "$(date '+%F %T') MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
idle() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 2000 ]; do sleep 60; done; }
freegpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 60; done; }
f() { grep --line-buffered -v -i -E "warn|pad_token|generation flags|INFO|it/s\]"; }
mkdir -p $K

mark "1 labels (waits for a free GPU)"
[ -f $K/sents.jsonl ] || $PY core/zprobe/build_prefix.py sents --out $K/sents.jsonl
if [ ! -f $K/smoke.done ]; then
  idle
  $PY -u core/zprobe/build_prefix.py phrases --sents $K/sents.jsonl --limit 64 --out $K/smoke.jsonl 2>&1 | f | tail -2
  python3 - $K/smoke.jsonl <<'PYEOF' || { echo "smoke labels too poor, stopping"; exit 1; }
import json, sys, random
rs = [json.loads(l) for l in open(sys.argv[1])]; ok = [r for r in rs if not r["fail"]]
print(f"smoke ok {len(ok)}/{len(rs)}")
for r in random.Random(0).sample(ok, min(6, len(ok))):
    print(f"  [{r['lang']}->{r['tgt']}] " + " | ".join(f"{p} => {x}" for p, x in zip(r["phrases"], r["adds"])))
sys.exit(0 if len(ok) >= 0.4 * len(rs) else 1)
PYEOF
  touch $K/smoke.done
fi
[ -f $K/phrases1.done ] || { idle; $PY -u core/zprobe/build_prefix.py phrases --sents $K/sents.jsonl --out $K/phrases1.jsonl 2>&1 | f | tail -1 && touch $K/phrases1.done; }
[ -f $K/phrases1_hy.done ] || { freegpu 12000; PYTHONPATH=. $PY -u core/zprobe/build_prefix.py hyprefix --phrases $K/phrases1.jsonl --out $K/phrases1_hy.jsonl 2>&1 | f | tail -1 && touch $K/phrases1_hy.done; }
[ -f $K/phrases.done ] || { idle; $PY -u core/zprobe/build_prefix.py leak --phrases $K/phrases1_hy.jsonl --out $K/phrases.jsonl 2>&1 | f | tail -1 && touch $K/phrases.done; }
[ -f $K/aligned.done ] || { freegpu 8000; PYTHONPATH=Qwen3-ASR $PY -u core/zprobe/build_prefix.py align --phrases $K/phrases.jsonl --out $K/aligned.jsonl 2>&1 | f | tail -1 && touch $K/aligned.done; }
[ -f $D/train_$NAME.jsonl ] || $PY core/zprobe/build_prefix.py build --aligned $K/aligned.jsonl --base $D/train_seg9_mix.jsonl --wav_dir $K/cuts --out $D/train_$NAME.jsonl

mark "2 SFT"
export PYTHONPATH=Qwen3-ASR
if [ ! -d $A/$NAME ]; then
  for TRY in 1 2 3 4 5 6; do
    [ -d $OUT/sft_$NAME/final ] && break
    freegpu 18000; echo "SFT try $TRY"
    $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path $BASE --train_file $D/train_$NAME.jsonl --eval_file $D/val_seg9_mix_clean.jsonl \
      --output_dir $OUT/sft_$NAME --batch_size 2 --grad_acc 8 --lr 1e-4 --epochs 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
      --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj --save_steps 200 --group_by_length 0 > $LOG/train_$NAME.log 2>&1; echo "SFT_EXIT=$?"
  done
  [ -d $OUT/sft_$NAME/final ] || { echo "no final adapter"; tail -20 $LOG/train_$NAME.log; exit 1; }
  cp -r $OUT/sft_$NAME/final $A/$NAME
fi

mark "3 offline"
export PROBE_MODEL=$BASE
for P in en-ko ko-en ja-en zh-en zh-ko ja-ko; do
  O=$H/${NAME}_$P.jsonl
  [ "$(lines $O)" -eq "$(lines $D/ev_big2_$P.jsonl)" ] || { freegpu 8000; PROBE_ADAPTER=$A/$NAME $PY -u core/zprobe/probe_z.py tagswap --pair $P --out $O 2>&1 | f | tail -1; }
done
ITEMS=""; for P in en-ko ko-en ja-en zh-en zh-ko ja-ko; do [ -f $H/scores/${NAME}_$P.jsonl ] || ITEMS="$ITEMS ${NAME}_$P=$H/${NAME}_$P.jsonl"; done
[ -n "$ITEMS" ] && $CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores $ITEMS 2>&1 | grep -v -i warn | tail -8

mark "4 streaming (append-only)"
score() {  # score <run name> <pair>
  local N=$1 P=$2
  (cd $BENCH && PYTHONPATH=.:bench/seamless bench/.venv/bin/python /home/mobility/STiTy/core/zprobe/stream_bench_score.py \
     --export $W/export/$P --run $W/out/$N.jsonl --name fleurs_$P/$N --run-dir $W/runs/$N > /dev/null && \
   PATH=$HOME/.local/bin:$PATH uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $W/runs/$N 2>&1 | tail -1)
}
for P in en-ko ko-en ja-en zh-en; do
  X=$W/export/$P; N=zprobe-$NAME-prefix-beam1_$P; F=$W/out/$N.jsonl
  [ "$(lines $F)" -eq "$(lines $X/manifest.jsonl)" ] || { idle; mark "stream $N"; PROBE_ADAPTER=$A/$NAME $PY -u core/zprobe/stream_bench.py \
    --mode prefix --manifest $X/manifest.jsonl --tgt ${P#*-} --beams 1 --out $F 2>&1 | f | grep --line-buffered -E "DONE|Error|Traceback"; }
  grep -q laal $W/runs/$N/summary.json 2>/dev/null || score $N $P
done
mark "4b LAAL for the earlier own-<SEG> runs"
for N in $(ls $W/out | sed 's/\.jsonl$//' | grep -E "^zprobe-(segconn2|segconnx)-agree2"); do
  if [[ $N == *_* ]]; then P=${N##*_}; else P=en-ko; fi
  grep -q laal $W/runs/$N/summary.json 2>/dev/null || score $N $P
done

mark "5 report"
{
echo "## offline: $NAME vs segconnx"
$CPY core/zprobe/paired_delta.py --scores $H/scores --a segconnx --b $NAME --pairs en-ko ko-en ja-en zh-en zh-ko ja-ko
echo "## streaming (real time); prefix = append-only token streaming, agree2 = own <SEG>, two equal pieces"
echo "| run | COMET | YAAL CA ms | LAAL CA ms | updates / item | decodes / item | YAAL items |"; echo "|---|---|---|---|---|---|---|"
for N in $(ls $W/runs | grep -E "^zprobe-($NAME|segconnx|segconn2)-" | sort); do
  python3 -c "import json; s=json.load(open('$W/runs/$N/summary.json')); m=s['metrics']; c=s['counts']; g=lambda k: m.get(k) or 0; print(f\"| $N | {g('comet'):.3f} | {g('yaal_ca_ms'):.0f} | {g('laal_ca_ms'):.0f} | {g('segments_per_item'):.2f} | {g('decodes_per_item'):.1f} | {c['yaal_items']}/{c['items']} |\")"
done
} | tee $LOG/${NAME}_report.md
mark CHAIN_DONE
