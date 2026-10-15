#!/bin/bash
# Connective-ending pieces for en-ko commit-and-continue: labels (SEG units -> Hy reference -> Qwen3-8B pieces in 해요체,
# rule + leak checks), SFT on koen-seg-mix (segmix recipe, en-ko rows replaced), then
#   offline FLEURS (en-ko/ja/zh, <SEG> stripped) and commit-at-SEG-units en-ko, both against segmix.
#   tmux new-session -d -s connective -c /home/mobility/STiTy "bash core/zprobe/chains/connective.sh > core/zprobe/chains/connective.log 2>&1"
cd /home/mobility/STiTy
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
D=models/zprobe/data; K=$D/connective; H=$D/hy; U=$D/segunits; A=models/zprobe/adapters; LOG=models/zprobe/logs; OUT=models/zprobe/grpo
NAME=segconn
mkdir -p $K
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
mark "1 manifest"
[ -f $K/manifest.jsonl ] || $PY core/zprobe/build_connective.py manifest --fleurs --covost 15000 --out $K/manifest.jsonl
mark "2 units"
[ "$(lines $K/units.jsonl)" -ge "$(lines $K/manifest.jsonl)" ] || PYTHONPATH=Qwen3-ASR PROBE_MODEL=$BASE $PY -u core/zprobe/build_connective.py units \
  --manifest $K/manifest.jsonl --out $K/units.jsonl 2>&1 | grep -v -i -E "warn|pad_token|Loading"
mark "3 Hy reference"
[ -f $K/hyref.jsonl ] || $PY -u core/zprobe/build_connective.py hyref --units $K/units.jsonl --out $K/hyref.jsonl 2>&1 | grep -v -i warn | tail -2
mark "4 pieces"
$PY -u core/zprobe/build_connective.py pieces --units $K/hyref.jsonl --out $K/pieces.jsonl --batch 20 2>&1 | grep -v -i warn
mark "5 build"
[ -f $D/train_$NAME.jsonl ] || $PY core/zprobe/build_connective.py build --pieces $K/pieces.jsonl --base $D/train_seg9_mix.jsonl \
  --out $D/train_$NAME.jsonl --wav_dir $K/cuts
mark "6 SFT"
if [ ! -d $A/$NAME ]; then
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 18000 ]; do sleep 60; done
  $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path $BASE --train_file $D/train_$NAME.jsonl --eval_file $D/val_seg9_mix_clean.jsonl \
    --output_dir $OUT/sft_$NAME --batch_size 2 --grad_acc 8 --lr 1e-4 --epochs 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj --save_steps 200 --group_by_length 0 > $LOG/train_$NAME.log 2>&1; echo "SFT_EXIT=$?"
  [ -d $OUT/sft_$NAME/final ] || { echo "no final adapter"; tail -20 $LOG/train_$NAME.log; exit 1; }
  cp -r $OUT/sft_$NAME/final $A/$NAME
fi
mark "7 offline outputs"
for p in en-ko en-ja en-zh; do
  [ "$(lines $H/${NAME}_$p.jsonl)" -eq "$(lines $D/ev_big2_$p.jsonl)" ] && continue
  PYTHONPATH=Qwen3-ASR PROBE_MODEL=$BASE PROBE_ADAPTER=$A/$NAME $PY -u core/zprobe/probe_z.py tagswap --pair $p --out $H/${NAME}_$p.jsonl 2>&1 | grep -v -i warn | tail -1
done
mark "8 commit at SEG units (en-ko)"
for ad in segft9_mix $NAME; do
  [ -f $U/${ad}_en-ko.jsonl ] || PYTHONPATH=Qwen3-ASR PROBE_MODEL=$BASE PROBE_ADAPTER=$A/$ad $PY -u core/zprobe/seg_units.py adapter --no_repeat 4 \
    --units $U/units_en.jsonl --ids $D/ev_big2_en-ko.jsonl --tgt ko --out $U/${ad}_en-ko.jsonl 2>&1 | grep -i -E "error|trace"
done
mark "9 COMET"
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores $(for p in en-ko en-ja en-zh; do echo -n "${NAME}_$p=$H/${NAME}_$p.jsonl "; done) \
  su_segmix_en-ko=$U/segft9_mix_en-ko.jsonl su_${NAME}_en-ko=$U/${NAME}_en-ko.jsonl 2>&1 | grep -v -i warn | tail -7
mark "10 report"
{
echo "## offline: $NAME vs segmix"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a segmix --b $NAME --pairs en-ko en-ja en-zh
echo "## commit at SEG units vs own offline: segmix"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a segmix --b su_segmix --pairs en-ko
echo "## commit at SEG units vs own offline: $NAME"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a $NAME --b su_$NAME --pairs en-ko
echo "## commit at SEG units: $NAME vs segmix"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a su_segmix --b su_$NAME --pairs en-ko
NAME=$NAME python3 - <<'PYEOF'
import json, os
n = os.environ["NAME"]; U = "models/zprobe/data/segunits"
rs = [json.loads(l) for l in open(f"{U}/{n}_en-ko.jsonl")]
multi = [r for r in rs if len(r["units_tgt"]) > 1]
closed = sum(r["units_tgt"][0].rstrip().endswith((".", "다", "요")) for r in multi)
print(f"\n{n}: first piece closes the sentence in {closed}/{len(multi)} cut sentences; empty last piece {sum(not r['units_tgt'][-1] for r in multi)}")
for r in multi[:6]:
    print("  ", r["units_tgt"])
PYEOF
} | tee $LOG/${NAME}_report.md
mark CHAIN_DONE
