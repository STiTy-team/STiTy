#!/bin/bash
# segconn2 + "wait" rows (build --wait): audio stops inside a piece, target = finished pieces only, so the adapter learns
# not to commit a piece it has not heard to the end. Then the same true-streaming test as stream_commit.sh.
#   tmux new-session -d -s segconn3 -c <repo> "bash core/zprobe/chains/segconn3.sh > core/zprobe/chains/segconn3.log 2>&1"
cd "$(dirname "$0")/../../.."
PY=${PY:-.venv/bin/python}; CPY=${CPY:-.venv/bin/python}  # PY: peft + qwen_asr, CPY: comet
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
D=models/zprobe/data; K=$D/connective; H=$D/hy; U=$D/segunits; A=models/zprobe/adapters; LOG=models/zprobe/logs; OUT=models/zprobe/grpo
NAME=segconn3
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
export PYTHONPATH=Qwen3-ASR
mark "1 build"
[ -f $D/train_$NAME.jsonl ] || $PY core/zprobe/build_connective.py build --seg_full --wait --wait_single 0.25 --pieces $K/pieces.jsonl --base $D/train_seg9_mix.jsonl \
  --out $D/train_$NAME.jsonl --wav_dir $K/cuts
mark "2 wait for stream_commit chain (skipped when its outputs are already there)"
[ -f $H/scores/sc_agree2_en-ko.jsonl ] || until grep -q CHAIN_DONE core/zprobe/chains/stream_commit.log 2>/dev/null; do sleep 60; done
mark "3 SFT"
if [ ! -d $A/$NAME ]; then
  # another project on this GPU can take the memory while the dataset map runs on CPU; retry after an OOM
  for TRY in 1 2 3 4 5 6; do
  [ -d $OUT/sft_$NAME/final ] && break
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 18000 ]; do sleep 60; done
  echo "SFT try $TRY"
  $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path $BASE --train_file $D/train_$NAME.jsonl --eval_file $D/val_seg9_mix_clean.jsonl \
    --output_dir $OUT/sft_$NAME --batch_size 2 --grad_acc 8 --lr 1e-4 --epochs 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj --save_steps 200 --group_by_length 0 > $LOG/train_$NAME.log 2>&1; echo "SFT_EXIT=$?"
  done
  [ -d $OUT/sft_$NAME/final ] || { echo "no final adapter"; tail -20 $LOG/train_$NAME.log; exit 1; }
  cp -r $OUT/sft_$NAME/final $A/$NAME
fi
export PROBE_MODEL=$BASE PROBE_ADAPTER=$A/$NAME
mark "4 offline en-ko"
[ "$(lines $H/${NAME}_en-ko.jsonl)" -eq "$(lines $D/ev_big2_en-ko.jsonl)" ] || $PY -u core/zprobe/probe_z.py tagswap --pair en-ko --out $H/${NAME}_en-ko.jsonl 2>&1 | grep -v -i warn | tail -1
mark "5 commit at SEG units (+0.3 s)"
[ -f $U/${NAME}_pad03_en-ko.jsonl ] || $PY -u core/zprobe/seg_units.py adapter --seg_prefix --pad 0.3 --no_repeat 4 --units $U/units_en.jsonl \
  --ids $D/ev_big2_en-ko.jsonl --tgt ko --out $U/${NAME}_pad03_en-ko.jsonl 2>&1 | grep -i -E "error|trace"
N=$(lines $D/ev_big2_en-ko.jsonl)
for AG in 1 2; do
  F=$U/sc3_agree${AG}_en-ko.jsonl
  mark "6 streaming agree$AG"
  [ "$(lines $F)" -eq "$N" ] || $PY -u core/zprobe/stream_commit.py --ids $D/ev_big2_en-ko.jsonl --tgt ko --agree $AG --out $F 2>&1 | grep -v -i -E "warn|pad_token"
done
mark "7 COMET"
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores ${NAME}_en-ko=$H/${NAME}_en-ko.jsonl su03_${NAME}_en-ko=$U/${NAME}_pad03_en-ko.jsonl \
  sc3_agree1_en-ko=$U/sc3_agree1_en-ko.jsonl sc3_agree2_en-ko=$U/sc3_agree2_en-ko.jsonl 2>&1 | grep -v -i warn | tail -6
mark "8 report"
{
echo "## offline: $NAME vs segconn2"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a segconn2 --b $NAME --pairs en-ko
echo "## commit at SEG units (+0.3 s) vs own offline"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a $NAME --b su03_$NAME --pairs en-ko
echo "## streaming, adapter's own <SEG>: vs own offline"
for AG in 1 2; do $CPY core/zprobe/paired_delta.py --scores $H/scores --a $NAME --b sc3_agree$AG --pairs en-ko; done
echo "## streaming: segconn3 vs segconn2"
for AG in 1 2; do $CPY core/zprobe/paired_delta.py --scores $H/scores --a sc_agree$AG --b sc3_agree$AG --pairs en-ko; done
echo "## latency"
$PY core/zprobe/seg_units.py score --files su03_segconn2=$U/segconn2_pad03_en-ko.jsonl su03_$NAME=$U/${NAME}_pad03_en-ko.jsonl \
  sc_agree1=$U/sc_agree1_en-ko.jsonl sc_agree2=$U/sc_agree2_en-ko.jsonl sc3_agree1=$U/sc3_agree1_en-ko.jsonl sc3_agree2=$U/sc3_agree2_en-ko.jsonl
} | tee $LOG/${NAME}_report.md
mark CHAIN_DONE
