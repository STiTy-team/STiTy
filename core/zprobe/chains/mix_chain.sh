#!/bin/bash
# Full chain on the koen-seg-mix base: en-en pseudo-labels -> translation adapter (seg9 recipe) -> its eval
# launch: tmux new-session -d -s mix-chain -c <repo> "bash core/zprobe/chains/mix_chain.sh > core/zprobe/chains/mix_chain.log 2>&1"
# -> stage B GRPO (ko) -> stage C distil -> SFT -> eval -> COMET comparison.
cd /home/mobility/STiTy
set -a; . ./.env; set +a
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
D=models/zprobe/data; A=models/zprobe/adapters; OUT=models/zprobe/grpo; LOG=models/zprobe/logs
mkdir -p $OUT $LOG
mark() { echo "$(date +%T) MARK $1"; }
ev() { # label adapter -> segswap set + tagswap en-ko
  local L=$1 AD=$2
  for spec in en-ko:1:60 en-ko:2:40 zh-ko:1:60 en-ja:1:60 en-en:1:60 ko-ko:1:60; do
    IFS=: read pair k lim <<< "$spec"
    PROBE_MODEL=$BASE PROBE_ADAPTER=$AD $PY -u core/zprobe/probe_z.py segswap --pair $pair --k $k --limit $lim --out $D/sg_${L}_${pair}_k${k}.jsonl 2>&1 | grep -v -E "pad_token_id|Warning" | tail -8
  done
  PROBE_MODEL=$BASE PROBE_ADAPTER=$AD $PY -u core/zprobe/probe_z.py tagswap --pair en-ko --limit 60 --out $D/st_${L}_en-ko.jsonl 2>&1 | grep -v -E "pad_token_id|Warning" | tail -6
}
sft() { # train_file eval_file out_dir
  $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path $BASE --train_file $1 --eval_file $2 --output_dir $3 \
    --batch_size 2 --grad_acc 8 --lr 1e-4 --epochs 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj --save_steps 80 --group_by_length 0
}

mark "seglabel start"
PROBE_MODEL=$BASE $PY -u core/zprobe/probe_z.py seglabel --pair en-en --inp $D/train_seg9_local.jsonl --out $D/train_seg9_mix.jsonl 2>&1 | grep -v -E "pad_token_id|Warning" | tee $LOG/seglabel_mix_train.log
PROBE_MODEL=$BASE $PY -u core/zprobe/probe_z.py seglabel --pair en-en --inp $D/val_seg9_local.jsonl --out $D/val_seg9_mix.jsonl 2>&1 | grep -v -E "pad_token_id|Warning" | tee $LOG/seglabel_mix_val.log
per=$(grep -o "([0-9.]* per utt)" $LOG/seglabel_mix_train.log | grep -o "[0-9.]*" | head -1)
echo "SEG per utt on English (mix base): $per"
awk -v p="$per" 'BEGIN{ if (p+0 < 0.5) { print "ABORT: mix base barely segments English"; exit 1 } }' || exit 1

mark "SFT segft9_mix start"
sft $D/train_seg9_mix.jsonl $D/val_seg9_mix.jsonl $OUT/sft_segft9_mix > $LOG/train_segft9_mix.log 2>&1; echo "SFT_EXIT=$?"
[ -d $OUT/sft_segft9_mix/final ] || { echo "no segft9_mix final"; exit 1; }
rm -rf $A/segft9_mix && cp -r $OUT/sft_segft9_mix/final $A/segft9_mix

mark "eval ft9mix start"
ev ft9mix $A/segft9_mix 2>&1 | tee $LOG/eval_ft9mix.log

mark "GRPO smoke start"
$PY -u core/zprobe/grpo_seg.py --seg_model $BASE --trans_adapter $A/segft9_mix --tgt ko --utts 4 --G 4 --steps 2 --batch_utts 1 --out $OUT/smoke_mix 2>&1 | grep -E "^step|done|Traceback|Error" | tee $LOG/grpo_smoke_mix.log
grep -q "^done" $LOG/grpo_smoke_mix.log || { echo "smoke failed"; exit 1; }
mark "GRPO full start"
$PY -u core/zprobe/grpo_seg.py --seg_model $BASE --trans_adapter $A/segft9_mix --tgt ko --utts 200 --G 8 --steps 100 --batch_utts 2 --lam 0.02 --max_seg 6 --out $OUT/ko_mix 2>&1 | grep -E "^step|done|utterances|Traceback|Error" | tee $LOG/grpo_ko_mix.log
POL=$OUT/ko_mix/adapter_step100
[ -d $POL ] || { echo "no $POL"; exit 1; }

mark "distill start"
$PY -u core/zprobe/distill_seg.py --seg_model $BASE --policy_adapter $POL --trans_adapter $A/segft9_mix --tgt ko --split train \
  --limit 300 --ids_from $D/train_seg9_mix.jsonl --out $D/train_distill_komix_rows.jsonl 2>&1 | grep -v -E "pad_token_id|Warning" | tee $LOG/distill_komix.log
$PY core/zprobe/build_distill_train.py --base $D/train_seg9_mix.jsonl --distill $D/train_distill_komix_rows.jsonl --tgt ko \
  --out $D/train_distill_komix.jsonl --val_in $D/val_seg9_mix.jsonl --val_out $D/val_distill_komix.jsonl
mark "SFT distill start"
sft $D/train_distill_komix.jsonl $D/val_distill_komix.jsonl $OUT/sft_distill_komix > $LOG/train_distill_komix.log 2>&1; echo "SFT_EXIT=$?"
AD=$OUT/sft_distill_komix/final
[ -d $AD ] || { echo "no $AD"; exit 1; }
mark "eval distillmix start"
ev distillmix $AD 2>&1 | tee $LOG/eval_distillmix.log
$CPY core/zprobe/comet_eval.py --n 60 ft9mix_tag=$D/st_ft9mix_en-ko.jsonl distillmix_tag=$D/st_distillmix_en-ko.jsonl \
  ft9mix_seg_k1=$D/sg_ft9mix_en-ko_k1.jsonl distillmix_seg_k1=$D/sg_distillmix_en-ko_k1.jsonl \
  ft9mix_zhko_k1=$D/sg_ft9mix_zh-ko_k1.jsonl distillmix_zhko_k1=$D/sg_distillmix_zh-ko_k1.jsonl 2>&1 | grep -E "^\|" | tee $LOG/comet_distillmix_vs_ft9mix.md
mark "CHAIN_DONE"
