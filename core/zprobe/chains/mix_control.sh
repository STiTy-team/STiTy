#!/bin/bash
# Control for stage C: same distillation labels (prefix translations by segft9_mix) but the SEG model's OWN
# English segmentation instead of the GRPO policy. Separates "policy effect" from "label-generation effect".
# launch after mix_chain.sh (waits for its CHAIN_DONE in core/zprobe/chains/mix_chain.log):
#   tmux new-session -d -s mix-control -c <repo> "bash core/zprobe/chains/mix_control.sh > core/zprobe/chains/mix_control.log 2>&1"
cd /home/mobility/STiTy
set -a; . ./.env; set +a
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
D=models/zprobe/data; A=models/zprobe/adapters; OUT=models/zprobe/grpo; LOG=models/zprobe/logs
until grep -q "CHAIN_DONE" core/zprobe/chains/mix_chain.log; do sleep 30; done
mark() { echo "$(date +%T) MARK $1"; }
ev() { local L=$1 AD=$2
  for spec in en-ko:1:60 en-ko:2:40 zh-ko:1:60 en-en:1:60; do
    IFS=: read pair k lim <<< "$spec"
    PROBE_MODEL=$BASE PROBE_ADAPTER=$AD $PY -u core/zprobe/probe_z.py segswap --pair $pair --k $k --limit $lim --out $D/sg_${L}_${pair}_k${k}.jsonl 2>&1 | grep -v -E "pad_token_id|Warning" | tail -8
  done
  PROBE_MODEL=$BASE PROBE_ADAPTER=$AD $PY -u core/zprobe/probe_z.py tagswap --pair en-ko --limit 60 --out $D/st_${L}_en-ko.jsonl 2>&1 | grep -v -E "pad_token_id|Warning" | tail -6
}
mark "control distill start"
$PY -u core/zprobe/distill_seg.py --seg_model $BASE --policy_adapter none --trans_adapter $A/segft9_mix --tgt ko --split train \
  --limit 300 --ids_from $D/train_seg9_mix.jsonl --out $D/train_distill_koctl_rows.jsonl 2>&1 | grep -v -E "pad_token_id|Warning" | tee $LOG/distill_koctl.log
$PY core/zprobe/build_distill_train.py --base $D/train_seg9_mix.jsonl --distill $D/train_distill_koctl_rows.jsonl --tgt ko \
  --out $D/train_distill_koctl.jsonl --val_in $D/val_seg9_mix.jsonl --val_out $D/val_distill_koctl.jsonl
mark "control SFT start"
$PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path $BASE --train_file $D/train_distill_koctl.jsonl --eval_file $D/val_distill_koctl.jsonl --output_dir $OUT/sft_distill_koctl \
  --batch_size 2 --grad_acc 8 --lr 1e-4 --epochs 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj --save_steps 80 --group_by_length 0 > $LOG/train_distill_koctl.log 2>&1; echo "SFT_EXIT=$?"
AD=$OUT/sft_distill_koctl/final; [ -d $AD ] || { echo "no $AD"; exit 1; }
mark "control eval start"
ev distillctl $AD 2>&1 | tee $LOG/eval_distillctl.log
$CPY core/zprobe/comet_eval.py --n 60 ft9mix_tag=$D/st_ft9mix_en-ko.jsonl distillctl_tag=$D/st_distillctl_en-ko.jsonl distillmix_tag=$D/st_distillmix_en-ko.jsonl \
  ft9mix_seg_k1=$D/sg_ft9mix_en-ko_k1.jsonl distillctl_seg_k1=$D/sg_distillctl_en-ko_k1.jsonl distillmix_seg_k1=$D/sg_distillmix_en-ko_k1.jsonl \
  ft9mix_zhko_k1=$D/sg_ft9mix_zh-ko_k1.jsonl distillctl_zhko_k1=$D/sg_distillctl_zh-ko_k1.jsonl distillmix_zhko_k1=$D/sg_distillmix_zh-ko_k1.jsonl 2>&1 | grep -E "^\|" | tee $LOG/comet_three_way.md
mark "CONTROL_DONE"
