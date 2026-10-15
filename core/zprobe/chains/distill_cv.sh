#!/bin/bash
# Distillation: CoVoST2 English train clips (one shard, new sentences) with gold-English -> Hy-MT2 ko/ja/zh targets,
# added to big2's training set (train_big, 31.6k FLEURS rows). Same recipe as big2 (base Qwen3-ASR-1.7B, LoRA r64 qkvo,
# lr 3e-5, dropout 0.1, 1 epoch). Compared with big2 and the Hy cascade on the same FLEURS ids.
#   tmux new-session -d -s distill-cv -c /home/mobility/STiTy "bash core/zprobe/chains/distill_cv.sh > core/zprobe/chains/distill_cv.log 2>&1"
# Steps skip when their output exists, so rerunning resumes.
cd /home/mobility/STiTy
set -a; . ./.env; set +a
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
D=models/zprobe/data; H=$D/hy; A=models/zprobe/adapters; LOG=models/zprobe/logs; OUT=models/zprobe/grpo
CV=/home/mobility/datasets/covost2_train_en
SHARD=/home/mobility/datasets/covost2/en_ja/train-00000-of-00008.parquet
PAIRS="en-ko en-ja en-zh ko-en ko-ja ko-zh ja-en ja-ko ja-zh zh-en zh-ko zh-ja"
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
quiet() { grep -v -E "pad_token_id|Warning|warn" ; }
gpu_wait() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge $1 ]; do
  echo "$(date +%T) waiting for $1 MB free GPU"; sleep 60; done; }
until [ -f $SHARD ]; do echo "$(date +%T) waiting for $SHARD"; sleep 30; done

mark "1 audio"
[ -f $CV/meta.jsonl ] || $PY -u core/zprobe/build_distill_cv.py audio --parquet $SHARD --out_dir $CV --workers 16
[ -f $CV/meta.jsonl ] || { echo "no meta"; exit 1; }

mark "2 Hy labels"
gpu_wait 8000
$PY -u core/zprobe/build_distill_cv.py label --out_dir $CV --batch 64 2>&1 | quiet

mark "3 merge"
[ -f $D/train_big_cv.jsonl ] || $PY core/zprobe/build_distill_cv.py merge --out_dir $CV --base $D/train_big_local.jsonl --out $D/train_big_cv.jsonl

mark "4 SFT bigcv"
gpu_wait 18000
if [ ! -d $A/bigcv ]; then
  $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path Qwen/Qwen3-ASR-1.7B --train_file $D/train_big_cv.jsonl --eval_file $D/val_big_local.jsonl \
    --output_dir $OUT/sft_bigcv --batch_size 2 --grad_acc 8 --lr 3e-5 --epochs 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.1 \
    --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj --save_steps 500 --group_by_length 0 \
    > $LOG/train_bigcv.log 2>&1; echo "SFT_EXIT=$?"
  [ -d $OUT/sft_bigcv/final ] || { echo "no final adapter"; tail -20 $LOG/train_bigcv.log; exit 1; }
  cp -r $OUT/sft_bigcv/final $A/bigcv
fi
grep -o "'eval_loss': [0-9.]*" $LOG/train_bigcv.log | tr '\n' ' '; echo

mark "5 outputs"
for p in $PAIRS en-en ko-ko; do
  [ "$(lines $H/bigcv_$p.jsonl)" -eq "$(lines $D/ev_big2_$p.jsonl)" ] && { echo "skip $p"; continue; }
  PROBE_ADAPTER=$A/bigcv $PY -u core/zprobe/probe_z.py tagswap --pair $p --out $H/bigcv_$p.jsonl 2>&1 | quiet | tail -2
done

mark "6 COMET + report"
items=""; for p in $PAIRS; do items="$items bigcv_$p=$H/bigcv_$p.jsonl"; done
$CPY -u core/zprobe/comet_eval.py --n 0 --per_sent $H/scores $items 2>&1 | quiet | tail -14
$CPY core/zprobe/hy_vs_adapter_report.py --hy $H --scores $H/scores --data $D --b bigcv --b_files "$H/bigcv_{p}.jsonl" --out $LOG/hy_vs_bigcv.md
$CPY core/zprobe/paired_delta.py --scores $H/scores --a big2 --b bigcv | tee $LOG/bigcv_vs_big2.md
$CPY - <<'EOF'
import json, sacrebleu
D = "models/zprobe/data"
for p in ["en-en", "ko-ko"]:
    for k, f in [("big2", f"{D}/ev_big2_{p}.jsonl"), ("bigcv", f"{D}/hy/bigcv_{p}.jsonl")]:
        r = [json.loads(l) for l in open(f)]
        print(f"{k} {p} ASR chrF {sacrebleu.corpus_chrf([x['hyp'] for x in r], [[x['ref_tgt'] for x in r]]).score:.1f}")
r = {x["id"]: x["hyp"] for x in map(json.loads, open(f"{D}/hy/bigcv_en-ko.jsonl"))}
for i in ["1662", "1930", "2005", "1679", "1663"]:
    print(f"case {i}: {r[i]}")
EOF
mark CHAIN_DONE
