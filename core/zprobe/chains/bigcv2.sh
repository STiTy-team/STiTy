#!/bin/bash
# bigcv data (train_big + CoVoST2 Hy-distilled rows) with more learning capacity: lr 1e-4 (was 3e-5),
# LoRA r128/alpha 256 on q/k/v/o AND the LLM MLP (was r64 attention only). bigcv's train loss went flat at ~0.7 from a
# quarter epoch; this asks whether the gap to the Hy cascade is a fitting limit. Checkpoints every 500 steps are kept
# because lr 1e-4 overfit FLEURS within half an epoch before (watch eval_loss).
#   tmux new-session -d -s bigcv2 -c /home/mobility/STiTy "bash core/zprobe/chains/bigcv2.sh > core/zprobe/chains/bigcv2.log 2>&1"
cd /home/mobility/STiTy
set -a; . ./.env; set +a
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
D=models/zprobe/data; H=$D/hy; C=$D/cvtest; A=models/zprobe/adapters; LOG=models/zprobe/logs; OUT=models/zprobe/grpo
PAIRS="en-ko en-ja en-zh ko-en ko-ja ko-zh ja-en ja-ko ja-zh zh-en zh-ko zh-ja"
NAME=bigcv2
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
quiet() { grep -v -E "pad_token_id|Warning|warn" ; }
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 18000 ]; do
  echo "$(date +%T) waiting for 18 GB free GPU"; sleep 60; done

mark "1 SFT $NAME"
if [ ! -d $A/$NAME ]; then
  $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path Qwen/Qwen3-ASR-1.7B --train_file $D/train_big_cv.jsonl --eval_file $D/val_big_local.jsonl \
    --output_dir $OUT/sft_$NAME --batch_size 2 --grad_acc 8 --lr 1e-4 --epochs 1 --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
    --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --save_steps 500 --group_by_length 0 \
    > $LOG/train_$NAME.log 2>&1; echo "SFT_EXIT=$?"
  [ -d $OUT/sft_$NAME/final ] || { echo "no final adapter"; tail -20 $LOG/train_$NAME.log; exit 1; }
  cp -r $OUT/sft_$NAME/final $A/$NAME
fi
grep -o "'eval_loss': [0-9.]*" $LOG/train_$NAME.log | tr '\n' ' '; echo

mark "2 FLEURS outputs"
for p in $PAIRS en-en ko-ko; do
  [ "$(lines $H/${NAME}_$p.jsonl)" -eq "$(lines $D/ev_big2_$p.jsonl)" ] && continue
  PROBE_ADAPTER=$A/$NAME $PY -u core/zprobe/probe_z.py tagswap --pair $p --out $H/${NAME}_$p.jsonl 2>&1 | quiet | tail -1
done

mark "3 CoVoST2 test outputs"
for t in ja zh ko; do
  [ "$(lines $C/${NAME}_en-$t.jsonl)" -eq 600 ] && continue
  PROBE_ADAPTER=$A/$NAME $PY -u core/zprobe/eval_manifest.py --manifest $C/man_en-$t.jsonl --tgt $t --out $C/${NAME}_en-$t.jsonl 2>&1 | quiet | tail -1
done

mark "4 COMET"
items=""; for p in $PAIRS; do items="$items ${NAME}_$p=$H/${NAME}_$p.jsonl"; done
$CPY -u core/zprobe/comet_eval.py --n 0 --per_sent $H/scores $items 2>&1 | quiet | tail -14
$CPY -u core/zprobe/comet_eval.py --n 0 --per_sent $C/da ${NAME}_en-ja=$C/${NAME}_en-ja.jsonl ${NAME}_en-zh=$C/${NAME}_en-zh.jsonl 2>&1 | quiet | tail -3
$CPY -u core/zprobe/comet_eval.py --n 0 --kiwi --per_sent $C/kiwi ${NAME}_en-ja=$C/${NAME}_en-ja.jsonl ${NAME}_en-zh=$C/${NAME}_en-zh.jsonl ${NAME}_en-ko=$C/${NAME}_en-ko.jsonl 2>&1 | quiet | tail -4

mark "5 report"
{
echo "## FLEURS: $NAME vs bigcv"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a bigcv --b $NAME
echo "## FLEURS: $NAME vs ASR->Hy"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a asr --b $NAME
echo "## CoVoST2 DA: $NAME vs bigcv"; $CPY core/zprobe/paired_delta.py --scores $C/da --a bigcv --b $NAME --pairs en-ja en-zh
echo "## CoVoST2 DA: $NAME vs ASR->Hy"; $CPY core/zprobe/paired_delta.py --scores $C/da --a asr --b $NAME --pairs en-ja en-zh
echo "## CoVoST2 Kiwi: $NAME vs bigcv"; $CPY core/zprobe/paired_delta.py --scores $C/kiwi --a bigcv --b $NAME --pairs en-ja en-zh en-ko
echo "## CoVoST2 Kiwi: $NAME vs ASR->Hy"; $CPY core/zprobe/paired_delta.py --scores $C/kiwi --a asr --b $NAME --pairs en-ja en-zh en-ko
NAME=$NAME $CPY - <<'EOF'
import json, os, sacrebleu
D = "models/zprobe/data"; n = os.environ["NAME"]
for p in ["en-en", "ko-ko"]:
    for k, f in [("big2", f"{D}/ev_big2_{p}.jsonl"), ("bigcv", f"{D}/hy/bigcv_{p}.jsonl"), (n, f"{D}/hy/{n}_{p}.jsonl")]:
        r = [json.loads(l) for l in open(f)]
        print(f"{k} {p} ASR chrF {sacrebleu.corpus_chrf([x['hyp'] for x in r], [[x['ref_tgt'] for x in r]]).score:.1f}")
EOF
} | tee $LOG/${NAME}_report.md
mark CHAIN_DONE
