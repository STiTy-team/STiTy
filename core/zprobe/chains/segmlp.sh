#!/bin/bash
# segft9_mix recipe with LoRA on the LLM MLP too (gate/up/down + q/k/v/o). Same data, base, lr, epochs.
# Question: does the last-layer Korean word choice (lens_trans: right English concept at L22-26, wrong Korean
# first syllable at L27-28) improve when the adapter can change MLPs? Compared to segmix on the Hy-comparison ids.
#   tmux new-session -d -s segmlp -c /home/mobility/STiTy "bash core/zprobe/chains/segmlp.sh > core/zprobe/chains/segmlp.log 2>&1"
cd /home/mobility/STiTy
set -a; . ./.env; set +a
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
D=models/zprobe/data; H=$D/hy; A=models/zprobe/adapters; LOG=models/zprobe/logs; OUT=models/zprobe/grpo
PAIRS="en-ko en-ja en-zh ko-en ko-ja ko-zh ja-en ja-ko ja-zh zh-en zh-ko zh-ja"
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
quiet() { grep -v -E "pad_token_id|Warning|warn" ; }
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 18000 ]; do
  echo "$(date +%T) waiting for 18 GB free GPU"; sleep 60; done

mark "1 SFT segft9_mix_mlp"
if [ ! -d $A/segft9_mix_mlp ]; then
  $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path $BASE --train_file $D/train_seg9_mix.jsonl --eval_file $D/val_seg9_mix.jsonl \
    --output_dir $OUT/sft_segft9_mix_mlp --batch_size 2 --grad_acc 8 --lr 1e-4 --epochs 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --save_steps 80 --group_by_length 0 \
    > $LOG/train_segft9_mix_mlp.log 2>&1; echo "SFT_EXIT=$?"
  [ -d $OUT/sft_segft9_mix_mlp/final ] || { echo "no final adapter"; tail -20 $LOG/train_segft9_mix_mlp.log; exit 1; }
  cp -r $OUT/sft_segft9_mix_mlp/final $A/segft9_mix_mlp
fi
grep -o "'eval_loss': [0-9.]*" $LOG/train_segft9_mix_mlp.log | tail -3

mark "2 outputs, 12 directions + en-en"
for p in $PAIRS en-en; do
  ref=$D/ev_big2_$p.jsonl; [ $p = en-en ] && ref=$H/segmix_en-en.jsonl
  [ "$(lines $H/segmlp_$p.jsonl)" -eq "$(lines $ref)" ] && { echo "skip $p"; continue; }
  PROBE_MODEL=$BASE PROBE_ADAPTER=$A/segft9_mix_mlp $PY -u core/zprobe/probe_z.py tagswap --pair $p --out $H/segmlp_$p.jsonl 2>&1 | quiet | tail -2
done

mark "3 COMET"
items=""; for p in $PAIRS; do items="$items segmlp_$p=$H/segmlp_$p.jsonl"; done
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores $items 2>&1 | quiet | tail -14

mark "4 report"
$CPY core/zprobe/hy_vs_adapter_report.py --hy $H --scores $H/scores --data $D --b segmlp \
  --b_files "$H/segmlp_{p}.jsonl" --strip_seg --out $LOG/hy_vs_segmlp.md
$CPY - <<'EOF'
import json, numpy as np
S = "models/zprobe/data/hy/scores"; H = "models/zprobe/data/hy"
P = "en-ko en-ja en-zh ko-en ko-ja ko-zh ja-en ja-ko ja-zh zh-en zh-ko zh-ja".split()
rng = np.random.default_rng(0); alld = []
print("\n| pair | segmix | segmlp | segmlp − segmix [95%] |\n|---|---|---|---|")
for p in P:
    a = {r["id"]: r["comet"] for r in map(json.loads, open(f"{S}/segmix_{p}.jsonl"))}
    b = {r["id"]: r["comet"] for r in map(json.loads, open(f"{S}/segmlp_{p}.jsonl"))}
    d = np.array([b[i] - a[i] for i in a]); alld.append(d)
    m = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    print(f"| {p} | {np.mean(list(a.values())):.3f} | {np.mean(list(b.values())):.3f} | {d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}] |")
d = np.concatenate(alld); m = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
print(f"| **all** | | | {d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}] |")
for p in ["en-en"]:
    import sacrebleu
    for k in ["segmix", "segmlp"]:
        r = [json.loads(l) for l in open(f"{H}/{k}_{p}.jsonl")]
        strip = lambda h: h.replace("<SEG>", " ").split()
        print(f"{k} en-en ASR chrF {sacrebleu.corpus_chrf([' '.join(strip(x['hyp'])) for x in r], [[x['ref_tgt'] for x in r]]).score:.1f}  <SEG>/utt {np.mean([x['hyp'].count('<SEG>') for x in r]):.2f}")
cases = {"1662": "기하학", "1930": "소시지", "2005": "단어", "1679": "에일"}
r = {x["id"]: x["hyp"] for x in map(json.loads, open(f"{H}/segmlp_en-ko.jsonl"))}
for i, bad in cases.items():
    print(f"case {i}: {'STILL ' + bad if bad in r[i] else 'changed'} | {r[i]}")
EOF
mark CHAIN_DONE
