#!/bin/bash
# prefix13: source-language tag (probe-LID verdict semantics) + transcription track sized to the
# translation track + the prefix12 axes (terra-fine x->en labels, ASR-context prompt, reset-tail).
# Runs the GPU stages the labeling machine could not: word alignment -> Hy prefixes for the luna
# rechunk -> terra adds for it (API, LETSUR_API_KEY) -> aligned/corpus build -> SFT -> merge ->
# LID head refit on the merged encoder (the LoRA moves audio_tower, so the head must be refit per
# merge) -> benches (no-ctx / ctx / ctx+probelid).
#
# Artifacts expected under models/zprobe/data (HF: Doo12/stity-zprobe-adapters, prefix13-handoff/):
#   prefix12/{word_times.jsonl,aligned.jsonl}  prefix13/{ko_all_manifest,fail_ko_manifest,
#   pieces_ko_fail,pieces_ok285,pieces_ok285_hy}.jsonl  relabel/{full_fine_terra,fine_ok285_terra}.jsonl
#   prefix3/{phrases1,phrases1_hy}.jsonl  conv/{kspon,dailytalk}_manifest.jsonl
#   train_prefix8b.jsonl  val_prefix9d.jsonl  silence_tail_4s.wav
# If dataset roots differ from /home/mobility, set P13_SED='s#/home/mobility/datasets#/your/root#g'
# and the chain rewrites the jsonl artifacts in place before building.
cd "$(git rev-parse --show-toplevel)" || exit 1
set -o pipefail
trap 'echo "P13S_EXIT $(date "+%F %T")"' EXIT
set -a; . ./.env; set +a
export PATH=$HOME/.local/bin:$PATH

roomy() { local ok=0; while [ $ok -lt 3 ]; do
  if [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -lt 3 ] && sleep 60; done; }

if [ -n "$P13_SED" ]; then
  echo "$(date '+%F %T') P13S MARK pathfix"
  for f in models/zprobe/data/prefix12/*.jsonl models/zprobe/data/prefix13/*.jsonl \
           models/zprobe/data/prefix3/phrases1*.jsonl models/zprobe/data/conv/*_manifest.jsonl \
           models/zprobe/data/train_prefix8b.jsonl; do
    [ -f "$f" ] && sed -i "$P13_SED" "$f"
  done
fi

echo "$(date '+%F %T') P13S MARK align"
roomy 4000
[ -f models/zprobe/data/prefix13/align.done ] || {
PYTHONPATH=Qwen3-ASR .venv/bin/python -u core/research/zprobe/lid/align_more.py \
  && touch models/zprobe/data/prefix13/align.done || { echo P13S_FAIL align; exit 1; }; }

echo "$(date '+%F %T') P13S MARK hyprefix (luna rechunk leak material)"
roomy 8000
[ -f models/zprobe/data/prefix13/pieces_ko_fail_hy.jsonl ] || \
PYTHONPATH=. .venv/bin/python core/research/zprobe/labels/build_connective_en.py hyprefix \
  --pieces models/zprobe/data/prefix13/pieces_ko_fail.jsonl \
  --out models/zprobe/data/prefix13/pieces_ko_fail_hy.jsonl || { echo P13S_FAIL hyprefix; exit 1; }

echo "$(date '+%F %T') P13S MARK terra (fail692)"
[ -f models/zprobe/data/relabel/fine_fail692_terra.jsonl ] || \
.venv/bin/python -u core/research/zprobe/labels/relabel_adds_en.py \
  --pieces models/zprobe/data/prefix13/pieces_ko_fail.jsonl \
  --hy models/zprobe/data/prefix13/pieces_ko_fail_hy.jsonl \
  --langs ko --model gpt-6.1-terra --workers 256 --budget 4 \
  --out models/zprobe/data/relabel/fine_fail692_terra.jsonl || { echo P13S_FAIL terra; exit 1; }

echo "$(date '+%F %T') P13S MARK aligned"
.venv/bin/python core/research/zprobe/labels/build_prefix13_aligned.py || { echo P13S_FAIL aligned; exit 1; }

echo "$(date '+%F %T') P13S MARK corpus"
[ -f models/zprobe/data/prefix13/train_base.jsonl ] || \
.venv/bin/python core/research/zprobe/labels/build_prefix10.py \
  --aligned models/zprobe/data/prefix13/aligned.jsonl \
  --out models/zprobe/data/prefix13/train_base.jsonl || { echo P13S_FAIL base; exit 1; }
[ -f models/zprobe/data/train_prefix13.jsonl ] || \
.venv/bin/python core/research/zprobe/labels/build_prefix13.py || { echo P13S_FAIL corpus; exit 1; }
.venv/bin/python - <<'PYEOF'
import json
import os
if not os.path.exists("models/zprobe/data/val_prefix13.jsonl"):
    NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
    with open("models/zprobe/data/val_prefix13.jsonl", "w", encoding="utf-8") as out:
        for l in open("models/zprobe/data/val_prefix9d.jsonl", encoding="utf-8"):
            r = json.loads(l)
            s = r.get("pair", "x-x").split("-")[0]
            if s in NAME and r.get("text", "").startswith("language "):
                r["text"] = f"speech {NAME[s]} " + r["text"]
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("val_prefix13 written")
PYEOF

echo "$(date '+%F %T') P13S MARK sft"
roomy 20000
sft() { .venv/bin/python -u Qwen3-ASR/finetuning/qwen3_asr_sft.py \
  --model_path models/Qwen3-ASR-1.7B-koen-seg-mix-merged \
  --train_file models/zprobe/data/train_prefix13.jsonl \
  --eval_file models/zprobe/data/val_prefix13.jsonl \
  --output_dir models/zprobe/adapters/prefix13 \
  --use_lora 1 --no_seg 1 --load_best 0 \
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --lora_targets q_proj,k_proj,v_proj,o_proj \
  --lr 1e-4 --epochs 1 --batch_size 2 --grad_acc 8 \
  --save_steps 200 --save_total_limit 2 --resume 1 "$@"; }
[ -f models/zprobe/adapters/prefix13/final/adapter_model.safetensors ] || \
sft --gradient_checkpointing 0 || { rm -rf ~/.cache/huggingface/datasets/json; sft --gradient_checkpointing 1; } \
  || { echo P13S_FAIL sft; exit 1; }

echo "$(date '+%F %T') P13S MARK merge"
[ -d models/zprobe/merged/prefix13_full ] || \
PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/train/merge_adapter.py \
  --base models/Qwen3-ASR-1.7B-koen-seg-mix-merged \
  --adapter models/zprobe/adapters/prefix13/final \
  --out models/zprobe/merged/prefix13_full || { echo P13S_FAIL merge; exit 1; }

echo "$(date '+%F %T') P13S MARK lid head refit"
[ -f models/zprobe/data/prefix13/lid_probe_head.npz ] || \
PYTHONPATH=Qwen3-ASR .venv/bin/python -u core/research/zprobe/lid/lid_probe.py \
  --model models/zprobe/merged/prefix13_full --skip-fused \
  --head-out models/zprobe/data/prefix13/lid_probe_head.npz || echo P13S_FAIL headrefit

for CFG in brig.koen+prefix13-lidreset brig.koen+prefix13-ctx-lidreset brig.koen+prefix13-ctx-probelid; do
  KO=bench/runs/northstar_ko+en-ko@v2/$CFG
  EN=bench/runs/northstar_ko+en-en@v2/$CFG
  echo "$(date '+%F %T') P13S MARK bench $CFG"
  roomy 20000
  [ -f $KO/summary.json ] || uv run --project bench python -m bench --config $CFG --dataset northstar_ko+en-ko || { echo P13S_FAIL bench-$CFG; continue; }
  [ -f $EN/summary.json ] || uv run --project bench python scripts/brig_rescore.py \
    --pipeline $CFG --src-dataset northstar_ko+en-ko --dst-dataset northstar_ko+en-en || echo P13S_FAIL rescore-$CFG
  for R in $KO $EN; do
    [ -f $R/summary.json ] && { [ -f $R/comet_scores.jsonl ] || uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $R 2>&1 | tail -1; }
  done
done

echo "$(date '+%F %T') P13S MARK report"
.venv/bin/python - <<'PYEOF'
import json
print("| run | COMET | LongYAAL |")
for label, cfg in [
    ("v10ts (직전 SFT)", "brig.koen+prefix10ts-lidreset"),
    ("prefix13 (태그, ctx 끔)", "brig.koen+prefix13-lidreset"),
    ("prefix13 (태그+ctx)", "brig.koen+prefix13-ctx-lidreset"),
    ("prefix13 (태그+ctx+탐침LID)", "brig.koen+prefix13-ctx-probelid"),
    ("ceren+lid (기준선)", "brig.koen+grpoceren-lidreset"),
]:
    for tgt in ("ko", "en"):
        path = f"bench/runs/northstar_ko+en-{tgt}@v2/{cfg}"
        try:
            m = json.load(open(path + "/summary.json")).get("metrics", {})
            print(f"| {label} {tgt} | {m.get('comet', 0):.3f} | {m.get('longyaal_ms', 0):.0f} |")
        except FileNotFoundError:
            print(f"| {label} {tgt} | 없음 |")
print("판독: ① en 지연 1500대(잘게+선방출 라벨) ② 전사/번역 모드 분리 효과 ③ ctx 어휘 교정 ④ 탐침 LID 접기·리셋")
PYEOF
echo "P13S_DONE $(date '+%F %T')"
