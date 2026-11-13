"""Fold a LoRA adapter into its base model and save a standalone checkpoint (for vLLM, or as the base of a further
adapter such as grpo_prefix.py's). The base's non-weight files (tokenizer, processor, chat template) are copied over;
a missing thinker.lm_head.weight in the saved file is expected (the config ties it to the embeddings).

usage: PYTHONPATH=Qwen3-ASR python core/research/zprobe/train/merge_adapter.py --base <model dir> --adapter <adapter dir> --out <dir>
       [--extra-rows <adapter dir>/extra_token_rows.pt]   (rows trained by qwen3_asr_sft.py --extra_train_tokens; copied
       into the embedding, and into lm_head when it is not tied, after the merge)"""

import argparse, os, shutil

import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--extra-rows", default="", help="extra_token_rows.pt from qwen3_asr_sft.py --extra_train_tokens")
a = ap.parse_args()
m = Qwen3ASRModel.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu")
mm = PeftModel.from_pretrained(m.model, a.adapter).merge_and_unload()
mm.generation_config.temperature = None
if a.extra_rows:
    rows = torch.load(a.extra_rows, map_location="cpu")
    emb = mm.thinker.get_input_embeddings()
    head = mm.thinker.get_output_embeddings()
    with torch.no_grad():
        for k, i in enumerate(rows["ids"]):
            emb.weight[i] = rows["emb"][k].to(emb.weight.dtype)
            if head is not None and head.weight is not emb.weight:
                src = rows["lm_head"][k] if rows.get("lm_head") is not None else rows["emb"][k]
                head.weight[i] = src.to(head.weight.dtype)
    print("extra rows ->", rows["ids"], "from", a.extra_rows)
mm.save_pretrained(a.out)
for f in os.listdir(a.base):
    src = os.path.join(a.base, f)
    if f.endswith((".safetensors", ".bin", ".index.json")) or os.path.exists(
        os.path.join(a.out, f)
    ):
        continue
    (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, os.path.join(a.out, f))
print("merged ->", a.out, sorted(os.listdir(a.out)))
