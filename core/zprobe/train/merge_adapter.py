"""Fold a LoRA adapter into its base model and save a standalone checkpoint (for vLLM, or as the base of a further
adapter such as grpo_prefix.py's). The base's non-weight files (tokenizer, processor, chat template) are copied over;
a missing thinker.lm_head.weight in the saved file is expected (the config ties it to the embeddings).

usage: PYTHONPATH=Qwen3-ASR python core/zprobe/train/merge_adapter.py --base <model dir> --adapter <adapter dir> --out <dir>"""
import argparse, os, shutil

import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
m = Qwen3ASRModel.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu")
mm = PeftModel.from_pretrained(m.model, a.adapter).merge_and_unload()
mm.generation_config.temperature = None
mm.save_pretrained(a.out)
for f in os.listdir(a.base):
    src = os.path.join(a.base, f)
    if f.endswith((".safetensors", ".bin", ".index.json")) or os.path.exists(os.path.join(a.out, f)):
        continue
    (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, os.path.join(a.out, f))
print("merged ->", a.out, sorted(os.listdir(a.out)))
