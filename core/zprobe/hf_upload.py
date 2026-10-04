"""Upload zprobe adapters/data/logs and the SEG model to the Hugging Face Hub (private repos).
Needs a WRITE token: `huggingface-cli login` or HF_TOKEN=... in the environment.
usage: python core/zprobe/hf_upload.py [--user Doo12] [--public]"""
import argparse, os
from huggingface_hub import HfApi
ap = argparse.ArgumentParser(); ap.add_argument("--user", default="Doo12"); ap.add_argument("--public", action="store_true")
ap.add_argument("--root", default=os.path.expanduser("~/STiTy/models")); a = ap.parse_args()
api = HfApi(); who = api.whoami(); role = who.get("auth", {}).get("accessToken", {}).get("role")
assert role in ("write", "fineGrained"), f"token role is {role}; need a write token"
jobs = [("zprobe", f"{a.user}/stity-zprobe-adapters"),
        ("Qwen3-ASR-1.7B-en-covost2-dailytalk-mix-c200-merged", f"{a.user}/Qwen3-ASR-1.7B-en-seg-c200")]
for folder, repo in jobs:
    api.create_repo(repo, private=not a.public, exist_ok=True)
    print("uploading", folder, "->", repo, flush=True)
    api.upload_folder(folder_path=os.path.join(a.root, folder), repo_id=repo, commit_message=f"upload {folder} from STiTy zprobe (2026-10-04)")
    print("done", repo, flush=True)
