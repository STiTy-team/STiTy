"""Doc-COMET-w2 채점 → results/<run_id>/doc2_cache.jsonl (다시 돌리면 새 것만 잰다)

Vernikos et al.(2022), DialogueContext/score_doc_comet.py 와 같은 방식. 앞 2문장의 원문과 정답을 원문·정답 앞에
붙이고, 번역 쪽에도 **정답**의 앞 2문장을 붙인다. 앞 문장 번역 실수는 다시 깎지 않고 현재 문장만 앞뒤 흐름 안에서
잰다. 채점 문맥이 모든 조건에 같으므로 조건 간 차이는 현재 문장 번역에서만 난다.

    $METRICS_PY -u evaluation/ContextCard/scripts/card_doc_comet.py card-20260930
"""
import gc
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "LongContextMT" / "scripts"))

from score import read_jsonl  # noqa: E402

MODEL = "Unbabel/wmt22-comet-da"
WINDOW = 2


def doc_key(idx: int, hyp: str) -> str:
    return hashlib.sha1(json.dumps(["doc2", idx, hyp], ensure_ascii=False).encode()).hexdigest()


def main():
    import torch
    from comet import download_model, load_from_checkpoint

    cfg = yaml.safe_load((HERE / "configs" / "card.yml").read_text(encoding="utf-8"))
    run_dir = HERE / "results" / (sys.argv[1] if len(sys.argv) > 1 else cfg["run_id"])
    talk = {r["idx"]: r for r in read_jsonl(HERE / cfg["talk"])}
    mdir = run_dir / cfg["translator"]
    rows = read_jsonl(mdir / "translations.jsonl") + read_jsonl(mdir / "baseline_translations.jsonl")
    cache_path = run_dir / "doc2_cache.jsonl"
    have = {c["key"] for c in read_jsonl(cache_path)}
    todo = {}
    for r in rows:
        k = doc_key(r["idx"], r["hyp"])
        if k not in have:
            todo[k] = (r["idx"], r["hyp"])
    print(f"{len(rows)} rows, {len(todo)} to score", flush=True)
    if not todo:
        return
    model = load_from_checkpoint(download_model(MODEL))
    model.enable_context()
    sep = model.encoder.tokenizer.sep_token

    def join(cur, ctx):
        return f" {sep} ".join(ctx + [cur])

    ks, data = list(todo), []
    for k in ks:
        i, hyp = todo[k]
        prev = [talk[j] for j in range(max(0, i - WINDOW), i)]
        data.append({"src": join(talk[i]["en"], [p["en"] for p in prev]),
                     "mt": join(hyp, [p["ko"] for p in prev]),
                     "ref": join(talk[i]["ko"], [p["ko"] for p in prev])})
    out = model.predict(data, batch_size=32, gpus=1, progress_bar=False)
    with cache_path.open("a", encoding="utf-8") as f:
        for k, s in zip(ks, out.scores):
            f.write(json.dumps({"key": k, "idx": todo[k][0], "doc2": float(s)}, ensure_ascii=False) + "\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"doc2 done {len(ks)}", flush=True)


if __name__ == "__main__":
    main()
