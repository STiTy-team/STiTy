"""2.1 채점 → results/<run_id>/stream_scores.jsonl (캐시, 다시 돌리면 새 것만 잰다)

nli   경계마다 (전제 = 같은 모델의 2.3 S0 문장 전체 번역, 가설 = 그 경계까지 내보낸 조각 번역을 이어 붙인 것).
      contradiction·entailment 확률. 모델은 autoseg 의 모순률과 같은 vicgalle/xlm-roberta-large-xnli-anli
      (fp16, 512 토큰에서 자름). 마지막으로 내보낸 조각 뒤는 경계가 아니다 — 그 뒤에 올 말이 없다.
kiwi  내보낸 조각마다 CometKiwi(조각 원문, 조각 번역). 정답 없이 조각만 보고 지어낸 말이 있는지 본다.
comet 내보낸 조각 번역을 이어 붙인 최종 번역의 COMET(원문 문장, 이어 붙인 것, 정답). 같은 모델의 2.3 S0 문장 전체
      번역도 같이 잰다 — 조각을 나누지 않았을 때의 기준(상한 쪽)이다.
metricx  같은 (원문 문장, 이어 붙인 것, 정답)의 MetricX-24 (낮을수록 좋음).
doc2  Doc-COMET-w2 (Vernikos et al., 2022; DialogueContext/score_doc_comet.py 와 같은 방식). 앞 2문장의 원문과
      정답을 원문·정답 앞에 붙이고, 번역 쪽에도 **정답**의 앞 2문장을 붙인다. 앞 문장 번역 실수는 다시 깎지 않고
      현재 문장만 앞뒤 흐름 안에서 잰다.

    $METRICS_PY -u evaluation/TranslatorPrompt/scripts/stream_score.py stream-20260930
"""
import gc
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "LongContextMT" / "scripts"))

from score import read_jsonl, run_comet  # noqa: E402

NLI_MODEL = "vicgalle/xlm-roberta-large-xnli-anli"
KIWI_MODEL = "Unbabel/wmt22-cometkiwi-da"
COMET_MODEL = "Unbabel/wmt22-comet-da"


def key(kind, a, b):
    return hashlib.sha1(json.dumps([kind, a, b], ensure_ascii=False).encode()).hexdigest()


def full_translations(cfg: dict, model: str) -> dict[int, str]:
    rows = read_jsonl(HERE / cfg["full_from"].format(model=model))
    return {int(r["id"][3:]): r["hyp"] for r in rows if r["cond"] == cfg["full_cond"] and r["dataset"] == "ted"}


def emitted(r: dict) -> list[tuple[str, str]]:
    return [(s, t) for s, t in r["emitted"]]


def joined(r: dict) -> str:
    return " ".join(t for _, t in emitted(r) if t).strip()


def boundaries(r: dict, full: str) -> list[tuple[str, str]]:
    """(전제, 가설) — 마지막 방출 앞의 경계마다 하나. 빈 번역을 낸 조각도 방출이지만 가설은 그대로다."""
    em = emitted(r)
    out = []
    for j in range(len(em) - 1):
        hyp = " ".join(t for _, t in em[:j + 1] if t).strip()
        if hyp:
            out.append((full, hyp))
    return out


def run_metricx(triples: list[tuple[str, str, str]]) -> list[float]:
    import torch
    from core.utils.metrics.meaning import metricx24_score

    out = metricx24_score([{"id": i, "src": a, "mt": b, "ref": c} for i, (a, b, c) in enumerate(triples)], batch_size=8)
    gc.collect()
    torch.cuda.empty_cache()
    return [out["per_item"][str(i)] for i in range(len(triples))]


def run_doc_comet(triples: list[tuple[str, str, str]]) -> list[float]:
    import torch
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(COMET_MODEL))
    model.enable_context()
    out = model.predict([{"src": a, "mt": b, "ref": c} for a, b, c in triples], batch_size=32, gpus=1,
                        progress_bar=False)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return [float(x) for x in out.scores]


def doc_triple(talk: dict, idx: int, hyp: str, sep: str, window: int = 2) -> tuple[str, str, str]:
    prev = [talk[j] for j in range(max(0, idx - window), idx)]
    join = lambda cur, ctx: f" {sep} ".join(ctx + [cur])  # noqa: E731
    return (join(talk[idx]["en"], [p["en"] for p in prev]), join(hyp, [p["ko"] for p in prev]),
            join(talk[idx]["ko"], [p["ko"] for p in prev]))


def run_nli(pairs: list[tuple[str, str]], batch_size: int = 32) -> list[tuple[float, float]]:
    import torch
    from transformers import pipeline

    pipe = pipeline("text-classification", model=NLI_MODEL, device=0, top_k=None, truncation=True,
                    max_length=512, torch_dtype=torch.float16)
    res = pipe([{"text": p, "text_pair": h} for p, h in pairs], batch_size=batch_size)

    def prob(scores, prefix):
        return next((s["score"] for s in scores if s["label"].lower().startswith(prefix)), 0.0)

    out = [(prob(s, "contr"), prob(s, "entail")) for s in res]
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    run_id = sys.argv[1]
    cfg = yaml.safe_load((HERE / "configs" / "stream.yml").read_text(encoding="utf-8"))
    run_dir = HERE / "results" / run_id
    cache_path = run_dir / "stream_scores.jsonl"
    cache = {c["key"] for c in read_jsonl(cache_path)}

    nli, kiwi, comet, mx, doc = {}, {}, {}, {}, {}
    talk = {r["idx"]: r for r in read_jsonl(HERE / cfg["seg"])}
    sep = "</s>"  # wmt22-comet-da(XLM-R) 토크나이저의 sep_token
    for mdir in sorted(p for p in run_dir.iterdir() if (p / "stream.jsonl").exists()):
        full = full_translations(cfg, mdir.name)
        for r in read_jsonl(mdir / "stream.jsonl"):
            for p, h in boundaries(r, full[r["idx"]]):
                k = key("nli", p, h)
                if k not in cache:
                    nli[k] = (p, h)
            for s, t in emitted(r):
                k = key("kiwi", s, t)
                if t and k not in cache:
                    kiwi[k] = (s, t)
            for hyp in (joined(r), full[r["idx"]]):
                k = key("comet", r["en"], [hyp, r["ref"]])
                if k not in cache:
                    comet[k] = (r["en"], hyp, r["ref"])
                k = key("metricx", r["en"], [hyp, r["ref"]])
                if k not in cache:
                    mx[k] = (r["en"], hyp, r["ref"])
                k = key("doc2", r["idx"], hyp)
                if k not in cache:
                    doc[k] = doc_triple(talk, r["idx"], hyp, sep)
    print(f"to score: nli {len(nli)}, kiwi {len(kiwi)}, comet {len(comet)}, metricx {len(mx)}, doc2 {len(doc)}",
          flush=True)

    with cache_path.open("a", encoding="utf-8") as f:
        if nli:
            ks = list(nli)
            for k, (c, e) in zip(ks, run_nli([nli[k] for k in ks])):
                f.write(json.dumps({"key": k, "kind": "nli", "premise": nli[k][0], "hyp": nli[k][1],
                                    "contra": c, "entail": e}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"nli done {len(ks)}", flush=True)
        if kiwi:
            ks = list(kiwi)
            scores = run_comet(KIWI_MODEL, False, [{"src": kiwi[k][0], "hyp": kiwi[k][1]} for k in ks], 32)
            for k, sc in zip(ks, scores):
                f.write(json.dumps({"key": k, "kind": "kiwi", "src": kiwi[k][0], "hyp": kiwi[k][1],
                                    "kiwi": sc}, ensure_ascii=False) + "\n")
            print(f"kiwi done {len(ks)}", flush=True)
        if comet:
            ks = list(comet)
            scores = run_comet(COMET_MODEL, True, [dict(zip(("src", "hyp", "ref"), comet[k])) for k in ks], 32)
            for k, sc in zip(ks, scores):
                f.write(json.dumps({"key": k, "kind": "comet", "src": comet[k][0], "hyp": comet[k][1],
                                    "ref": comet[k][2], "comet": sc}, ensure_ascii=False) + "\n")
            print(f"comet done {len(ks)}", flush=True)
        for kind, todo, fn in (("metricx", mx, run_metricx), ("doc2", doc, run_doc_comet)):
            if not todo:
                continue
            ks = list(todo)
            for k, sc in zip(ks, fn([todo[k] for k in ks])):
                f.write(json.dumps({"key": k, "kind": kind, kind: sc}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{kind} done {len(ks)}", flush=True)
    (run_dir / "stream_score_status.json").write_text(json.dumps(
        {"nli": NLI_MODEL, "kiwi": KIWI_MODEL, "comet": COMET_MODEL,
         "metricx": "google/metricx-24-hybrid-large-v2p6", "doc2": COMET_MODEL + " (context, window 2)",
         "entries": len(cache) + len(nli) + len(kiwi) + len(comet) + len(mx) + len(doc)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
