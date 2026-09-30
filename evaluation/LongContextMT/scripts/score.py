"""고정 지표 채점. 판정자(LLM)는 쓰지 않는다.

문장마다: COMET-DA(wmt22-comet-da, 정답 필요), XCOMET-XL(정답 필요), CometKiwi-DA(정답 없음),
MetricX-24 hybrid large(정답 필요, 낮을수록 좋음, 0~25), chrF++, spBLEU(flores200 토크나이저, 문장 단위).
MetricX 는 `metricx24` 모듈이 필요하다 — pip 로 설치되지 않는 저장소(google-research/metricx)라 받아 둔
디렉터리를 채점 환경의 site-packages 에 .pth 로 넣는다 (README "환경"). 코퍼스 chrF++·spBLEU 는 aggregate.py 가 낸다.

같은 (src, hyp, ref) 는 한 번만 잰다. 점수는 scores_cache.jsonl 에 쌓여 다시 돌리면 새 것만 잰다.
COMET 계열은 unbabel-comet 2.2.x 가 있는 채점 환경에서 돌린다 (체인 스크립트의 METRICS_PY):

    $METRICS_PY -u evaluation/LongContextMT/scripts/score.py --run-dir <run>
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

METRICS = {
    "comet": ("Unbabel/wmt22-comet-da", True),
    "xcomet": ("Unbabel/XCOMET-XL", True),
    "kiwi": ("Unbabel/wmt22-cometkiwi-da", False),
    "metricx": ("google/metricx-24-hybrid-large-v2p6", True),
}


def key(src, hyp, ref):
    return hashlib.sha1(json.dumps([src, hyp, ref], ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def run_comet(name: str, with_ref: bool, triples: list[dict], batch_size: int) -> list[float]:
    import torch
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(name))
    data = [{"src": t["src"], "mt": t["hyp"], **({"ref": t["ref"]} if with_ref else {})}
            for t in triples]
    out = model.predict(data, batch_size=batch_size, gpus=1, progress_bar=False)
    scores = [float(s) for s in out.scores]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def run_metricx(triples: list[dict]) -> list[float]:
    import torch
    from core.utils.metrics.meaning import metricx24_score

    out = metricx24_score([{"id": i, "src": t["src"], "mt": t["hyp"], "ref": t["ref"]}
                           for i, t in enumerate(triples)], batch_size=8)
    gc.collect()
    torch.cuda.empty_cache()
    return [out["per_item"][str(i)] for i in range(len(triples))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=list(METRICS))
    ap.add_argument("--glob", default="*/translations.jsonl",
                    help="run-dir 아래 결과 파일 패턴. 정답(ref)이 없는 행은 CometKiwi 만 잰다")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    rows = [r for p in sorted(run_dir.glob(args.glob)) for r in read_jsonl(p)]
    triples = {}
    for r in rows:
        triples.setdefault(key(r["src"], r["hyp"], r["ref"]),
                           {"src": r["src"], "hyp": r["hyp"], "ref": r["ref"]})
    cache_path = run_dir / "scores_cache.jsonl"
    cache = {c["key"]: c for c in read_jsonl(cache_path)}
    print(f"{len(rows)} rows, {len(triples)} unique triples, {len(cache)} cached", flush=True)

    from sacrebleu.metrics import BLEU, CHRF

    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200", effective_order=True)
    for k, t in triples.items():
        c = cache.setdefault(k, {"key": k})
        if "chrfpp" not in c and t["ref"]:
            c["chrfpp"] = chrf.sentence_score(t["hyp"], [t["ref"]]).score
            c["spbleu"] = bleu.sentence_score(t["hyp"], [t["ref"]]).score

    for metric in args.only:
        name, with_ref = METRICS[metric]
        todo = [k for k in triples if metric not in cache[k] and (triples[k]["ref"] or not with_ref)]
        if not todo:
            continue
        print(f"[{metric}] scoring {len(todo)} with {name}", flush=True)
        if metric == "metricx":
            scores = run_metricx([triples[k] for k in todo])
        else:
            scores = run_comet(name, with_ref, [triples[k] for k in todo], args.batch_size)
        for k, s in zip(todo, scores):
            cache[k][metric] = s
        with open(cache_path, "w", encoding="utf-8") as f:
            for c in cache.values():
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[{metric}] done", flush=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        for c in cache.values():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (run_dir / "score_status.json").write_text(json.dumps(
        {"metrics": {m: METRICS[m][0] for m in args.only},
         "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature()),
         "triples": len(triples)}, indent=2) + "\n")
    print("scoring finished", flush=True)


if __name__ == "__main__":
    main()
