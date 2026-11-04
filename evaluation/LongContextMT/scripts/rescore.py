"""정답 두 벌(IWSLT, sol) × 채점 단위 세 가지(문장, 4문장 묶음, 강연 전체)로 1.1 번역을 다시 채점한다.

번역은 다시 돌리지 않는다. results/<run>/<모델>/translations.jsonl 의 idx >= score_from_idx 문장을 쓴다.
번역이 원문 한 줄마다 따로 나왔으므로 묶음·전체는 같은 idx 구간의 번역과 정답을 이어 붙이면 된다(정렬 불필요).

단위
- sentence: 문장 하나. COMET·XCOMET·MetricX(GPU), chrF++·spBLEU(CPU).
- block: idx 256 부터 겹치지 않게 BLOCK 문장씩(끝에 남는 문장은 버린다). 묶음 안 문장은 공백으로 잇는다.
  · Doc-COMET(wmt22-comet-da + enable_context): 입력 = 앞 문맥 문장들 </s> 묶음. 점수는 마지막 </s> 뒤
    토큰(묶음)에서만 나온다. 앞 문맥은 512 토큰 안에 들어가는 만큼 최대로 채우고(원문·정답·번역 세 입력이
    같은 문장 수), 번역 쪽 문맥도 정답 앞 문장을 쓴다(Vernikos et al. 2022) — 문맥이 모든 N 조건에서 같고
    앞 실수를 두 번 감점하지 않는다. 문맥 길이는 그 묶음의 모든 조건 중 가장 긴 번역에 맞춰 정해 조건끼리 같다.
  · MetricX: 문맥 모드가 없어 묶음만 넣는다. chrF++·spBLEU 는 묶음 문자열 하나로.
- doc: 채점 대상 문장 전부를 문자열 하나로 이어 chrF++·spBLEU (문장 경계를 넘는 n-gram 도 센다).

    # GPU 없이: 묶음 정의·문맥 길이, chrF++·spBLEU
    $METRICS_PY -u evaluation/LongContextMT/scripts/rescore.py --stage cpu
    # GPU: COMET·XCOMET·MetricX·Doc-COMET
    $METRICS_PY -u evaluation/LongContextMT/scripts/rescore.py --stage gpu [--only comet doc_comet metricx xcomet]

결과는 results/<run>/rescore/ 아래: blocks.json, scores_{sentence,block,doc}.jsonl, gpu_cache.jsonl
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
BLOCK = 4
CTX_FIRST_IDX = 240      # sol 정답이 있는 첫 줄. 두 정답 모두 이 줄부터만 문맥으로 쓴다
MAX_TOKENS = 500         # COMET 입력 상한 512 에서 특수 토큰 여유를 뺀 값
COMET_TOKENIZER = "xlm-roberta-large"  # wmt22-comet-da 의 인코더
REFS = ("iwslt", "sol")


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


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def h(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def load(run_dir: Path, start: int):
    cfg = yaml.safe_load((HERE / "configs" / "experiment.yml").read_text(encoding="utf-8"))
    talk = read_jsonl(HERE / cfg["talk"])
    sol = {r["idx"]: r["ko_sol"] for r in read_jsonl(HERE / "data" / "talk231_ref_sol.jsonl")}
    missing = [i for i in range(CTX_FIRST_IDX, len(talk)) if i not in sol]
    if missing:
        raise SystemExit(f"sol reference missing for {len(missing)} lines (first {missing[0]})")
    src = {r["idx"]: r["en"] for r in talk}
    refs = {"iwslt": {r["idx"]: r["ko"] for r in talk}, "sol": sol}
    hyps = {}  # (model, n) -> {idx: hyp}
    for p in sorted(run_dir.glob("*/translations.jsonl")):
        for r in read_jsonl(p):
            if r["idx"] >= start:
                hyps.setdefault((p.parent.name, r["n"]), {})[r["idx"]] = r["hyp"]
    idxs = list(range(start, len(talk)))
    for k, v in hyps.items():
        if set(v) != set(idxs):
            raise SystemExit(f"{k}: {len(v)}/{len(idxs)} sentences")
    return cfg, src, refs, hyps, idxs


def make_blocks(src, refs, hyps, idxs):
    """묶음 경계와 정답별 문맥 문장 수. 문맥 수는 세 입력이 모두 MAX_TOKENS 안에 들어가는 최대값."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(COMET_TOKENIZER)
    sep = tok.sep_token
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    starts = list(range(idxs[0], idxs[-1] + 1 - BLOCK + 1, BLOCK))
    blocks = []
    for s in starts:
        span = list(range(s, s + BLOCK))
        longest_mt = max((" ".join(v[i] for i in span) for v in hyps.values()), key=ntok)
        b = {"start": s, "idxs": span, "ctx": {}}
        for ref in REFS:
            cur = {"src": " ".join(src[i] for i in span), "ref": " ".join(refs[ref][i] for i in span),
                   "mt": longest_mt}
            ctx_lang = {"src": src, "ref": refs[ref], "mt": refs[ref]}
            c = 0
            while s - (c + 1) >= CTX_FIRST_IDX:
                cand = c + 1
                ok = all(ntok(f" {sep} ".join([ctx_lang[f][j] for j in range(s - cand, s)] + [cur[f]]))
                         <= MAX_TOKENS for f in cur)
                if not ok:
                    break
                c = cand
            b["ctx"][ref] = c
        blocks.append(b)
    return blocks, sep


def stage_cpu(run_dir: Path, out: Path, src, refs, hyps, idxs):
    from sacrebleu.metrics import BLEU, CHRF

    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200", effective_order=True)
    blocks, sep = make_blocks(src, refs, hyps, idxs)
    (out / "blocks.json").write_text(json.dumps({"block": BLOCK, "ctx_first_idx": CTX_FIRST_IDX,
                                                 "max_tokens": MAX_TOKENS, "sep": sep,
                                                 "blocks": blocks}, ensure_ascii=False, indent=1) + "\n")
    sent, blk, doc = [], [], []
    for (model, n), hy in sorted(hyps.items()):
        for ref in REFS:
            rf = refs[ref]
            for i in idxs:
                sent.append({"ref": ref, "model": model, "n": n, "idx": i,
                             "chrfpp": chrf.sentence_score(hy[i], [rf[i]]).score,
                             "spbleu": bleu.sentence_score(hy[i], [rf[i]]).score})
            for b in blocks:
                m, r = " ".join(hy[i] for i in b["idxs"]), " ".join(rf[i] for i in b["idxs"])
                blk.append({"ref": ref, "model": model, "n": n, "start": b["start"], "ctx": b["ctx"][ref],
                            "chrfpp": chrf.sentence_score(m, [r]).score,
                            "spbleu": bleu.sentence_score(m, [r]).score})
            m, r = " ".join(hy[i] for i in idxs), " ".join(rf[i] for i in idxs)
            doc.append({"ref": ref, "model": model, "n": n, "sentences": len(idxs),
                        "chrfpp": chrf.sentence_score(m, [r]).score,
                        "spbleu": bleu.sentence_score(m, [r]).score})
    write_jsonl(out / "scores_sentence.jsonl", sent)
    write_jsonl(out / "scores_block.jsonl", blk)
    write_jsonl(out / "scores_doc.jsonl", doc)
    (out / "cpu_status.json").write_text(json.dumps(
        {"bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature()),
         "blocks": len(blocks), "conditions": len(hyps)}, indent=2) + "\n")
    cs = [b["ctx"] for b in blocks]
    print(f"{len(blocks)} blocks of {BLOCK}; context sentences "
          + ", ".join(f"{ref} min {min(c[ref] for c in cs)} max {max(c[ref] for c in cs)}" for ref in REFS))
    print(f"wrote {len(sent)} sentence, {len(blk)} block, {len(doc)} doc rows")


def stage_gpu(run_dir: Path, out: Path, src, refs, hyps, idxs, only, batch_size):
    import torch

    blocks = {b["start"]: b for b in json.loads((out / "blocks.json").read_text())["blocks"]}
    sep = json.loads((out / "blocks.json").read_text())["sep"]
    cache_path = out / "gpu_cache.jsonl"
    cache = {c["key"]: c["score"] for c in read_jsonl(cache_path)}
    # 문장 단위 IWSLT 점수는 score.py 가 이미 잰 것을 쓴다 (키 형식이 다르므로 옮겨 담는다)
    old = {c["key"]: c for c in read_jsonl(run_dir / "scores_cache.jsonl")}

    def old_key(s, hyp, ref):
        return hashlib.sha1(json.dumps([s, hyp, ref], ensure_ascii=False).encode()).hexdigest()

    jobs = {m: {} for m in only}  # metric -> key -> triple
    sent_keys, blk_keys = [], []
    for (model, n), hy in sorted(hyps.items()):
        for ref in REFS:
            rf = refs[ref]
            for i in idxs:
                t = {"src": src[i], "mt": hy[i], "ref": rf[i]}
                row = {"ref": ref, "model": model, "n": n, "idx": i, "keys": {}}
                for m in ("comet", "xcomet", "metricx"):
                    if m not in only:
                        continue
                    k = h(m, t["src"], t["mt"], t["ref"])
                    if k not in cache and m in old.get(old_key(src[i], hy[i], rf[i]), {}):
                        cache[k] = old[old_key(src[i], hy[i], rf[i])][m]
                    if k not in cache:
                        jobs[m][k] = t
                    row["keys"][m] = k
                sent_keys.append(row)
            for s, b in blocks.items():
                span, c = b["idxs"], b["ctx"][ref]
                cur = {"src": " ".join(src[i] for i in span), "mt": " ".join(hy[i] for i in span),
                       "ref": " ".join(rf[i] for i in span)}
                row = {"ref": ref, "model": model, "n": n, "start": s, "keys": {}}
                if "doc_comet" in only:
                    ctx = {"src": [src[j] for j in range(s - c, s)], "mt": [rf[j] for j in range(s - c, s)],
                           "ref": [rf[j] for j in range(s - c, s)]}
                    t = {f: f" {sep} ".join(ctx[f] + [cur[f]]) for f in cur}
                    k = h("doc_comet", t["src"], t["mt"], t["ref"])
                    if k not in cache:
                        jobs["doc_comet"][k] = t
                    row["keys"]["doc_comet"] = k
                if "metricx" in only:
                    k = h("metricx", cur["src"], cur["mt"], cur["ref"])
                    if k not in cache:
                        jobs["metricx"][k] = cur
                    row["keys"]["metricx_block"] = k
                blk_keys.append(row)

    def flush():
        write_jsonl(cache_path, [{"key": k, "score": v} for k, v in cache.items()])

    flush()
    names = {"comet": "Unbabel/wmt22-comet-da", "doc_comet": "Unbabel/wmt22-comet-da",
             "xcomet": "Unbabel/XCOMET-XL"}
    for m in only:
        todo = list(jobs[m].items())
        if not todo:
            continue
        print(f"[{m}] scoring {len(todo)}", flush=True)
        if m == "metricx":
            from core.utils.metrics.meaning import metricx24_score

            res = metricx24_score([{"id": i, **t} for i, (_, t) in enumerate(todo)], batch_size=8)
            scores = [res["per_item"][str(i)] for i in range(len(todo))]
        else:
            from comet import download_model, load_from_checkpoint

            model = load_from_checkpoint(download_model(names[m]))
            if m == "doc_comet":
                model.enable_context()
                if not model.use_context:
                    raise SystemExit("COMET model does not support context")
            o = model.predict([t for _, t in todo], batch_size=batch_size, gpus=1, progress_bar=False)
            scores = [float(s) for s in o.scores]
            del model
        for (k, _), s in zip(todo, scores):
            cache[k] = s
        flush()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{m}] done", flush=True)

    # CPU 점수 파일에 GPU 점수를 붙인다
    for fname, keyrows, ident in (("scores_sentence.jsonl", sent_keys, "idx"),
                                  ("scores_block.jsonl", blk_keys, "start")):
        rows = read_jsonl(out / fname)
        by = {(r["ref"], r["model"], r["n"], r[ident]): r for r in rows}
        for kr in keyrows:
            r = by[(kr["ref"], kr["model"], kr["n"], kr[ident])]
            for m, k in kr["keys"].items():
                if k in cache:
                    r["metricx" if m == "metricx_block" else m] = cache[k]
        write_jsonl(out / fname, rows)
    print("gpu stage finished", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--only", nargs="*", default=["comet", "doc_comet", "metricx", "xcomet"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "configs" / "experiment.yml").read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir) if args.run_dir else HERE / "results" / cfg["run_id"]
    out = run_dir / "rescore"
    out.mkdir(parents=True, exist_ok=True)
    _, src, refs, hyps, idxs = load(run_dir, cfg["score_from_idx"])
    if args.stage == "cpu":
        stage_cpu(run_dir, out, src, refs, hyps, idxs)
    else:
        stage_gpu(run_dir, out, src, refs, hyps, idxs, args.only, args.batch_size)


if __name__ == "__main__":
    main()
