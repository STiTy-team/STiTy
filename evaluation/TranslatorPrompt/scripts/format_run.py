"""2.4 실행기. 모델 하나 × 조건 × 문맥(0/16) × (일반 398 + 스트레스 90) 을 번역하고 위반을 판정한다.

요청끼리 독립이라(문맥은 1.1 로그에서 고정해 가져온다) API 는 병렬로 보낸다. 지연은 재지 않는다.
결과는 results/<run_id>/<model>/format.jsonl 에 호출마다 붙이고, 다시 돌리면 끝난 것을 건너뛴다.

    PYTHONPATH=$PWD python -u evaluation/TranslatorPrompt/scripts/format_run.py --model gpt-6-luna
"""
import argparse
import random
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE.parent / "LongContextMT" / "scripts"))

from tp.format_prompts import build, extract  # noqa: E402
from tp.violations import check  # noqa: E402
from lcmt.backends import CostLedger, LocalBf16, LunaChat  # noqa: E402
from run_translate import read_env_key  # noqa: E402
from dctx.ledger import append, now, read_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None, help="세트마다 앞 몇 개만 (스모크)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "configs" / "format.yml").read_text(encoding="utf-8"))
    m = cfg["models"][args.model]
    run_dir = HERE / "results" / (args.run_id or cfg["run_id"])
    out_path = run_dir / args.model / "format.jsonl"

    talk = read_rows((HERE / cfg["talk"]).resolve())
    stress = read_rows(HERE / cfg["stress"])
    chain_rows = read_rows((HERE / cfg["context_source"]).resolve() / args.model / "translations.jsonl")
    chain = {r["idx"]: r["hyp"] for r in chain_rows if r["n"] == 16}
    if len(chain) != len(talk):
        raise SystemExit(f"N=16 chain incomplete for {args.model}: {len(chain)}/{len(talk)}")
    if args.limit:
        talk, stress = talk[:args.limit], stress[:args.limit]

    rng = random.Random(cfg["seed"])
    items = [{"set": "talk", "id": f"t{r['idx']:03d}", "src": r["en"], "ref": r["ko"], "pos": r["idx"]}
             for r in talk]
    items += [{"set": "stress", "id": s["id"], "category": s["category"], "src": s["src"], "ref": None,
               "pos": s.get("origin_idx", rng.randrange(16, len(chain)))} for s in stress]

    conds = [(c, c, {}) for c in cfg["conditions"]]
    if m["kind"] == "local":
        conds += [(name, x["prompt"], x.get("gen_kwargs", {})) for name, x in cfg["local_extra"].items()]
    jobs = [(name, prompt, gk, n, it) for name, prompt, gk in conds for n in cfg["context_n"]
            for it in items]
    done = {(r["cond"], r["ctx_n"], r["id"]) for r in read_rows(out_path)}
    jobs = [j for j in jobs if (j[0], j[3], j[4]["id"]) not in done]
    print(f"[{args.model}] {len(done)} done, {len(jobs)} to go", flush=True)

    if m["kind"] == "local":
        backend, ledger = LocalBf16(args.model, m["model"], m.get("max_new_tokens", 256)), None
    else:
        ledger = CostLedger(run_dir / "api_usage.jsonl", cfg["budget_usd"]["translation"])
        backend = LunaChat(args.model, m, read_env_key(m["key_env"], [str(HERE.parents[1] / e) for e in cfg["env_files"]]),
                           cfg["prices"][m["model"]], ledger, cfg["api_timeout_sec"])
    append(run_dir / args.model / "load.jsonl", {"at": now(), "model": args.model, **backend.load()})

    def run(job):
        name, prompt, gk, n, it = job
        if ledger and ledger.over_budget():
            raise SystemExit(f"budget ${ledger.budget} reached — stopping")
        prev = [chain[j] for j in range(max(0, it["pos"] - n), it["pos"])] if n else []
        system, user = build(prompt, it["src"], prev)
        tag = {"model_name": args.model, "cond": name, "ctx_n": n, "id": it["id"]}
        if isinstance(backend, LocalBf16):
            res = backend.translate(system, user, gen_kwargs=gk)
        else:
            res = backend.translate(system, user, tag=tag)
        hyp, js = extract(prompt, res["raw_output"])
        append(out_path, {
            "at": now(), "model": args.model, "cond": name, "prompt": prompt, "gen_kwargs": gk,
            "ctx_n": n, **{k: it.get(k) for k in ("set", "id", "category", "src", "ref", "pos")},
            "raw_output": res["raw_output"], "hyp": hyp, "json_status": js,
            "violations": check(res["raw_output"], hyp, it["src"], prev, res["truncated"], js),
            "output_tokens": res["output_tokens"], "input_tokens": res["input_tokens"],
            "truncated": res["truncated"], "api_meta": res["api_meta"]})

    if m["kind"] == "local":
        for k, job in enumerate(jobs):
            run(job)
            if k % 200 == 0:
                print(f"[{args.model}] {k}/{len(jobs)}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=m.get("workers", 4)) as pool:
            for k, f in enumerate([pool.submit(run, j) for j in jobs]):
                f.result()
                if k % 200 == 0:
                    print(f"[{args.model}] {k}/{len(jobs)}, spent ${ledger.spent():.4f}", flush=True)
    print(f"[{args.model}] finished", flush=True)


if __name__ == "__main__":
    main()
