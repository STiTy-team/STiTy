"""2.3 실행기. 모델 하나 × 조건 × 평가 문장 2,380 을 문맥 없이 번역한다.

요청끼리 독립이라 API 는 병렬로 보낸다. 결과는 results/<run_id>/<model>/quality.jsonl 에 호출마다
붙이고, 다시 돌리면 끝난 것을 건너뛴다. 채점은 LongContextMT/scripts/score.py (--glob "*/quality.jsonl").

    PYTHONPATH=$PWD python -u evaluation/TranslatorPrompt/scripts/quality_run.py --model gpt-6-luna
"""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE.parent / "LongContextMT" / "scripts"))

from tp.quality_prompts import build  # noqa: E402
from tp.violations import check  # noqa: E402
from lcmt.backends import CostLedger, LocalBf16, LunaChat  # noqa: E402
from run_translate import read_env_key  # noqa: E402
from dctx.ledger import append, now, read_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None, help="앞 몇 문장만 (스모크)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--conditions", nargs="*", default=None, help="설정 대신 이 조건만 (S5 추가 등)")
    ap.add_argument("--batch", type=int, default=1,
                    help="로컬 전용. 묶어서 생성한다. 순차와 약 17%% 문장에서 표현이 달라지므로(64개 중 53개 일치,"
                         " check_batch_equiv.py) 한 런 안에서는 한 가지 방식만 쓴다")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "configs" / "quality.yml").read_text(encoding="utf-8"))
    m = cfg["models"][args.model]
    run_dir = HERE / "results" / (args.run_id or cfg["run_id"])
    out_path = run_dir / args.model / "quality.jsonl"
    items = read_rows(HERE / cfg["eval"])
    if args.limit:
        items = items[:args.limit]
    conds = args.conditions or cfg["conditions"]
    done = {(r["cond"], r["id"]) for r in read_rows(out_path)}
    jobs = [(c, it) for c in conds for it in items if (c, it["id"]) not in done]
    print(f"[{args.model}] {len(done)} done, {len(jobs)} to go", flush=True)

    if m["kind"] == "local":
        backend, ledger = LocalBf16(args.model, m["model"], m.get("max_new_tokens", 256)), None
    else:
        ledger = CostLedger(run_dir / "api_usage.jsonl", cfg["budget_usd"]["translation"])
        backend = LunaChat(args.model, m, read_env_key(m["key_env"], [str(HERE.parents[1] / e) for e in cfg["env_files"]]),
                           cfg["prices"][m["model"]], ledger, cfg["api_timeout_sec"])
    append(run_dir / args.model / "load.jsonl", {"at": now(), "model": args.model, **backend.load()})

    def record(cond, it, res):
        hyp = res["raw_output"].strip()
        append(out_path, {
            "at": now(), "model": args.model, "cond": cond, "batch": args.batch,
            **{k: it.get(k) for k in ("dataset", "id", "src_lang", "tgt_lang", "src", "ref", "doc_id", "sender")},
            "raw_output": res["raw_output"], "hyp": hyp,
            "violations": check(res["raw_output"], hyp, it["src"], [], res["truncated"], None)
            if it["tgt_lang"] == "ko" else [],
            "output_tokens": res["output_tokens"], "input_tokens": res["input_tokens"],
            "truncated": res["truncated"], "api_meta": res["api_meta"]})

    def run(job):
        cond, it = job
        if ledger and ledger.over_budget():
            raise SystemExit(f"budget ${ledger.budget} reached — stopping")
        system, user = build(cond, it["src_lang"], it["tgt_lang"], it["src"])
        tag = {"model_name": args.model, "cond": cond, "id": it["id"]}
        res = (backend.translate(system, user) if isinstance(backend, LocalBf16)
               else backend.translate(system, user, tag=tag))
        record(cond, it, res)

    if m["kind"] == "local" and args.batch > 1:
        for k in range(0, len(jobs), args.batch):
            chunk = jobs[k:k + args.batch]
            outs = backend.translate_batch([build(c, it["src_lang"], it["tgt_lang"], it["src"]) for c, it in chunk])
            for (c, it), res in zip(chunk, outs):
                record(c, it, res)
            if (k // args.batch) % 30 == 0:
                print(f"[{args.model}] {k}/{len(jobs)}", flush=True)
    elif m["kind"] == "local":
        for k, job in enumerate(jobs):
            run(job)
            if k % 500 == 0:
                print(f"[{args.model}] {k}/{len(jobs)}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=m.get("workers", 4)) as pool:
            for k, f in enumerate([pool.submit(run, j) for j in jobs]):
                f.result()
                if k % 500 == 0:
                    print(f"[{args.model}] {k}/{len(jobs)}, spent ${ledger.spent():.4f}", flush=True)
    print(f"[{args.model}] finished", flush=True)


if __name__ == "__main__":
    main()
