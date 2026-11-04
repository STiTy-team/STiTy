"""2.1 H2F: H2 의 기록에서 마지막 호출만 문장 끝맺기 문면(FINISH)으로 다시 한다.

마지막 앞 단계는 H2 결과를 그대로 가져온다. 그래서 H2 와 H2F 의 차이는 마지막 호출 하나로만 생긴다.
마지막 단계의 원문은 H2 가 마지막에 받은 것(앞에서 보류해 넘어온 원문 + 마지막 조각)과 같다.

    PYTHONPATH=$PWD python -u evaluation/TranslatorPrompt/scripts/stream_finish.py --model gpt-6-luna
"""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE.parent / "LongContextMT" / "scripts"))

from tp.stream_prompts import build_finish  # noqa: E402
from lcmt.backends import CostLedger, LocalBf16, LunaChat  # noqa: E402
from run_translate import read_env_key  # noqa: E402
from dctx.ledger import append, now, read_rows  # noqa: E402

BASE, COND = "H2", "H2F"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "configs" / "stream.yml").read_text(encoding="utf-8"))
    m = cfg["models"][args.model]
    run_dir = HERE / "results" / (args.run_id or cfg["run_id"])
    out_path = run_dir / args.model / "stream.jsonl"
    rows = read_rows(out_path)
    done = {r["idx"] for r in rows if r["cond"] == COND}
    base = [r for r in rows if r["cond"] == BASE and r["idx"] not in done]
    print(f"[{args.model}] {len(done)} done, {len(base)} to go", flush=True)

    if m["kind"] == "local":
        backend, ledger = LocalBf16(args.model, m["model"], m.get("max_new_tokens", 256)), None
    else:
        ledger = CostLedger(run_dir / "api_usage.jsonl", cfg["budget_usd"]["translation"])
        backend = LunaChat(args.model, m, read_env_key(m["key_env"], [str(HERE.parents[1] / e) for e in cfg["env_files"]]),
                           cfg["prices"][m["model"]], ledger, cfg["api_timeout_sec"])
    append(run_dir / args.model / "load.jsonl", {"at": now(), "model": args.model, "cond": COND, **backend.load()})

    def request(r):
        steps = r["steps"]
        prev = [tuple(p) for p in r["emitted"][:-1]]
        return build_finish("en", "ko", steps[-1]["src"], prev), prev

    def record(r, prev, res):
        last = r["steps"][-1]
        hyp = res["raw_output"].strip()
        step = {"j": last["j"], "src": last["src"], "raw_output": res["raw_output"], "hyp": hyp, "hold": "",
                "parse_fail": False, "hold_invalid": False, "final": True, "final_hold": "", "empty": not hyp,
                "forced": False, "finish": True, "input_tokens": res["input_tokens"],
                "output_tokens": res["output_tokens"], "truncated": res["truncated"],
                "latency_ms": res.get("latency_ms"), "emitted": True}
        append(out_path, {"at": now(), "model": args.model, "cond": COND, "idx": r["idx"], "en": r["en"],
                          "ref": r["ref"], "pieces": r["pieces"], "steps": r["steps"][:-1] + [step],
                          "emitted": [list(p) for p in prev] + [[last["src"], hyp]], "base_cond": BASE})

    if m["kind"] == "local":
        for k in range(0, len(base), args.batch):
            chunk = base[k:k + args.batch]
            reqs = [request(r) for r in chunk]
            for r, (_, prev), res in zip(chunk, reqs, backend.translate_batch([q for q, _ in reqs])):
                record(r, prev, res)
    else:
        from concurrent.futures import ThreadPoolExecutor

        def run(r):
            if ledger.over_budget():
                raise SystemExit(f"budget ${ledger.budget} reached — stopping")
            (system, user), prev = request(r)
            record(r, prev, backend.translate(system, user, tag={"model_name": args.model, "cond": COND,
                                                                 "idx": r["idx"]}))

        with ThreadPoolExecutor(max_workers=m.get("workers", 4)) as pool:
            for f in [pool.submit(run, r) for r in base]:
                f.result()
    print(f"[{args.model}] finished", flush=True)


if __name__ == "__main__":
    main()
