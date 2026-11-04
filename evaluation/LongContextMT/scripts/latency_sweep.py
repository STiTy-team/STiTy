"""1.2 지연 스윕. 문장을 고정하고 문맥 길이(N)만 바꿔 같은 요청을 반복한다.

1.1 은 N 마다 체인이 달라 입력이 조건마다 다르다. 여기서는 모든 N 이 같은 번역 로그
(n=context_chain 조건 체인의 출력)에서 앞 N 개를 잘라 쓰므로, 지연 차이가 문맥 길이에서만 나온다.
문장·반복마다 N 순서를 섞는다. 결과는 latency_sweep.jsonl 에 호출마다 붙인다.

    PYTHONPATH=$PWD python -u evaluation/LongContextMT/scripts/latency_sweep.py --model qwen3.5-4b-bf16
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_translate import HERE, WARMUP, load_config, make_backend  # noqa: E402
from lcmt.backends import LocalBf16  # noqa: E402
from lcmt.prompt import SYSTEM, user_message  # noqa: E402
from dctx.ledger import append, now, read_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg = load_config()
    sw = cfg["latency_sweep"]
    run_dir = HERE / "results" / (args.run_id or cfg["run_id"])
    out_path = run_dir / args.model / "latency_sweep.jsonl"
    talk = read_rows(HERE / cfg["talk"])
    chain = {r["idx"]: r["hyp"] for r in read_rows(run_dir / args.model / "translations.jsonl")
             if r["n"] == sw["context_chain"]}
    if len(chain) != len(talk):
        raise SystemExit(f"n={sw['context_chain']} chain incomplete: {len(chain)}/{len(talk)}")
    targets = [r for r in talk[-sw["targets"]:]]
    if targets[0]["idx"] < max(sw["n_grid"]):
        raise SystemExit("first target has fewer preceding sentences than max N")

    done = {(r["rep"], r["idx"], r["n"]) for r in read_rows(out_path)}
    backend, ledger = make_backend(args.model, cfg, run_dir)
    append(run_dir / args.model / "load.jsonl", {"at": now(), "model": args.model,
                                                  "phase": "latency_sweep", **backend.load()})
    if isinstance(backend, LocalBf16):
        for _ in range(WARMUP):
            backend.translate(SYSTEM, user_message(talk[0]["en"], []))

    for rep in range(sw["repeats"]):
        for row in targets:
            i = row["idx"]
            order = list(sw["n_grid"])
            random.Random(cfg["seed"] * 7919 + rep * 1000 + i).shuffle(order)
            for n in order:
                if (rep, i, n) in done:
                    continue
                if ledger and ledger.over_budget():
                    raise SystemExit(f"budget ${ledger.budget} reached — stopping")
                prev = [chain[j] for j in range(i - n, i)] if n else []
                tag = {"model_name": args.model, "phase": "latency_sweep", "n": n, "idx": i, "rep": rep}
                res = backend.translate(SYSTEM, user_message(row["en"], prev), tag=tag)
                append(out_path, {"at": now(), "model": args.model, "rep": rep, "idx": i, "n": n,
                                  "context_chars": sum(len(p) for p in prev), **res})
        print(f"[{args.model}] sweep rep {rep} done", flush=True)
    print(f"[{args.model}] sweep finished", flush=True)


if __name__ == "__main__":
    main()
