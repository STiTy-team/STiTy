"""1.1 번역 실행기. 모델 하나를 받아 강연 전체를 N 조건마다 처음부터 차례로 번역한다.

- 문맥 = 같은 N 조건에서 이 모델이 바로 앞 N 문장에 낸 번역. 조건마다 번역 로그(체인)가 따로 있다.
- 문장 i 를 N 조건 전부에 대해 번역한 다음 i+1 로 넘어간다. 조건 순서는 문장마다 섞는다
  — 조건별로 시간대를 몰아 돌리면 GPU 를 같이 쓰는 다른 작업이 특정 N 의 지연에만 섞인다.
- 결과는 호출마다 translations.jsonl 에 바로 붙인다. 다시 돌리면 끝난 (N, idx) 를 건너뛰고
  체인을 파일에서 복원해 이어 간다.
- ``--context src_tgt`` 는 문맥에 앞 문장의 영어 원문을 번역과 쌍으로 넣는다. 결과는 ``<run_id>-src_tgt``
  에 따로 쌓인다. ``--n-grid`` 로 N 조건을 고를 수 있다. N=0 도 system 문면이 문맥 형태마다
  한 구절 다르므로, 비교 기준선은 같은 형태로 따로 돌린다.

    PYTHONPATH=$PWD python -u evaluation/LongContextMT/scripts/run_translate.py --model gpt-6-luna
    PYTHONPATH=$PWD python -u evaluation/LongContextMT/scripts/run_translate.py --model qwen3.5-4b-bf16 \
        --context src_tgt --n-grid 1 2 4 16
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lcmt.backends import CostLedger, LocalBf16, LunaChat  # noqa: E402
from lcmt.prompt import CONTEXT_MODES, SYSTEMS, clean, user_message  # noqa: E402
from dctx.ledger import append, now, read_rows  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
WARMUP = 3


def load_config():
    return yaml.safe_load((HERE / "configs" / "experiment.yml").read_text(encoding="utf-8"))


def read_env_key(name: str, env_files: list[str]) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for env in map(Path, env_files):
        if not env.exists():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    raise SystemExit(f"{name} not found in environment or {env_files}")


def make_backend(name: str, cfg: dict, run_dir: Path):
    m = cfg["models"][name]
    if m["kind"] == "local":
        return LocalBf16(name, m["model"], m.get("max_new_tokens", 256)), None
    ledger = CostLedger(run_dir / "api_usage.jsonl", cfg["budget_usd"]["translation"])
    return LunaChat(name, m, read_env_key(m["key_env"], cfg["env_files"]), cfg["prices"][m["model"]], ledger,
                    cfg["api_timeout_sec"]), ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None, help="앞 문장 몇 개만 (스모크)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--parallel-chains", action="store_true",
                    help="API 전용. N 체인을 동시에 돌린다")
    ap.add_argument("--workers", type=int, default=4,
                    help="--parallel-chains 동시 체인 수. 10 이면 TPM 20만 한도에 걸렸다(429)")
    ap.add_argument("--context", choices=CONTEXT_MODES, default="tgt",
                    help="문맥 형태. tgt = 앞 번역만, src_tgt = 앞 원문과 번역을 쌍으로")
    ap.add_argument("--n-grid", type=int, nargs="+", default=None, help="N 조건 (기본: 설정 파일의 n_grid)")
    args = ap.parse_args()

    cfg = load_config()
    default_run = cfg["run_id"] if args.context == "tgt" else f"{cfg['run_id']}-{args.context}"
    run_dir = HERE / "results" / (args.run_id or default_run)
    system = SYSTEMS[args.context]
    out_dir = run_dir / args.model
    out_path = out_dir / "translations.jsonl"
    talk = read_rows(HERE / cfg["talk"])
    if args.limit:
        talk = talk[:args.limit]
    grid = list(args.n_grid or cfg["n_grid"])

    done = {}
    for r in read_rows(out_path):
        done[(r["n"], r["idx"])] = r
    chains = {n: {} for n in grid}
    for (n, i), r in done.items():
        if n in chains:
            chains[n][i] = r["hyp"]

    backend, ledger = make_backend(args.model, cfg, run_dir)
    info = backend.load()
    append(out_dir / "load.jsonl", {"at": now(), "model": args.model, **info})
    print(f"[{args.model}] loaded {info}", flush=True)

    todo = sum(1 for i in range(len(talk)) for n in grid if (n, i) not in done)
    print(f"[{args.model}] {len(done)} done, {todo} to go", flush=True)
    if todo and isinstance(backend, LocalBf16):
        for _ in range(WARMUP):
            backend.translate(system, user_message(talk[0]["en"], []))

    def translate_one(row, n):
        i = row["idx"]
        if ledger and ledger.over_budget():
            raise SystemExit(f"budget ${ledger.budget} reached — stopping")
        span = range(max(0, i - n), i) if n else range(0)
        prev = [chains[n][j] for j in span]
        prev_src = [talk[j]["en"] for j in span]
        tag = {"model_name": args.model, "n": n, "idx": i}
        res = backend.translate(system, user_message(row["en"], prev, prev_src, args.context), tag=tag)
        hyp = clean(res["raw_output"])
        chains[n][i] = hyp
        append(out_path, {
            "at": now(), "model": args.model, "context_mode": args.context, "n": n, "idx": i,
            "context_sentences": len(prev),
            "context_chars": sum(map(len, prev)) + (sum(map(len, prev_src)) if args.context == "src_tgt" else 0), "src": row["en"], "ref": row["ko"],
            "hyp": hyp, "multi_line": "\n" in hyp, "parallel_chains": args.parallel_chains, **res})

    started = time.time()
    if args.parallel_chains:
        # 체인(N)끼리는 서로의 번역을 안 쓰므로 체인끼리 동시에 돌린다. 체인 안에서는 문장 순서를 지킨다.
        # 긴 체인(큰 N)이 가장 늦게 끝나므로 먼저 넣는다.
        # 동시 요청이 API 지연에 섞이므로 이 모드의 1.1 지연은 참고용이다 — 지연 결론은 1.2 스윕으로 낸다.
        if isinstance(backend, LocalBf16):
            raise SystemExit("--parallel-chains is for API backends only")
        from concurrent.futures import ThreadPoolExecutor

        def run_chain(n):
            for row in talk:
                if (n, row["idx"]) not in done:
                    translate_one(row, n)
            print(f"[{args.model}] chain n={n} finished ({time.time() - started:.0f}s)", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for f in [pool.submit(run_chain, n) for n in sorted(grid, reverse=True)]:
                f.result()
    else:
        calls = 0
        for row in talk:
            i = row["idx"]
            order = grid[:]
            random.Random(cfg["seed"] * 1000 + i).shuffle(order)
            for n in order:
                if (n, i) not in done:
                    translate_one(row, n)
                    calls += 1
            if i % 10 == 0 or i == len(talk) - 1:
                rate = calls / max(1e-9, time.time() - started)
                spent = f", spent ${ledger.spent():.4f}" if ledger else ""
                print(f"[{args.model}] idx {i}/{len(talk) - 1}, {calls} calls this run, "
                      f"{rate:.2f} calls/s{spent}", flush=True)
    print(f"[{args.model}] finished", flush=True)


if __name__ == "__main__":
    main()
