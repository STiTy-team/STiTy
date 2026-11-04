"""2.1 실행기. 모델 하나 × 조건 × 문장 398. 문장마다 조각을 차례로 번역하고, 낸 번역은 고치지 않는다.

한 문장이 끝나면 results/<run_id>/<model>/stream.jsonl 에 한 줄(조각별 단계 목록)을 붙인다. 다시 돌리면 끝난
문장을 건너뛴다. 도중에 죽은 문장은 처음부터 다시 번역한다 — 그 사이 API 비용은 api_usage.jsonl 에 이미 남아 있다.

P3 에서 빈 출력이 나오면 그 조각은 내보내지 않고 다음 조각 앞에 붙여 다시 보낸다. 마지막 조각까지 비면 P1 문면으로
한 번 더 부른다 (forced).

API 는 문장끼리 병렬로 보낸다. 로컬은 문장 --batch 개를 묶어 같은 순번의 조각을 한 번에 생성한다.

    PYTHONPATH=$PWD python -u evaluation/TranslatorPrompt/scripts/stream_run.py --model gpt-6-luna
"""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE.parent / "LongContextMT" / "scripts"))

from tp.stream_prompts import HOLD_CONDS, build, build_h3, covered, is_empty, parse_hold  # noqa: E402
from lcmt.backends import CostLedger, LocalBf16, LunaChat  # noqa: E402
from run_translate import read_env_key  # noqa: E402
from dctx.ledger import append, now, read_rows  # noqa: E402

SRC, TGT = "en", "ko"


class Sentence:
    """한 문장의 스트리밍 상태. step() 에 넣을 요청을 내고, 응답을 받아 다음 상태로 간다."""

    def __init__(self, cond: str, row: dict):
        self.cond, self.row = cond, row
        self.pieces = row["pieces"]
        self.j = 0                # 다음에 보낼 조각 순번
        self.carry = ""           # P3 빈 출력으로 미뤄 둔 원문
        self.prev: list[tuple[str, str]] = []
        self.steps: list[dict] = []
        self.forcing = False
        self.prev3: list[tuple[str, str, str]] = []   # H3: (받은 원문 전체, 보여 준 번역, 남긴 원문)

    @property
    def done(self) -> bool:
        return self.j >= len(self.pieces)

    def piece_src(self) -> str:
        return (self.carry + " " + self.pieces[self.j]).strip() if self.carry else self.pieces[self.j]

    def request(self) -> tuple[str, str]:
        if self.cond == "H3":
            return build_h3(SRC, TGT, self.carry, self.pieces[self.j], self.prev3,
                            last=self.j == len(self.pieces) - 1)
        return build("P1" if self.forcing else self.cond, SRC, TGT, self.piece_src(), self.prev,
                     last=self.j == len(self.pieces) - 1)

    def take(self, res: dict):
        if self.cond in HOLD_CONDS or self.cond == "H3":
            return self.take_hold(res)
        last = self.j == len(self.pieces) - 1
        src = self.piece_src()
        hyp = res["raw_output"].strip()
        empty = is_empty(hyp)
        step = {"j": self.j, "src": src, "raw_output": res["raw_output"], "hyp": "" if empty else hyp,
                "empty": empty, "forced": self.forcing, "final": last,
                "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
                "truncated": res["truncated"], "latency_ms": res.get("latency_ms")}
        self.steps.append(step)
        if empty and last and not self.forcing:
            self.forcing = True           # 같은 조각을 P1 로 다시
            return
        self.forcing = False
        if empty and not last:
            self.carry = src              # 내보내지 않고 다음 조각에 붙인다
        else:
            self.carry = ""
            self.prev.append((src, step["hyp"]))
            step["emitted"] = True
        self.j += 1

    def take_hold(self, res: dict):
        """H 조건: <out> 은 지금 내보내고, <hold> 원문은 다음 조각 앞에 붙인다. 마지막 조각의 hold 는 갈 곳이 없어
        번역되지 않은 채 남는다 (final_hold 로 센다)."""
        last = self.j == len(self.pieces) - 1
        src = self.piece_src()
        p = parse_hold(res["raw_output"], src)
        step = {"j": self.j, "src": src, "raw_output": res["raw_output"], "hyp": p["out"], "hold": p["hold"],
                "parse_fail": p["parse_fail"], "hold_invalid": p["hold_invalid"], "final": last,
                "final_hold": p["hold"] if last else "", "empty": not p["out"], "forced": False,
                "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
                "truncated": res["truncated"], "latency_ms": res.get("latency_ms"), "emitted": True}
        self.steps.append(step)
        self.carry = "" if last else p["hold"]
        self.prev.append((covered(src, p["hold"]), p["out"]))
        self.prev3.append((src, p["out"], p["hold"]))
        self.j += 1

    def record(self, model: str) -> dict:
        return {"at": now(), "model": model, "cond": self.cond, "idx": self.row["idx"],
                "en": self.row["en"], "ref": self.row["ko"], "pieces": self.pieces, "steps": self.steps,
                "emitted": [[s, t] for s, t in self.prev]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None, help="앞 몇 문장만 (스모크)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=16, help="로컬 전용. 문장 몇 개를 묶어 생성할지")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "configs" / "stream.yml").read_text(encoding="utf-8"))
    m = cfg["models"][args.model]
    run_dir = HERE / "results" / (args.run_id or cfg["run_id"])
    out_path = run_dir / args.model / "stream.jsonl"
    rows = read_rows(HERE / cfg["seg"])
    if args.limit:
        rows = rows[:args.limit]
    conds = args.conditions or cfg["conditions"]
    done = {(r["cond"], r["idx"]) for r in read_rows(out_path)}
    jobs = [(c, r) for c in conds for r in rows if (c, r["idx"]) not in done]
    print(f"[{args.model}] {len(done)} sentences done, {len(jobs)} to go", flush=True)

    if m["kind"] == "local":
        backend, ledger = LocalBf16(args.model, m["model"], m.get("max_new_tokens", 256)), None
    else:
        ledger = CostLedger(run_dir / "api_usage.jsonl", cfg["budget_usd"]["translation"])
        backend = LunaChat(args.model, m, read_env_key(m["key_env"], [str(HERE.parents[1] / e) for e in cfg["env_files"]]),
                           cfg["prices"][m["model"]], ledger, cfg["api_timeout_sec"])
    append(run_dir / args.model / "load.jsonl", {"at": now(), "model": args.model, "batch": args.batch,
                                                 **backend.load()})

    if m["kind"] == "local":
        # 같은 조건끼리만 묶는다 — 조건이 섞인 묶음은 패딩 길이가 달라져 조건 간 비교에 잡음이 된다.
        for c in conds:
            cj = [r for cc, r in jobs if cc == c]
            for k in range(0, len(cj), args.batch):
                group = [Sentence(c, r) for r in cj[k:k + args.batch]]
                while True:
                    live = [s for s in group if not s.done]
                    if not live:
                        break
                    for s, res in zip(live, backend.translate_batch([s.request() for s in live])):
                        s.take(res)
                for s in group:
                    append(out_path, s.record(args.model))
                print(f"[{args.model}] {c} {min(k + args.batch, len(cj))}/{len(cj)}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor

        def run(job):
            c, r = job
            s = Sentence(c, r)
            while not s.done:
                if ledger.over_budget():
                    raise SystemExit(f"budget ${ledger.budget} reached — stopping")
                system, user = s.request()
                s.take(backend.translate(system, user, tag={"model_name": args.model, "cond": c,
                                                            "idx": r["idx"], "j": s.j}))
            append(out_path, s.record(args.model))

        with ThreadPoolExecutor(max_workers=m.get("workers", 4)) as pool:
            for k, f in enumerate([pool.submit(run, j) for j in jobs]):
                f.result()
                if k % 100 == 0:
                    print(f"[{args.model}] {k}/{len(jobs)}, spent ${ledger.spent():.4f}", flush=True)
    print(f"[{args.model}] finished", flush=True)


if __name__ == "__main__":
    main()
