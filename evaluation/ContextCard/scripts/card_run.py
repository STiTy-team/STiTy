"""맥락 카드 실행기. 조건(K1~K4)마다 강연을 처음부터 차례로 번역하면서 8문장마다 카드를 고친다.

카드 추출은 실제로는 번역 경로 밖에서 비동기로 돈다. 여기서는 그 시간 흐름을 흉내 낸다 — 문장 i 까지 번역한 뒤
(i+1 이 8의 배수이면) 그 8문장의 원문과 **이 조건이 낸 번역**으로 카드를 고치고, 고친 카드는 lag 문장 뒤
(문장 i+1+lag)부터 쓴다. 그 사이 문장은 이전 카드를 쓴다.

결과: results/<run_id>/<model>/translations.jsonl (문장마다), cards.jsonl (카드 판마다), 비용은 api_usage.jsonl.
다시 돌리면 끝난 조건은 건너뛰고, 도중에 끊긴 조건은 기록에서 상태를 되살려 이어 간다.
비교군(1.1 의 N=0·8·16 체인)은 baseline_translations.jsonl 로 옮겨 둔다.

    PYTHONPATH=$PWD python -u evaluation/ContextCard/scripts/card_run.py
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE.parent / "LongContextMT" / "scripts"))

from card_prompts import (EMPTY_CARD, EXTRACT_SYSTEM, NOTES_SYSTEM, V2, WITH_REGISTER,  # noqa: E402
                          extract_system_v2, extract_user, parse_card, parse_card_v2, render, translate_user)
from lcmt.backends import CostLedger, LunaChat  # noqa: E402
from lcmt.prompt import clean  # noqa: E402
from run_translate import read_env_key  # noqa: E402
from dctx.ledger import append, now, read_rows  # noqa: E402


def copy_baseline(cfg: dict, out: Path, model: str):
    if out.exists():
        return
    rows = read_rows(HERE / cfg["baseline_run"] / model / "translations.jsonl")
    keep = [{"cond": f"N{r['n']}", "idx": r["idx"], "src": r["src"], "ref": r["ref"], "hyp": r["hyp"],
             "input_tokens": r["input_tokens"], "latency_ms": r["latency_ms"], "from": "lcmt"}
            for r in rows if r["n"] in cfg["baseline_n"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in sorted(keep, key=lambda r: (r["cond"], r["idx"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="앞 몇 문장만 (스모크)")
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "configs" / "card.yml").read_text(encoding="utf-8"))
    model = cfg["translator"]
    m = cfg["models"][model]
    run_dir = HERE / "results" / (args.run_id or cfg["run_id"])
    mdir = run_dir / model
    out_path, card_path = mdir / "translations.jsonl", mdir / "cards.jsonl"
    talk = read_rows(HERE / cfg["talk"])
    if args.limit:
        talk = talk[:args.limit]
    copy_baseline(cfg, mdir / "baseline_translations.jsonl", model)

    ledger = CostLedger(run_dir / "api_usage.jsonl", cfg["budget_usd"]["total"])
    key = read_env_key(m["key_env"], [str(HERE.parents[1] / e) for e in cfg["env_files"]])
    luna = LunaChat(model, m, key, cfg["prices"][m["model"]], ledger, cfg["api_timeout_sec"])
    append(mdir / "load.jsonl", {"at": now(), **luna.load(), "extractor": cfg["extractor"]})
    every, lag = cfg["update_every"], cfg["lag"]
    lock = threading.Lock()

    def chain(cond: str):
        done = sorted((r for r in read_rows(out_path) if r["cond"] == cond), key=lambda r: r["idx"])
        cards = [c for c in read_rows(card_path) if c["cond"] == cond]
        hyps = {r["idx"]: r["hyp"] for r in done}
        start = len(done)
        if start and [r["idx"] for r in done] != list(range(start)):
            raise SystemExit(f"{cond}: 기록된 문장 번호가 이어지지 않는다")
        # 카드 판: (쓰이기 시작하는 문장, 카드). 재개할 때는 기록된 판을 그대로 쓴다.
        versions = [(0, EMPTY_CARD, 0)] + [(c["usable_from"], c["card"], c["version"]) for c in cards]
        for i in range(start, len(talk)):
            if ledger.over_budget():
                raise SystemExit(f"budget ${ledger.budget} reached — stopping")
            _, card, ver = max((v for v in versions if v[0] <= i), key=lambda v: v[0])
            notes = render(cond, card)
            res = luna.translate(NOTES_SYSTEM, translate_user(talk[i]["en"], notes),
                                 tag={"model_name": model, "cond": cond, "idx": i, "kind": "translate"})
            hyp = clean(res["raw_output"])
            hyps[i] = hyp
            with lock:
                append(out_path, {"at": now(), "cond": cond, "idx": i, "src": talk[i]["en"], "ref": talk[i]["ko"],
                                  "hyp": hyp, "raw_output": res["raw_output"], "multi_line": "\n" in hyp,
                                  "card_version": ver, "notes_chars": len(notes),
                                  "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
                                  "latency_ms": res["latency_ms"], "ttft_ms": res["ttft_ms"]})
            if (i + 1) % every == 0 and not any(c["made_after"] == i for c in cards):
                prev = max(versions, key=lambda v: v[0])[1]
                rows = [(talk[j]["en"], hyps[j]) for j in range(i + 1 - every, i + 1)]
                shown = {**prev, "terms": [{"en": t["en"], "ko": t["ko"]} for t in prev["terms"]]}
                system = extract_system_v2(cond in WITH_REGISTER) if cond in V2 else EXTRACT_SYSTEM
                r = luna.translate(system, extract_user(shown, rows),
                                   tag={"model_name": model, "cond": cond, "idx": i, "kind": "extract"})
                keep = ("genre", "speaker", "audience") + (("register",) if cond in WITH_REGISTER else ())
                new, ok = (parse_card_v2(r["raw_output"], prev, [en for en, _ in rows], i, keep) if cond in V2
                           else parse_card(r["raw_output"], prev))
                rec = {"at": now(), "cond": cond, "version": len(versions), "made_after": i,
                       "usable_from": i + 1 + lag, "parsed": ok, "card": new, "raw_output": r["raw_output"],
                       "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                       "latency_ms": r["latency_ms"]}
                with lock:
                    append(card_path, rec)
                cards.append(rec)
                versions.append((rec["usable_from"], new, rec["version"]))
        print(f"[{cond}] finished {len(talk)} sentences, {len(versions) - 1} cards", flush=True)

    conds = args.conditions or cfg["conditions"]
    with ThreadPoolExecutor(max_workers=len(conds)) as pool:
        for f in [pool.submit(chain, c) for c in conds]:
            f.result()
    print(f"all finished, spent ${ledger.spent():.4f}", flush=True)


if __name__ == "__main__":
    main()
