"""문장 단위 채점에서 불이익이 없는 정답 번역을 gpt-6-sol 로 만든다 → data/talk231_ref_sol.jsonl

IWSLT 정답은 자막 번역이라 의역·누락·이웃 줄로 내용이 밀린 경우가 있다. 문장 단위로 채점하면 이런
정답 때문에 맞는 번역도 감점된다. 여기서는 원문 한 줄에 정답 한 줄이 정확히 대응하도록 다시 번역한다.

- 매 요청에 강연 원문 398줄 전체를 번호와 함께 준다(요청마다 같은 앞부분이라 입력 캐시가 걸린다).
- 대상 줄은 idx 240~397. 채점은 256 부터지만, 묶음 채점(Doc-COMET)의 문맥으로 앞 줄 정답이 필요해 앞을 더 만든다.
- 대상 줄을 20줄 안팎으로 나눠 차례로 요청하고, 앞 요청에서 만든 번역을 다음 요청에 함께 줘 용어·말투를 잇는다.
- 출력은 {"idx": 번역} JSON. 번호가 다 있고 비어 있지 않으며 한글인지 검사하고, 틀리면 그 묶음만 다시 요청한다.
- 호출마다 usage·비용을 results/<run>/api_usage_ref.jsonl 에 바로 붙인다. 예산을 넘으면 멈춘다.

    PYTHONPATH=$PWD python -u evaluation/LongContextMT/scripts/build_ref_sol.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_translate import HERE, load_config, read_env_key  # noqa: E402
from lcmt.backends import CostLedger, OpenAIChat  # noqa: E402
from dctx.ledger import append, now, read_rows  # noqa: E402

MODEL = "gpt-6-sol"
PRICE = (2.00, 0.20, 10.00)  # USD per 1M: input, cached input, output (DialogueContext 설정과 같음)
FIRST_IDX = 240
WINDOWS = 8

SYSTEM = """You are an expert English-to-Korean translator producing REFERENCE translations for \
evaluating machine translation of a TED talk. Each reference will be compared to a system \
translation of the same single line, so the line-to-line correspondence must be exact.

Rules:
1. One numbered English line -> one Korean translation. Translate ALL content of that line and \
NOTHING from any other line. Never move content to a neighbouring line, never merge or split lines.
2. Some lines start or end in the middle of a sentence (the talk was cut into subtitle lines). \
Translate such a fragment as a fragment; do not complete it with words from neighbouring lines.
3. Keep the meaning faithful and complete: no omissions, no additions, no summarising. Keep \
hesitations, asides and dashes when they carry meaning.
4. Write natural spoken Korean as a professional interpreter would say it to a live audience, \
using one consistent polite register (합니다/해요 style mixed naturally as in Korean talks is fine, \
but no 반말). Keep names, terms and the speaker's voice consistent across the whole talk.
5. Use the full talk only to understand meaning, references and terminology.

Output ONLY a JSON object mapping each requested line number (as a string) to its Korean \
translation. No other text."""


def build_user(talk: list[dict], targets: list[int], done: dict[int, str]) -> str:
    lines = "\n".join(f"[{r['idx']}] {r['en']}" for r in talk)
    parts = ["FULL TALK (English, numbered lines):", lines, ""]
    if done:
        prev = "\n".join(f"[{i}] {done[i]}" for i in sorted(done))
        parts += ["Your Korean reference translations so far (keep terms and register consistent):",
                  prev, ""]
    parts.append(f"Translate lines {targets[0]} to {targets[-1]} (inclusive), "
                 f"{len(targets)} lines. Return a JSON object with exactly these keys: "
                 + ", ".join(f'"{i}"' for i in targets))
    return "\n".join(parts)


def parse(raw: str, targets: list[int]) -> tuple[dict[int, str] | None, str]:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, "no JSON object"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"JSON error: {e}"
    keys = {str(i) for i in targets}
    if set(obj) != keys:
        return None, f"keys mismatch: missing {sorted(keys - set(obj))}, extra {sorted(set(obj) - keys)}"
    out = {}
    for i in targets:
        v = obj[str(i)]
        if not isinstance(v, str) or not v.strip():
            return None, f"empty line {i}"
        hangul = len(re.findall(r"[가-힣]", v))
        letters = len(re.findall(r"[A-Za-z가-힣]", v))
        if letters and hangul / letters < 0.5:
            return None, f"line {i} not Korean: {v[:60]!r}"
        out[i] = v.strip()
    return out, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--attempts", type=int, default=3, help="검사 실패 시 묶음당 요청 횟수")
    args = ap.parse_args()

    cfg = load_config()
    run_dir = HERE / "results" / cfg["run_id"]
    talk = read_rows(HERE / cfg["talk"])
    out_path = HERE / "data" / "talk231_ref_sol.jsonl"
    raw_path = run_dir / "ref_sol_raw.jsonl"
    ledger = CostLedger(run_dir / "api_usage_ref.jsonl", args.budget)
    chat = OpenAIChat(MODEL, MODEL, read_env_key("OPENAI_API_KEY_TRANS", cfg["env_files"]), PRICE, ledger,
                      reasoning_effort=args.reasoning_effort, temperature=None,
                      max_completion_tokens=32000, timeout=600, max_retries=5)

    done = {r["idx"]: r["ko_sol"] for r in read_rows(out_path)}
    idxs = list(range(FIRST_IDX, len(talk)))
    size = -(-len(idxs) // WINDOWS)
    windows = [idxs[k:k + size] for k in range(0, len(idxs), size)]
    print(f"{len(idxs)} lines in {len(windows)} windows, {len(done)} already done", flush=True)

    for w in windows:
        if all(i in done for i in w):
            continue
        for attempt in range(1, args.attempts + 1):
            if ledger.over_budget():
                raise SystemExit(f"budget ${args.budget} reached — stopping (spent ${ledger.spent():.4f})")
            tag = {"phase": "ref_sol", "window": f"{w[0]}-{w[-1]}", "attempt": attempt}
            res = chat.translate(SYSTEM, build_user(talk, w, {i: done[i] for i in done if i < w[0]}), {}, tag=tag)
            got, status = parse(res["raw_output"], w)
            append(raw_path, {"at": now(), **tag, "status": status, "raw_output": res["raw_output"],
                              "api_meta": res["api_meta"], "input_tokens": res["input_tokens"],
                              "output_tokens": res["output_tokens"], "latency_ms": res["latency_ms"]})
            print(f"window {w[0]}-{w[-1]} attempt {attempt}: {status}, "
                  f"in {res['input_tokens']} out {res['output_tokens']} "
                  f"(reasoning {res['api_meta']['reasoning_tokens']}), spent ${ledger.spent():.4f}", flush=True)
            if got:
                for i in w:
                    done[i] = got[i]
                    append(out_path, {"idx": i, "en": talk[i]["en"], "ko_iwslt": talk[i]["ko"],
                                      "ko_sol": got[i], "model": MODEL,
                                      "reasoning_effort": args.reasoning_effort})
                break
        else:
            raise SystemExit(f"window {w[0]}-{w[-1]} failed {args.attempts} times")
    print(f"finished: {len(done)} lines, spent ${ledger.spent():.4f}", flush=True)


if __name__ == "__main__":
    main()
