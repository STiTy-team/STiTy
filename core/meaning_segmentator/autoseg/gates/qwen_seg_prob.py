"""ASR 파인튜닝 모델의 `<SEG>` 확률을 텍스트 전용 분절 점수로 쓸 수 있는가.

질문: `<SEG>` 를 내도록 학습한 Qwen3-ASR 의 디코더(Qwen3 1.7B)에 **오디오 없이 텍스트만**
넣고, 단어 경계마다 다음 토큰이 `<SEG>` 일 확률을 읽으면 그게 절단 점수로 일하는가.
새로 학습하지 않는다. 생성도 안 한다 — 문장은 입력으로 고정하고 확률만 읽는다.

점수 (경계 j = `units[:j]` 뒤, 1..n−1):

    p0     log P(<SEG> | units[:j]).  앞만 본다 — 텍스트 스트리밍에서 바로 쓸 수 있는 값.
           학습 형식이 `word <SEG> next` 라 `' '` 다음 `<SEG>` 경로와 공백 없는 `<SEG>`
           경로를 합친다.
    lrK    p0 + [log P(다음 K 어절 | `<SEG>` 삽입) − log P(다음 K 어절 | 삽입 없음)].
           K 어절 뒤에 판정하는 셈이라 지연이 K 어절 는다. `lrall` 은 문장 끝까지.

비교 대상은 judge13 최종표와 **같은 자**다: run27 test 200문장, k=1..99 (평균 조각 ≥ 2),
min_gap 1, 같은 `H_set`(CometKiwi × (1 − max 소스 contra)), 같은 번역 캐시.
오라클을 다시 재서 judge13 표의 0.6596 이 나오는지로 채점 경로가 같은지 확인한다.

LLM 호출 0 (`--prompt-run` 을 주면 그 런의 분절 캐시만 읽는다 — 적중하면 0원이고, 지출은
`metrics.json` 에 남는다).

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.autoseg.gates.qwen_seg_prob \
        --from-run en2x/en-multi/run27 --run-id en2x/en-multi/qwenseg01 --split test \
        --prompt-run en2x/en-multi/judge13
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
import time
import types
from pathlib import Path

from ..paths import RUNS_DIR

REPO = Path(__file__).resolve().parents[4]
DEFAULT_MODEL = REPO / "models/Qwen3-ASR-1.7B-en-covost2-dailytalk-mix-c200-merged"
LOOKAHEAD = (1, 2, 3)
PUNCT = re.compile(r"[,.;:!?)\"'”’]$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 1단계: Qwen3-ASR 디코더로 경계 점수 ─────────────────────────────────

def load_thinker(model_dir: Path):
    """`qwen_asr/__init__` 은 추론 모듈(nagisa 등)을 끌어오므로 건너뛰고 모델 정의만 읽는다."""
    import torch
    pkg = types.ModuleType("qwen_asr")
    pkg.__path__ = [str(REPO / "Qwen3-ASR/qwen_asr")]
    sys.modules.setdefault("qwen_asr", pkg)
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
    thinker = model.thinker.to("cuda").eval()
    return tok, thinker


def prompt_ids(tok, language: str) -> list[int]:
    """`Qwen3ASRModel._build_text_prompt` 와 같은 틀에서 오디오 자리만 비운다."""
    s = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_end|><|im_end|>\n"
         f"<|im_start|>assistant\nlanguage {language}<asr_text>")
    return tok(s, add_special_tokens=False)["input_ids"]


def word_tokens(tok, units: list[str]) -> list[list[int]]:
    return [tok(w if i == 0 else " " + w, add_special_tokens=False)["input_ids"]
            for i, w in enumerate(units)]


def score_sentences(tok, thinker, texts: list[str], language: str, batch_size: int) -> list[dict]:
    """문장마다 `{p0: [...], lr1: [...], ..., lrall: [...]}` — 길이 n−1, 인덱스 j−1."""
    import torch
    seg_id = tok.convert_tokens_to_ids("<SEG>")
    sp_id = tok.convert_tokens_to_ids("Ġ")
    assert seg_id != tok.unk_token_id and tok.decode([sp_id]) == " ", (seg_id, sp_id)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    P = prompt_ids(tok, language)

    # 문장별 기준열 1개 + 경계별 삽입열 n−1 개. 전부 한 줄로 세워 배치로 돈다.
    jobs = []            # (문장, j or 0, ids)
    meta = []
    for si, t in enumerate(texts):
        units = t.split()
        wt = word_tokens(tok, units)
        ends, pos = [], len(P)
        for w in wt:
            pos += len(w)
            ends.append(pos - 1)          # 기준열에서 어절 w 의 마지막 토큰 자리
        flat = [x for w in wt for x in w]
        meta.append({"wt": wt, "ends": ends})
        jobs.append((si, 0, P + flat))
        for j in range(1, len(units)):
            left = [x for w in wt[:j] for x in w]
            right = [x for w in wt[j:] for x in w]
            jobs.append((si, j, P + left + [sp_id, seg_id] + right))

    lp_of: dict[tuple[int, int], torch.Tensor] = {}       # (문장, j) → 토큰별 log P(실제 다음 토큰)
    direct: dict[tuple[int, int], float] = {}              # (문장, j) → 기준열에서 log P(<SEG>)
    order = sorted(range(len(jobs)), key=lambda k: len(jobs[k][2]))
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(order), batch_size):
            batch = [jobs[k] for k in order[b:b + batch_size]]
            L = max(len(x[2]) for x in batch)
            ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for r, (_si, _j, x) in enumerate(batch):
                ids[r, :len(x)] = torch.tensor(x)
                att[r, :len(x)] = 1
            ids, att = ids.cuda(), att.cuda()
            thinker.rope_deltas = None
            logits = thinker(input_ids=ids, attention_mask=att, use_cache=False).logits.float()
            logp = torch.log_softmax(logits, dim=-1)
            nxt = logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)   # 자리 p → 토큰 p+1
            for r, (si, j, x) in enumerate(batch):
                lp_of[(si, j)] = nxt[r, :len(x) - 1].cpu()
                if j == 0:
                    for jj, e in enumerate(meta[si]["ends"][:-1], start=1):
                        direct[(si, jj)] = float(logp[r, e, seg_id])
            if (b // batch_size) % 50 == 0:
                log(f"[qwen] {b + len(batch)}/{len(jobs)} 열 {time.time() - t0:.0f}s")

    out = []
    for si, m in enumerate(meta):
        wt, ends = m["wt"], m["ends"]
        n = len(wt)
        base = lp_of[(si, 0)]
        rec = {"p0": [], "lrall": [], **{f"lr{K}": [] for K in LOOKAHEAD}}
        for j in range(1, n):
            v = lp_of[(si, j)]
            e = ends[j - 1]                      # 기준열·삽입열 공통: 어절 j−1 의 마지막 토큰 자리
            lp_sp, lp_seg = float(v[e]), float(v[e + 1])   # 자리 e → ' ', 자리 e+1 → <SEG>
            p0 = float(torch.logaddexp(torch.tensor(direct[(si, j)]), torch.tensor(lp_sp + lp_seg)))
            rec["p0"].append(round(p0, 4))
            # 다음 어절 토큰들의 log P. 기준열에서 어절 j 는 자리 e 다음부터, 삽입열은 2칸 뒤.
            def cont(K):
                ntok = sum(len(w) for w in wt[j:j + K])
                return float(v[e + 2:e + 2 + ntok].sum() - base[e:e + ntok].sum())
            for K in LOOKAHEAD:
                rec[f"lr{K}"].append(round(p0 + cont(K), 4))
            rec["lrall"].append(round(p0 + cont(n - j), 4))
        out.append(rec)
    log(f"[qwen] {len(texts)}문장 / {len(jobs)}열 {time.time() - t0:.0f}s")
    return out


# ── 2단계: 절단집합 → H_set ─────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-run", required=True, help="분할·라벨·번역 캐시를 읽을 런 (judge13 은 run27)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--language", default="English")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-gap", type=int, default=1)
    p.add_argument("--min-chunk", type=int, default=2)
    p.add_argument("--max-k", type=int, default=99)
    p.add_argument("--comet-batch-size", type=int, default=32)
    p.add_argument("--prompt-run", default=None,
                   help="이 런의 best_prompt/prompt_v0 을 분절 캐시로 되살려 짝 비교한다 (judge13)")
    p.add_argument("--prompt-budget", type=float, default=0.5,
                   help="분절 캐시가 빗나갈 때 쓸 수 있는 상한 (USD). 적중하면 0")
    a = p.parse_args()

    src = RUNS_DIR / a.from_run
    run_dir = RUNS_DIR / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "cache").exists():
        (run_dir / "cache").symlink_to(Path("..") / src.name / "cache")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    assert cfg["spaced"], "어절 단위(공백) 소스만 다룬다"
    targets = cfg["targets"]
    rows = json.loads((src / f"data/{a.split}.json").read_text(encoding="utf-8"))
    lab = json.loads((src / f"oracle_labels_{a.split}.json").read_text(encoding="utf-8"))
    texts = [r["text"] for r in rows]
    log(f"[data] {a.from_run}/{a.split} {len(texts)}문장 / 타깃 {targets} / min_gap {a.min_gap} "
        f"/ k 1~{a.max_k} (평균 조각 ≥ {a.min_chunk})")

    # 1단계 — 캐시가 있으면 모델을 안 올린다
    tag = Path(a.model).name
    score_path = run_dir / f"qwen_scores_{a.split}_{tag}.json"
    if score_path.exists():
        qs = json.loads(score_path.read_text(encoding="utf-8"))
        log(f"[qwen] 캐시 {score_path.name}")
    else:
        import torch
        tok, thinker = load_thinker(Path(a.model))
        qs = score_sentences(tok, thinker, texts, a.language, a.batch_size)
        score_path.write_text(json.dumps(qs), encoding="utf-8")
        del thinker
        torch.cuda.empty_cache()
    for i in range(3):
        u = texts[i].split()
        top = sorted(range(1, len(u)), key=lambda j: -qs[i]["p0"][j - 1])[:4]
        log(f"[qwen] 예시 p0 상위 4: " + " ".join(
            w + (f" <SEG:{qs[i]['p0'][j]:.1f}>" if j + 1 in top else "") for j, w in enumerate(u)))

    # 2단계
    from ..loop import target_is_spaced
    from ..loop_judge import _sets_from_scores, by_latency, oracle_sets
    from ..runtime import data, hset, metrics
    from ..runtime import labels as L
    from ..runtime.pipeline import JsonCache, LocalTranslator, to_lang_code

    sents = [data.Sentence(**r) for r in rows]
    translators = {t: LocalTranslator(tgt_code=to_lang_code(t),
                                      cache=JsonCache(run_dir / "cache" / f"translate_{to_lang_code(t)}.json"))
                   for t in targets}
    scorer = hset.HsetScorer(translators=translators,
                             qe=metrics.make_adequacy_backend(cfg.get("adequacy_backend", "cometkiwi"),
                                                              batch_size=a.comet_batch_size),
                             spaced=True, target_spaced={t: target_is_spaced(t) for t in targets})

    def hset_of(sets):
        keys = sorted(sets)
        vals = scorer.score(texts, [(k[0], sets[k]) for k in keys],
                            lambda i, j: lab[targets[0]][i]["contra"][j - 1])
        for tr in translators.values():
            if tr.cache is not None:
                tr.cache.flush()
        return dict(zip(keys, vals))

    def sets_of(score_of):
        return _sets_from_scores(sents, score_of, True, a.min_gap, a.min_chunk, a.max_k)

    rng = random.Random(20260919)
    rand = {i: {j: rng.random() for j in range(1, len(s.text.split()))} for i, s in enumerate(sents)}
    policies = {
        "oracle": oracle_sets(lab, sents, True, a.min_gap, a.min_chunk, a.max_k),
        "random": sets_of(lambda i, s, u: rand[i]),
        # 구두점 뒤 1점, 아니면 0 — 동점은 앞쪽 우선(top_k_cuts 규칙)이라 구두점을 다 쓰면 앞에서부터 채운다
        "punct": sets_of(lambda i, s, u: {j: float(bool(PUNCT.search(u[j - 1]))) for j in range(1, len(u))}),
    }
    for name in ["p0", *(f"lr{K}" for K in LOOKAHEAD), "lrall"]:
        policies[f"qwen_{name}"] = sets_of(lambda i, s, u, n=name: {j: qs[i][n][j - 1] for j in range(1, len(u))})

    h: dict[str, dict] = {}
    for name, sets in policies.items():
        t0 = time.time()
        h[name] = hset_of(sets)
        log(f"[{name}] H_set {st.mean(h[name].values()):.4f} ({len(h[name])} 짝) "
            f"{by_latency(sents, h[name], True)} / {time.time() - t0:.0f}s")

    if a.prompt_run:
        from ..infra.gateway import Gateway
        from ..loop_distill import evaluate
        from ..loop_judge import policy_sets
        prun = RUNS_DIR / a.prompt_run
        gw = Gateway(provider="openai", model="gpt-5-mini", budget=a.prompt_budget,
                     reasoning_effort="medium", max_connections=16)
        seg_cache = JsonCache(prun / "cache" / "segment.json")
        for name, fname in (("judge_best", "best_prompt.txt"), ("judge_v0", "prompt_v0.txt")):
            pr = (prun / fname).read_text(encoding="utf-8")
            rws, _m = evaluate(gw, pr, sents, lab, True, a.min_gap, cfg["t_grid"], seg_cache,
                               16, 6, "medium", k_samples=3)
            u = gw.usage.snapshot()
            (run_dir / "metrics.json").write_text(json.dumps({"usage": u, "run_total_cost": u["cost"]},
                                                              indent=1), encoding="utf-8")
            log(f"[{name}] 분절 캐시 재생 — 호출 {u['calls']} / 누적 ${u['cost']:.4f}")
            h[name] = hset_of(policy_sets(rws, sents, True, a.min_gap, a.min_chunk, a.max_k))
            log(f"[{name}] H_set {st.mean(h[name].values()):.4f} ({len(h[name])} 짝) "
                f"{by_latency(sents, h[name], True)}")
        gw.close()

    # 짝 비교 — 문장 단위 클러스터 부트스트랩
    refs = [r for r in ("judge_best", "judge_v0", "punct", "random") if r in h]
    table = {}
    for name in h:
        table[name] = {"mean": round(st.mean(h[name].values()), 4), "n": len(h[name]),
                       "by_bin": by_latency(sents, h[name], True)}
        for r in refs:
            if r == name:
                continue
            kk = sorted(set(h[name]) & set(h[r]))
            b = hset.paired_bootstrap([h[name][k] for k in kk], [h[r][k] for k in kk],
                                      clusters=[i for i, _k in kk])
            table[name][f"vs_{r}"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in b.items()}
    (run_dir / f"report_{a.split}_{tag}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
    lines = [f"| 정책 | H_set | " + " | ".join(f"vs {r}" for r in refs) + " |",
             "|---|---|" + "---|" * len(refs)]
    for name, d in table.items():
        cells = [f"{d[f'vs_{r}']['mean']:+.4f} [{d[f'vs_{r}']['lo']:+.4f}, {d[f'vs_{r}']['hi']:+.4f}]"
                 if f"vs_{r}" in d else "—" for r in refs]
        lines.append(f"| {name} | {d['mean']:.4f} | " + " | ".join(cells) + " |")
    log("[표]\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
