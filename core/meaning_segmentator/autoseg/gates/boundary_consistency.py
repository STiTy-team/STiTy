"""경계 단위 `consistency` — 이어붙인 번역이 전체 번역과 같은 말인가.

`contradiction` 과 `adequacy` 는 경계마다 값이 나온다. 옛 `consistency` 는 합본 전체에
대한 값이라 경계 라벨 자리에 못 꽂혔는데, **경계를 하나만 쓰면 조각이 둘뿐**이므로
경계 단위로도 잘 정의된다.

    cons_j = min( ent(전체번역 ⇒ MT(앞) ⊕ MT(뒤)),  ent(MT(앞) ⊕ MT(뒤) ⇒ 전체번역) )

두 방향을 다 재는 이유는 함의가 비대칭이기 때문이다 — 전체⇒합본만 보면 누락이
통과하고(약한 명제는 함의된다), 합본⇒전체만 보면 환각이 통과한다. `min` 이라 어느 쪽
실패든 잡히고, 방향을 따로 저장해 두면 실패 유형이 나뉜다.

참조가 자기 offline 번역이라 COMET 으로 재면 어순을 단조화한 좋은 분절이 감점된다
(`benign_paraphrase` 0.8414 < `negation_flip` 0.8843). NLI 는 명제만 보므로 그 편향이
없다 — 이것이 옛 `Q`(COMET) 대신 양방향 NLI 를 쓰는 이유다.

**번역 비용 0.** `oracle_labels_*.json` 을 만들 때 경계마다 앞조각·뒤조각을 이미 다
번역해 런 캐시에 넣었다. 여기서는 그 캐시를 그대로 읽어 이어붙이고 NLI 만 돌린다.

    PYTHONPATH=. python -m core.meaning_segmentator.autoseg.gates.boundary_consistency \
        --run-id en2x/en-multi/run15 --split test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..runtime import metrics
from ..runtime.pipeline import JsonCache, LocalTranslator, to_lang_code
from ..loop import DEFAULT_TARGET_POOL, resolve_targets
from ..paths import RUNS_DIR


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class BidirectionalEntailment(metrics._NliBase):
    """양방향 함의. 채택되면 `runtime/metrics.py` 로 옮긴다 — 지금은 후보라 게이트에 둔다."""

    def score_pairs(self, refs: list[str], hyps: list[str]) -> tuple[list[float], list[float]]:
        """`(ent(ref ⇒ hyp), ent(hyp ⇒ ref))`. 같은 쌍은 한 번만 잰다."""
        n = len(refs)
        fwd = [0.0] * n
        bwd = [0.0] * n
        uniq: dict[tuple[str, str], list[int]] = {}
        for i, (r, h) in enumerate(zip(refs, hyps)):
            if r.strip() and h.strip():
                uniq.setdefault((r, h), []).append(i)
        if not uniq:
            return fwd, bwd
        keys = list(uniq)
        items = []
        for r, h in keys:
            items.append({"text": r, "text_pair": h})
            items.append({"text": h, "text_pair": r})
        res = self.load()(items, batch_size=self.batch_size)
        for k, key in enumerate(keys):
            f = self._prob(res[2 * k], "entail")
            b = self._prob(res[2 * k + 1], "entail")
            for i in uniq[key]:
                fwd[i], bwd[i] = f, b
        return fwd, bwd


def _units(text: str, spaced: bool) -> list[str]:
    return text.split() if spaced else list(text.replace(" ", ""))


def _join(units: list[str], spaced: bool) -> str:
    return (" " if spaced else "").join(units)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=0, help="문장 수 상한 (스모크)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run_id
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    measured = json.loads((run_dir / "measured_profile.json").read_text(encoding="utf-8"))
    spaced = bool(measured.get("uses_spaces_between_words", True))
    targets = args.targets or resolve_targets(cfg.get("tgt_langs") or DEFAULT_TARGET_POOL,
                                              cfg["src_lang"])
    sents = json.loads((run_dir / "data" / f"{args.split}.json").read_text(encoding="utf-8"))
    if args.n:
        sents = sents[: args.n]
    ids = [s["id"] for s in sents]
    texts = [s["text"] for s in sents]
    units = [_units(t, spaced) for t in texts]
    n_bnd = sum(max(0, len(u) - 1) for u in units)
    log(f"{args.run_id} {args.split}: 문장 {len(sents)} / 경계 {n_bnd} / 타깃 {targets}")

    out_path = run_dir / f"oracle_consistency_{args.split}.json"
    cached = (json.loads(out_path.read_text(encoding="utf-8"))
              if out_path.exists() else {})
    nli = BidirectionalEntailment(batch_size=args.batch_size)

    for tgt in targets:
        if tgt in cached and [x["id"] for x in cached[tgt]] == ids:
            log(f"[{tgt}] 캐시 사용")
            continue
        code = to_lang_code(tgt)
        tr = LocalTranslator(tgt_code=code,
                             cache=JsonCache(run_dir / "cache" / f"translate_{code}.json"),
                             model_id=cfg.get("local_mt_model", "google/madlad400-3b-mt"))
        t0 = time.time()
        full = tr.full(texts)
        pre_src, suf_src, owner = [], [], []
        for i, u in enumerate(units):
            for j in range(1, len(u)):
                pre_src.append(_join(u[:j], spaced))
                suf_src.append(_join(u[j:], spaced))
                owner.append(i)
        pre_tr = tr.full(pre_src)
        suf_tr = tr.full(suf_src)
        log(f"[{tgt}] 번역 확보 {time.time() - t0:.0f}s (캐시 적중이면 즉시) / NLI {len(owner) * 2} 쌍")
        # `LocalTranslator.seg_batch` 가 조각을 이어붙일 때 쓰는 것과 같은 규칙: 공백 하나.
        concat = [f"{p} {s}".strip() for p, s in zip(pre_tr, suf_tr)]
        fwd, bwd = nli.score_pairs([full[i] for i in owner], concat)
        per = [{"id": ids[i], "cons": [], "ent_fwd": [], "ent_bwd": []}
               for i in range(len(texts))]
        for k, i in enumerate(owner):
            d = per[i]
            d["ent_fwd"].append(round(fwd[k], 4))
            d["ent_bwd"].append(round(bwd[k], 4))
            d["cons"].append(round(min(fwd[k], bwd[k]), 4))
        cached[tgt] = per
        out_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        log(f"[{tgt}] 완료 {time.time() - t0:.0f}s -> {out_path.name}")
    log(f"끝 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
