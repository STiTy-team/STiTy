#!/usr/bin/env python3
"""CoVoST2 전체 평가셋의 **그리디 오라클** 절단을 `prompt_eval` 형식으로 낸다.

그리디 오라클은 경계별 라벨의 상위 k 개를 자르는 기준선이다 (`LOOP_JUDGE.md`). 라벨은
[`runtime/labels.py`](../../autoseg/runtime/labels.py) 의

    label_j = (1 − contra_j) × (adq_l_j + adq_r_j) / 2          (타깃 평균)
    contra_j = P_NLI(모순 | premise = 원문 전체, hypothesis = units[:j])

이고 `--contra-source source` 가 기본이다 — 번역 NLI 는 타깃 간 순위상관이 0.21 로
라벨 성분 중 가장 타깃 종속적이었다 (`PAPER_JUDGE.md` §3).

**왜 따로 스크립트인가.** `gates/oracle_label.py` 는 정책 열댓 개를 만들고 전부
`score_split` 으로 자기채점한다 — 213문장 dev 에서는 맞는 설계지만 15,430문장에서는
조각 번역이 수백만 건이 된다. 여기서는 정책 하나를 절단까지만 내고, 채점은 gold 참조를
쓰는 `scoring/bleu_eval` + `baselines/comet_score` 에 넘긴다.

**CoVoST2 런 디렉토리는 루프 런이 아니다.** `config.json`·`data/` 가 없으므로 문장은
매니페스트에서 직접 읽는다.

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.tools.covost2_oracle.emit_oracle_greedy \
        --manifest-tag full --targets de ja zh --t-grid 2 3 4 6 --min-gap 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.meaning_segmentator.autoseg.baselines import datasets as DS
from core.meaning_segmentator.autoseg.loop import target_is_spaced
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import hset as H
from core.meaning_segmentator.autoseg.runtime import labels as L
from core.meaning_segmentator.autoseg.runtime import metrics
from core.meaning_segmentator.autoseg.runtime.pipeline import (
    JsonCache, LocalTranslator, split_segments, to_lang_code, truncate)

TGT_NAME = {"de": "German", "ja": "Japanese", "zh": "Chinese",
            "es": "Spanish", "ko": "Korean", "en": "English"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="covost2")
    ap.add_argument("--manifest-tag", default="full")
    ap.add_argument("--src", default="en")
    ap.add_argument("--targets", nargs="+", default=["de", "ja", "zh"],
                    help="라벨의 타깃 평균에 들어갈 언어들 (짧은 코드)")
    ap.add_argument("--run-id", default="en2x/covost2/full_oracle_greedy",
                    help="산출 런 디렉토리. **원 런과 달라야 한다** — `bleu_eval` 은 조건을 "
                         "`auto_T*` 로 이름 붙여 `bleu/<tgt>.json` 에 쓰므로 같은 곳에 쓰면 "
                         "기존 제안 분절 값을 덮는다")
    ap.add_argument("--label", default="oracle_greedy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--t-grid", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--min-gap", type=int, default=1)
    ap.add_argument("--min-duration", type=float, default=1.0,
                    help="이 길이(초) 미만 발화를 뺀다. 기본 1.0 은 `full` 표(n=15,430)와 "
                         "같은 문장 집합을 쓰기 위한 값이다 — 매니페스트는 "
                         "`--min-duration 0` 으로 지어져 100개 더 들어 있다")
    ap.add_argument("--limit", type=int, default=0, help="스모크용 문장 수 상한")
    ap.add_argument("--mt-model", default="google/madlad400-3b-mt")
    ap.add_argument("--mt-batch", type=int, default=48)
    ap.add_argument("--comet-batch", type=int, default=64)
    ap.add_argument("--adequacy-backend", default="cometkiwi")
    ap.add_argument("--contra-source", default="source", choices=["source", "translation"])
    ap.add_argument("--cache-flush-every", type=int, default=20000,
                    help="번역 캐시를 몇 건마다 파일에 쓸지. 이 규모에서 기본 20 은 "
                         "파일 전체 재작성이 번역보다 오래 걸린다")
    ap.add_argument("--labels-only", action="store_true",
                    help="라벨만 만들고 끝낸다 (절단·emit 생략)")
    a = ap.parse_args()

    spec = DS.get(a.dataset, **({"src": a.src} if a.dataset == "covost2" else {}))
    ents = spec.entries(a.manifest_tag, a.targets[0])
    keep = [(k, e) for k, e in ents.items()
            if e.dur_ms is None or e.dur_ms >= a.min_duration * 1000]
    n_short = len(ents) - len(keep)
    if a.limit:
        keep = keep[: a.limit]
    ids = [k for k, _ in keep]
    texts = [e.src for _, e in keep]
    spaced = True                      # 소스는 영어
    tgt_names = [TGT_NAME[t] for t in a.targets]

    n_units = [len(t.split()) for t in texts]
    log(f"{a.dataset}/{a.manifest_tag} {a.split}: 문장 {len(ids)}"
        f" (매니페스트 {len(ents)}, {a.min_duration}s 미만 제외 {n_short})")
    log(f"  어절 평균 {sum(n_units) / len(n_units):.2f} 최대 {max(n_units)} "
        f"경계 후보 {sum(x - 1 for x in n_units)} / 타깃 {tgt_names}")

    run_dir = RUNS_DIR / a.run_id
    (run_dir / "cache").mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt_eval").mkdir(parents=True, exist_ok=True)
    prof = run_dir / "measured_profile.json"
    if not prof.exists():
        prof.write_text(json.dumps({"uses_spaces_between_words": spaced}), encoding="utf-8")

    adequacy = metrics.make_adequacy_backend(a.adequacy_backend, batch_size=a.comet_batch)
    contradiction = metrics.make_contradiction_backend()
    # 번역기를 여기서 만들어 캐시 flush 주기를 넘긴다. `compute_labels` 가 자기가 만들면
    # 기본 20 이라 25만 건 규모에서 디스크가 병목이 된다.
    translators = {
        name: LocalTranslator(
            tgt_code=to_lang_code(name),
            cache=JsonCache(run_dir / "cache" / f"translate_{to_lang_code(name)}.json",
                            flush_every=a.cache_flush_every),
            model_id=a.mt_model, batch=a.mt_batch)
        for name in tgt_names}

    lab = L.compute_labels(run_dir, a.split, ids, texts, tgt_names, spaced,
                           adequacy, contradiction, a.mt_model, target_is_spaced,
                           log=log, translators=translators,
                           contra_source=a.contra_source)
    for tr in translators.values():
        tr.cache.flush()
    log(f"라벨 완료 — contra 출처 {L.contra_source_of(lab)}")
    if a.labels_only:
        return 0

    rows = []
    for i, (sid, text) in enumerate(zip(ids, texts)):
        u = text.split()
        scores = {j: L.label_value(lab, i, j) for j in range(1, len(u))}
        seg = H.seg_with(u, {j: int(round(v * H.SCALE)) for j, v in scores.items()}, spaced)
        by_T = {}
        for T in a.t_grid:
            cut, miss = truncate(seg, T, spaced, a.min_gap)
            pieces = split_segments(cut) or [text]
            by_T[str(T)] = {"seg_text": cut, "k": len(pieces),
                            "missing_boundaries": miss, "pieces_src": pieces}
        rows.append({"id": sid, "text": text, "seg_text": seg, "valid": True,
                     "full_trans": None, "by_T": by_T})

    out = run_dir / "prompt_eval" / f"{a.label}_{a.split}.json"
    out.write_text(json.dumps({
        "prompt_file": f"oracle:greedy:{a.contra_source}_contra",
        "source": f"{a.dataset}/{a.manifest_tag} min_duration={a.min_duration}",
        "split": a.split, "t_grid": a.t_grid, "min_gap": a.min_gap,
        "src_spaced": spaced, "tag_convention": "score",
        "label": "(1-contra)*(adq_l+adq_r)/2, 타깃 평균",
        "label_targets": tgt_names, "mt_model": a.mt_model,
        "label_source": str(run_dir / f"oracle_labels_{a.split}.json"),
        "rows": rows}, ensure_ascii=False), encoding="utf-8")
    ks = {str(T): round(sum(r["by_T"][str(T)]["k"] for r in rows) / len(rows), 3)
          for T in a.t_grid}
    log(f"emit -> {out} ({len(rows)}문장), 평균 조각 수 {ks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
