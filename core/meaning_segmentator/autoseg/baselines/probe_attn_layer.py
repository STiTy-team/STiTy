"""AlignAtt 이 쓸 교차어텐션 층을 고른다 — 층마다 정렬 단조성을 재서 최고를 쓴다.

AlignAtt 의 판정은 "이 타깃 토큰이 소스 어디를 보는가" 하나에 달려 있다. 그 답이 층마다
다르고, **마지막 층은 `</s>` 로 쏠려(attention sink) 쓸 수 없다.** 그래서 모델을 바꾸면
층을 다시 골라야 한다 — nllb-600M 은 12층, madlad-3B 는 32층이라 번호를 옮겨 쓸 수 없다.

지표는 **단조성**이다: 타깃 토큰 순서대로 정렬된 소스 어절 인덱스가 비감소인 비율. 번역은
대체로 왼쪽에서 오른쪽으로 진행하므로, 정렬이 신호를 담고 있으면 이 값이 높다. 어순이
크게 바뀌는 쌍(en→ja)은 원래 낮으므로 **층 간 상대 비교로만 읽는다**. 곁가지로 `</s>` 나
언어 표지로 쏠린 비율(sink)도 찍는다 — 그 값이 크면 그 층은 정렬을 안 담고 있다.

    PYTHONPATH=. python -m core.meaning_segmentator.autoseg.baselines.probe_attn_layer \\
        --targets de ja zh --n 50

산출은 표준출력뿐이다 (값 하나를 고르는 1회성 측정이라 산출물을 남기지 않는다).
"""
from __future__ import annotations

import argparse
import json

from .datasets import get as get_dataset
from .nmt import MODEL, Nmt


def monotonicity(pairs: list[tuple[int, int]]) -> tuple[float, float]:
    """(비감소 비율, sink 비율). `pairs` 는 (토큰 id, 정렬된 어절 인덱스)."""
    ws = [w for _tok, w in pairs]
    sink = sum(1 for w in ws if w < 0) / len(ws) if ws else 1.0
    ok = [w for w in ws if w >= 0]
    if len(ok) < 2:
        return 0.0, sink
    inc = sum(1 for a, b in zip(ok, ok[1:]) if b >= a)
    return inc / (len(ok) - 1), sink


def main() -> int:
    ap = argparse.ArgumentParser(description="AlignAtt 교차어텐션 층 스윕")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--targets", nargs="+", default=["de", "ja", "zh"])
    ap.add_argument("--dataset", default="covost2")
    ap.add_argument("--manifest-tag", default="full")
    ap.add_argument("--n", type=int, default=50, help="문장 수")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="잴 층 (기본: 마지막 층을 뺀 전체를 균등하게 8개)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    best: dict[str, list[tuple[float, int]]] = {}
    for tgt in a.targets:
        ents = get_dataset(a.dataset, src="en").entries(a.manifest_tag, tgt)
        texts = [e.src for e in list(ents.values())[: a.n]]
        nmt = Nmt(src="en", tgt=tgt, device=a.device, model_name=a.model, attentions=True)
        n_layers = nmt.model.config.num_layers if hasattr(nmt.model.config, "num_layers") \
            else nmt.model.config.decoder_layers
        # 마지막 층은 sink 라 후보에서 뺀다.
        layers = a.layers or sorted({round(i * (n_layers - 2) / 7) for i in range(8)})
        print(f"[{tgt}] {a.model} / 층 {n_layers}개 / 후보 {layers} / 문장 {len(texts)}",
              flush=True)
        rows = []
        for layer in layers:
            nmt.attn_layer = layer
            mono, sink = [], []
            for t in texts:
                m, s = monotonicity(nmt.emit_with_alignment(t))
                mono.append(m)
                sink.append(s)
            mono_m = sum(mono) / len(mono)
            rows.append((mono_m, layer))
            print(f"  층 {layer:3d}  단조성 {mono_m:.4f}  sink {sum(sink)/len(sink):.4f}",
                  flush=True)
        best[tgt] = sorted(rows, reverse=True)
        del nmt

    print("\n합산 (타깃 평균):")
    per_layer: dict[int, list[float]] = {}
    for tgt, rows in best.items():
        for mono, layer in rows:
            per_layer.setdefault(layer, []).append(mono)
    ranked = sorted(((sum(v) / len(v), k) for k, v in per_layer.items()), reverse=True)
    for mono, layer in ranked:
        print(f"  층 {layer:3d}  {mono:.4f}  "
              + " ".join(f"{t}={dict((l, m) for m, l in best[t])[layer]:.3f}"
                         for t in a.targets))
    print(f"\n→ 최고 층 {ranked[0][1]} ({ranked[0][0]:.4f}). "
          f"`nmt.Nmt` 의 기본값을 이 값으로 둘 것")
    print(json.dumps({"model": a.model, "best_layer": ranked[0][1],
                      "by_target": {t: [[round(m, 4), l] for m, l in best[t]]
                                    for t in a.targets}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
