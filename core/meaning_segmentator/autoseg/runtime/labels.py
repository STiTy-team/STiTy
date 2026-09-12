"""위치별 실측 라벨 — 절단 점수의 정답.

문장의 모든 단위 경계 j(1..n−1)에 대해, 분할과 무관한 값을 잰다.

    contra_j   NLI(premise = full 번역, hypothesis = `units[:j]` 를 한 덩어리로 번역한 것)
               의 모순 확률. 앞에 경계가 하나도 없는 셈이라 앞 조각의 오역이 끼어들 통로가
               없다 — 루프의 `pieces_contra`(누적 방출분)와 다른 점이다.
    ent_j      같은 호출의 1 − 함의 확률 (추가 비용 0)
    adq_l_j    CometKiwi(`units[:j]`, 그 번역)   왼쪽 조각이 혼자 번역이 되는가
    adq_r_j    CometKiwi(`units[j:]`, 그 번역)   오른쪽 조각이 혼자 번역이 되는가

기본 점수 `label = (1 − contra) × (adq_l + adq_r) / 2` (타깃 평균). gold 검증(test 100,
FLEURS 참조 COMET)에서 entail 변형 셋이 이 조합을 넘지 못했다 — `gates/oracle_label.py`.

라벨은 프롬프트와 무관하므로 **런당 분할별 1회**만 만들고 `oracle_labels_<split>.json`
에 캐시한다. LLM 호출 0, 로컬 MT·NLI·QE 만 돈다.
"""
from __future__ import annotations

import json
import statistics as st
import time
from pathlib import Path

from . import metrics
from .pipeline import JsonCache, LocalTranslator, to_lang_code
from ..gates import noise_floor

LABEL_KEYS = ("contra", "ent", "contra_floor", "adq_l", "adq_r", "hyp_units")


def units_of(text: str, spaced: bool) -> list[str]:
    return text.split() if spaced else list(text.replace(" ", ""))


def join_units(units: list[str], spaced: bool) -> str:
    return (" " if spaced else "").join(units)


def compute_labels(run_dir: Path, split: str, ids: list[str], texts: list[str],
                   targets: list[str], spaced: bool, adequacy, contradiction,
                   mt_model: str, target_is_spaced, log=print,
                   translators: dict | None = None,
                   reuse_from: Path | None = None) -> dict[str, list[dict]]:
    """`{타깃: [문장별 {id, contra[], ent[], contra_floor[], adq_l[], adq_r[], hyp_units[]}]}`.

    `reuse_from` 은 같은 분할(같은 id 순서)의 라벨을 가진 다른 런 디렉토리다 — 시드·분할이
    같은 런끼리는 라벨이 동일하므로 그대로 복사한다.
    """
    label_path = run_dir / f"oracle_labels_{split}.json"
    cached: dict = {}
    for src in (label_path, (reuse_from / f"oracle_labels_{split}.json") if reuse_from else None):
        if src and src.exists():
            blob = json.loads(src.read_text(encoding="utf-8"))
            for tgt, per in blob.items():
                if tgt in targets and [x["id"] for x in per] == ids and tgt not in cached:
                    cached[tgt] = per
            if src != label_path and cached:
                log(f"[labels/{split}] {src} 에서 {sorted(cached)} 재사용")
    out: dict[str, list[dict]] = {}
    units = [units_of(t, spaced) for t in texts]
    translators = translators if translators is not None else {}
    for tgt in targets:
        if tgt in cached:
            out[tgt] = cached[tgt]
            continue
        code = to_lang_code(tgt)
        tsp = target_is_spaced(tgt)
        tr = translators.get(tgt)
        if tr is None:
            tr = LocalTranslator(tgt_code=code,
                                 cache=JsonCache(run_dir / "cache" / f"translate_{code}.json"),
                                 model_id=mt_model)
            translators[tgt] = tr
        t0 = time.time()
        full = tr.full(texts)
        pre_src, suf_src, owner = [], [], []
        for i, u in enumerate(units):
            for j in range(1, len(u)):
                pre_src.append(join_units(u[:j], spaced))
                suf_src.append(join_units(u[j:], spaced))
                owner.append(i)
        log(f"[labels/{split}] {tgt}: full {len(texts)} / prefix {len(pre_src)} 번역")
        pre_tr = tr.full(pre_src)
        suf_tr = tr.full(suf_src)
        contra, one_minus_ent = contradiction.score_dual([full[i] for i in owner], pre_tr)
        adq_l = adequacy.score(pre_src, pre_tr)
        adq_r = adequacy.score(suf_src, suf_tr)
        # 길이별 잡음 바닥 (보정값은 참고용으로만 싣는다 — gold 검증에서 보정이 이득이 없었다)
        from ..loop import load_contra_floor
        floor_fn = load_contra_floor(run_dir, [{"full_trans": f} for f in full], contradiction,
                                     filename=f"contra_floor_{code}.json", tgt_spaced=tsp)
        per = [{"id": ids[i], **{k: [] for k in LABEL_KEYS}} for i in range(len(texts))]
        for k, i in enumerate(owner):
            hl = len(noise_floor.split_units(pre_tr[k], tsp))
            c0 = floor_fn(hl) if floor_fn else 0.0
            d = per[i]
            d["contra"].append(round(contra[k], 4))
            d["ent"].append(round(one_minus_ent[k], 4))
            d["contra_floor"].append(round(max(0.0, contra[k] - c0), 4))
            d["adq_l"].append(round(adq_l[k], 4))
            d["adq_r"].append(round(adq_r[k], 4))
            d["hyp_units"].append(hl)
        out[tgt] = per
        cached[tgt] = per
        label_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        log(f"[labels/{split}] {tgt}: 완료 {time.time() - t0:.0f}s -> {label_path.name}")
    if not label_path.exists() or set(json.loads(label_path.read_text(encoding="utf-8"))) != set(cached):
        label_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    return out


def label_value(labels: dict[str, list[dict]], i: int, j: int) -> float:
    """위치 j(1..n−1) 의 점수 라벨 = (1 − contra) × (adq_l + adq_r)/2, 타깃 평균."""
    vals = []
    for per in labels.values():
        d = per[i]
        vals.append((1 - d["contra"][j - 1]) * (d["adq_l"][j - 1] + d["adq_r"][j - 1]) / 2)
    return st.mean(vals)


def label_parts(labels: dict[str, list[dict]], i: int, j: int) -> dict:
    """Critic 에게 넘길 분해값 (타깃 평균)."""
    def m(k):
        return round(st.mean(per[i][k][j - 1] for per in labels.values()), 3)
    return {"contra": m("contra"), "adq_left": m("adq_l"), "adq_right": m("adq_r"),
            "label": round(label_value(labels, i, j), 3)}
