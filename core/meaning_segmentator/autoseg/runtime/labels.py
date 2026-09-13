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


def contra_source_of(labels: dict[str, list[dict]]) -> str:
    """이 라벨의 `contra` 가 무엇으로 잰 것인가 — "translation"(기본) 또는 "source"."""
    for per in labels.values():
        for d in per:
            return d.get("contra_source", "translation")
    return "translation"


def apply_source_contra(labels: dict[str, list[dict]], texts: list[str], spaced: bool,
                        contradiction, log=print) -> dict[str, list[dict]]:
    """`contra`/`ent` 를 **소스 NLI** 로 덮어쓴다 — premise = 원문 전체, hypothesis = `units[:j]`.

    번역 NLI contra 는 타깃 간 순위상관 0.21 로 라벨 성분 중 가장 타깃 종속적이었다 (dev,
    2026-09-13). 원문끼리 재면 번역이 안 끼므로 타깃이 무엇이든 같은 값이고, 세 타깃 평균과
    +0.42 로 어느 단일 타깃보다 잘 맞는다. gold(test 100 COMET)에서 현행 대비 손해 없음
    (격자 평균 0.7642→0.7685, T=4 +0.013 유의). `run23/gold_test.md`.

    adq 는 건드리지 않는다 — 조각 번역 품질은 번역 없이는 정의가 안 된다. 번역 contra 는
    `contra_mt`/`ent_mt` 로 남기고 `contra_source: "source"` 를 찍어 재사용 시 모드를 가른다.
    이미 소스 모드면 그대로 돌려준다 (멱등).
    """
    if contra_source_of(labels) == "source":
        return labels
    units = [units_of(t, spaced) for t in texts]
    prem, hyp, owner = [], [], []
    for i, u in enumerate(units):
        for j in range(1, len(u)):
            prem.append(texts[i]); hyp.append(join_units(u[:j], spaced)); owner.append(i)
    t0 = time.time()
    contra, one_minus_ent = contradiction.score_dual(prem, hyp)
    per_sent: dict[int, tuple[list, list]] = {}
    for k, i in enumerate(owner):
        per_sent.setdefault(i, ([], []))
        per_sent[i][0].append(round(contra[k], 4)); per_sent[i][1].append(round(one_minus_ent[k], 4))
    out: dict[str, list[dict]] = {}
    for tgt, per in labels.items():
        rows = []
        for i, d in enumerate(per):
            c, e = per_sent.get(i, ([], []))
            assert len(c) == len(d["contra"]), f"{tgt} {d.get('id')}: 경계 수 불일치 {len(c)} vs {len(d['contra'])}"
            rows.append({**d, "contra": c, "ent": e, "contra_mt": d["contra"], "ent_mt": d["ent"],
                         "contra_source": "source"})
        out[tgt] = rows
    log(f"[labels] 소스 NLI contra {len(prem)}쌍 {time.time() - t0:.0f}s (번역 contra 는 contra_mt 로 보관)")
    return out


def units_of(text: str, spaced: bool) -> list[str]:
    return text.split() if spaced else list(text.replace(" ", ""))


def join_units(units: list[str], spaced: bool) -> str:
    return (" " if spaced else "").join(units)


def compute_labels(run_dir: Path, split: str, ids: list[str], texts: list[str],
                   targets: list[str], spaced: bool, adequacy, contradiction,
                   mt_model: str, target_is_spaced, log=print,
                   translators: dict | None = None,
                   reuse_from: Path | None = None,
                   contra_source: str = "translation") -> dict[str, list[dict]]:
    """`{타깃: [문장별 {id, contra[], ent[], contra_floor[], adq_l[], adq_r[], hyp_units[]}]}`.

    `reuse_from` 은 같은 분할(같은 id 순서)의 라벨을 가진 다른 런 디렉토리다 — 시드·분할이
    같은 런끼리는 라벨이 동일하므로 그대로 복사한다.

    `contra_source="source"` 면 번역 라벨(adq)을 그대로 두고 `contra` 만 소스 NLI 로 덮어쓴다
    (`apply_source_contra`). 재사용한 파일이 이미 소스 모드면 그대로, 번역 모드면 변환한다.
    반대 방향(소스 모드 파일을 번역 모드로 재사용)은 `contra_mt` 를 되돌린다.
    """
    label_path = run_dir / f"oracle_labels_{split}.json"
    cached: dict = {}
    for src in (label_path, (reuse_from / f"oracle_labels_{split}.json") if reuse_from else None):
        if src and src.exists():
            blob = json.loads(src.read_text(encoding="utf-8"))
            # **이 출처가 실제로 채운 것만 적는다.** 종전에는 누적된 `cached` 전체를
            # 찍어서, 앞 출처(런 자기 파일)가 다 채웠는데도 뒤 출처(`reuse_from`)가
            # 준 것처럼 나왔다. run17 재시작에서 train 200문장 라벨이 자기 파일에서
            # 왔는데 로그는 run15 (90문장, 겹침 30)에서 재사용했다고 적었다 —
            # 그대로 읽으면 분할이 run15 와 같다고 오독하게 된다.
            got = [tgt for tgt, per in blob.items()
                   if tgt in targets and [x["id"] for x in per] == ids and tgt not in cached]
            for tgt in got:
                cached[tgt] = blob[tgt]
            if got:
                where = "런 산출물" if src == label_path else str(src)
                log(f"[labels/{split}] {where} 에서 {sorted(got)} 재사용")
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
    if contra_source == "source":
        out = apply_source_contra(out, texts, spaced, contradiction, log=log)
        cached = out
        label_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    elif contra_source_of(out) == "source":
        # 소스 모드 파일을 번역 모드 런이 재사용하는 경우 — 보관해 둔 번역 contra 로 되돌린다.
        out = {tgt: [{**d, "contra": d["contra_mt"], "ent": d["ent_mt"], "contra_source": "translation"}
                     for d in per] for tgt, per in out.items()}
        cached = out
        label_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        log(f"[labels/{split}] 재사용한 라벨이 소스 모드라 번역 contra(contra_mt)로 되돌림")
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
