#!/usr/bin/env python3
"""ACL 60/60 장문 — 커밋 경계의 **소스 NLI contra**. 분절 위치가 얼마나 위험한지 잰다.

    .venv/bin/python evaluation/ast/contra_acl6060.py --tag 20260914_141234 --langs de ja zh

    contra_j = NLI(premise = gold 영어 문장,  hypothesis = 그 문장의 경계 j 까지 조각)

autoseg 의 `contra_source='source'` 와 같은 정의다(`core/meaning_segmentator/autoseg/`
[PAPER_JUDGE.md](../../core/meaning_segmentator/autoseg/PAPER_JUDGE.md) §2). **번역을 안 본다.**
참조 번역도, 교차언어 NLI 도 필요 없다 — premise·hypothesis 가 둘 다 영어다.

BLEU·COMET 가 못 보는 것을 본다. 채점은 mwerSegmenter 로 조각을 문장에 봉합한 뒤에
이뤄지므로, 어디서 끊었는지가 점수에서 지워진다. 실측(de): 문장이 커밋 하나에 통째로
들어온 경우 COMET 0.783, 잘린 경우 0.624 인데 BLEU 차이는 1.6 이하다.

**모델은 `.venv` 로만 돈다.** 기본 miniforge python 의 transformers 5.2.0 은
`vicgalle/xlm-roberta-large-xnli-anli` 의 토크나이저를 못 읽는다
(`ValueError: tiktoken is required to read a tiktoken file`). `.venv` 는 4.57.6 이다.

절단 위치는 **소스 쪽 텍스트 정렬**로 찾는다
--------------------------------------------
커밋의 ASR 원문을 gold 영어 전사에 difflib 로 붙여 문장 안의 절단 위치를 gold 어절
인덱스로 얻는다(실측 매핑률 97~100%). 타이밍(`audio_end_sec`)은 청크 격자에 양자화돼
있어 못 쓴다 — 문장 끝에서 커밋하는 punct 축조차 0.5초 허용으로 블록 중앙값이 9문장,
최대 1,161어절까지 뭉쳤다. 텍스트 정렬로 바꾸면 같은 축이 중앙값 1문장이 된다.

같은 이유로 **contra 는 타깃 언어와 무관하다.** static 축은 시간으로만 끊으므로 de/ja/zh
세 런에서 값이 완전히 같게 나온다(실측 확인). 표에는 정책당 한 값으로 싣는다.

문장 끝 가까이의 절단은 따로 센다 (`--edge-words`)
--------------------------------------------------
NLI 는 **명사구 한복판, 특히 문장 끝 직전에서 잘린 조각에 모순 확률을 높게 준다.** 의미가
뒤집혀서가 아니다. de 실측:

    절단 뒤 남은 어절   1: 0.195   2: 0.135   3-4: 0.076   5-9: 0.081   10+: 0.080

autoseg 는 `min_gap` 으로 절단 후보를 문장 양끝에서 떼어 놓아 이 구간을 아예 안 만든다
(`runtime/hset.py: top_k_cuts`). 여기서는 실제 시스템이 낸 경계라 뺄 수 없으므로, 끝
`--edge-words` 어절 이내 절단을 **분류해서 따로 보고**한다. 빼고 보면 punct-c1 이
0.1135 → 0.0776 으로 내려간다.

`contra` 와 `1 − entailment` 는 이 데이터에서 상관 r = 0.975 다. 둘 중 하나만 실으면 된다.

잡음 바닥을 뺀다 — 안 빼면 값의 대부분이 바닥이다
------------------------------------------------
NLI 는 무해한 미완성에도 0 이 아닌 모순 확률을 준다. **그 크기가 hypothesis 길이에 따라
달라진다.** 이 코퍼스(gold 영어 문장 416개의 모든 접두사)에서 잰 바닥:

    접두사 어절   1-2: 0.019   3-4: 0.044   5-6: 0.079   7-9: 0.119   10-14: 0.161   15+: 0.187

전체 평균 0.112 로, 우리가 보고하는 raw 값(0.06~0.13)과 같은 자릿수다. 그래서 기본으로
`max(0, contra − c0(길이))` 를 쓴다(`--no-floor` 로 끌 수 있다). autoseg 의
`gates/noise_floor.measure_floor` 를 그대로 부르고 결과는 `contra_floor_en.json` 에 캐시한다.

바닥을 안 빼면 접두사가 긴 축이 구조적으로 불리해진다. 실측: 평균 접두사 길이가 punct-c1
만 14.6어절(나머지는 11.1~11.9)이라 raw A 0.1213 → 보정 후 0.0726 으로 내려간다.

**자기 자신 쌍도 0 이 아니다.** premise = hypothesis = 같은 문장인데 평균 0.053, 최대 0.985
가 나온다(문장이 자기 자신과 모순일 수는 없다). 문장 안에 대비되는 두 항이 병렬로 놓이면
모델이 두 사본을 교차 정렬해 "같은 틀, 다른 값"으로 읽는 것으로 보인다 — 합성 확인:

    "The left side is a commit message."                                    0.002
    "The left side is a commit message and the right side is the release notes."  0.939
    "We reuse the results and then get the other results."                  0.018
    "We reuse the results from the second step and then get the results of the fourth step."  0.755

수량 표현이 있고 16어절 이상인 문장의 자기쌍 평균은 0.149, 둘 다 아니면 0.024 다. 그래서
**문장 전체를 hypothesis 로 넣는 쌍은 만들지 않는다** — 값이 없는데 잡음만 평균에 섞인다.
문장을 넘나든 커밋을 gold 경계에서 잘라 그 쌍을 추가하면 경계를 많이 가로지르는 축일수록
평균이 내려간다(실측: static-c6 −0.0364, seg-c1 −0.0057 로 격차가 반 토막).

집계 셋 — B 가 주지표다
-----------------------
    B  문장당 기대위험    Σcontra ÷ **전체 문장**. 무분절 문장이 0 을 기여한다. **주지표**
    A  경계 평균          절단이 1개 이상인 문장만. "끊을 때 얼마나 위험하게 끊나"
    C  문장당 위험절단    Σ1[contra > 0.5] ÷ 전체 문장

autoseg 는 A 만 쓴다 — 거기서는 지연이 노브(`target_chunk_words`)로 고정돼 **안 끊는 것이
공짜**라, 무분절을 0 으로 세면 "경기를 안 뛰어서 만점"이 된다. AST 는 지연이 x축에 있어
공짜가 아니다(punct-c1 은 무분절의 대가로 StreamLAAL 7.08초를 낸다). 그래서 B 를 주지표로
두고 A 를 함께 낸다 — B 는 사용자가 겪는 총 노출, A 는 정책의 판단력이다. 실측에서 순위가
뒤집힌다: punct-c1 은 A 에서 하위지만 B 에서 1위다.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

AXES = ["static-c4", "static-c5", "static-c6", "seg-c0.5", "seg-c1", "seg-c1.5",
        "segdot-c1", "punct-c1"]

_KEY = re.compile(r"[^a-z0-9']+")
_SENT_END = ("." , "?", "!")


def keys_of(tokens: list[str]) -> tuple[list[str], list[int]]:
    """공백 토큰 → (비교용 키, 키 i 가 가리키는 원 토큰 인덱스). 구두점만인 토큰은 빠진다."""
    ks, idx = [], []
    for i, t in enumerate(tokens):
        k = _KEY.sub("", t.lower())
        if k:
            ks.append(k)
            idx.append(i)
    return ks, idx


def cuts_of(row: dict, sents: list[dict]) -> tuple[list[str], list[tuple[int, int]],
                                                   set[int], int, int]:
    """한 talk → (gold 토큰, 문장 구간, 절단 위치, 매핑 성공 커밋 수, 총 커밋 수).

    절단 위치와 문장 구간은 gold **원 토큰** 인덱스다. 절단 위치 c 는 "gold[c] 앞에서
    끊겼다" 는 뜻이다.
    """
    gold: list[str] = []
    spans: list[tuple[int, int]] = []
    for x in sents:
        toks = x["src"].split()
        spans.append((len(gold), len(gold) + len(toks)))
        gold += toks
    gkeys, gmap = keys_of(gold)

    asr: list[str] = []
    ends: list[int] = []
    for s in row["segments"]:
        asr += (s["original"] or "").split()
        ends.append(len(asr))
    _akeys, amap = keys_of(asr)
    akeys = _akeys
    a_raw2key = {r: i for i, r in enumerate(amap)}

    sm = difflib.SequenceMatcher(None, akeys, gkeys, autojunk=False)
    a2g: dict[int, int] = {}
    for a, b, n in sm.get_matching_blocks():
        for t in range(n):
            a2g[a + t] = b + t

    cuts: set[int] = set()
    mapped = 0
    for e in ends:
        # 커밋의 마지막 키를 gold 로 옮긴다. 매핑이 없으면(ASR 오류 구간) 몇 칸 물러선다.
        last = None
        for r in range(e - 1, max(e - 6, -1), -1):
            if r in a_raw2key and a_raw2key[r] in a2g:
                last = a2g[a_raw2key[r]]
                break
        if last is None:
            continue
        mapped += 1
        nxt = last + 1
        cuts.add(gmap[nxt] if nxt < len(gmap) else len(gold))
    return gold, spans, cuts, mapped, len(ends)


def classify(gold: list[str], lo: int, hi: int, c: int, edge: int) -> str:
    """절단 하나를 원인별로 나눈다. 순서가 중요하다 — 끝 근처를 먼저 걷어낸다."""
    if hi - c <= edge:
        return f"문장 끝 {edge}어절 이내"
    prev = gold[c - 1].rstrip("\"')")
    if prev.endswith(_SENT_END):
        return "gold 문장 안의 마침표 자리"
    if prev.endswith(","):
        return "쉼표 자리"
    return "문장 중간"


def load_floor(fulls: list[str], nli, out_dir: Path, use_floor: bool):
    """길이별 잡음 바닥 `c0(접두사 어절)`. 소스가 영어라 언어마다 같아서 한 번만 잰다."""
    if not use_floor:
        return lambda n: 0.0
    from core.meaning_segmentator.autoseg.gates import noise_floor
    fp = out_dir / "contra_floor_en.json"
    if fp.exists():
        floor = json.loads(fp.read_text(encoding="utf-8"))
    else:
        t0 = time.time()
        floor = noise_floor.measure_floor(fulls, nli, tgt_spaced=True)
        fp.write_text(json.dumps(floor, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[floor] 측정 {time.time() - t0:.0f}s -> {fp}", flush=True)
    print("[floor] " + "  ".join(f"{b} {v['mean']:.3f}"
                                 for b, v in floor["by_length_bucket"].items() if v["n"]),
          flush=True)
    return lambda n: noise_floor.floor_lookup(floor, n)


def run_lang(lang: str, tag: str, axes: list[str], edge: int, nli, out_dir: Path,
             use_floor: bool = True) -> dict:
    man = {}
    with open(HERE / f"manifests/acl6060_eval_en-{lang}.jsonl", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            man[r["utt_id"]] = r["sentences"]
    floor_fn = load_floor([" ".join(s["src"].split()) for v in man.values() for s in v],
                          nli, out_dir, use_floor)

    per_axis: dict[str, dict] = {}
    pairs: dict[tuple[str, str], None] = {}
    for ax in axes:
        m = json.loads((HERE / f"results/ACL6060/{ax}/eval-{lang}/{tag}/metric.json")
                       .read_text(encoding="utf-8"))
        rows, mapped, tot = {}, 0, 0
        for row in m["rows"]:
            sents = man[row["utt_id"]]
            gold, spans, cuts, mp, nc = cuts_of(row, sents)
            mapped += mp
            tot += nc
            for i, (lo, hi) in enumerate(spans):
                inside = sorted(c for c in cuts if lo < c < hi)
                prem = " ".join(gold[lo:hi])
                rows[f"{row['utt_id']}#{sents[i]['seg_id']}"] = {
                    "premise": prem,
                    "cuts": [{"hyp": " ".join(gold[lo:c]), "tail": hi - c, "units": c - lo,
                              "kind": classify(gold, lo, hi, c, edge)} for c in inside]}
                for c in inside:
                    pairs[(prem, " ".join(gold[lo:c]))] = None
        per_axis[ax] = {"rows": rows, "map_rate": mapped / max(tot, 1)}
        print(f"[{lang}/{ax}] 문장 {len(rows)}  절단 {sum(len(r['cuts']) for r in rows.values())}"
              f"  커밋 매핑 {100 * mapped / max(tot, 1):.0f}%", flush=True)

    keys = [k for k in pairs if k[1].strip()]
    n_all = sum(len(r["cuts"]) for d in per_axis.values() for r in d["rows"].values())
    print(f"[{lang}] NLI 고유 쌍 {len(keys)} (중복 제거 전 {n_all})", flush=True)
    t0 = time.time()
    contra, one_minus_ent = nli.score_dual([k[0] for k in keys], [k[1] for k in keys])
    print(f"[{lang}] NLI {len(keys)}쌍 {time.time() - t0:.0f}s", flush=True)
    T = {k: (c, e) for k, c, e in zip(keys, contra, one_minus_ent)}

    out = {"tag": tag, "lang": lang, "edge_words": edge, "floor_subtracted": use_floor,
           "definition": "contra_j = max(0, NLI(premise=gold 영어 문장, "
                         "hypothesis=경계 j 까지 조각) − c0(접두사 길이))",
           "axes": {}}
    for ax, d in per_axis.items():
        per_sent, kinds = {}, {}
        A, A_far, B, C, B_raw = [], [], [], [], []
        for key, r in d["rows"].items():
            vals = []
            for cut in r["cuts"]:
                v = T.get((r["premise"], cut["hyp"]))
                if v is None:
                    continue
                c0 = floor_fn(cut["units"])
                vals.append((max(0.0, v[0] - c0), max(0.0, v[1] - c0), cut["tail"],
                             cut["kind"], v[0]))
                kinds[cut["kind"]] = kinds.get(cut["kind"], 0) + 1
            B.append(sum(v[0] for v in vals))
            B_raw.append(sum(v[4] for v in vals))
            C.append(sum(1 for v in vals if v[0] > 0.5))
            if not vals:
                per_sent[key] = None
                continue
            c = st.mean(v[0] for v in vals)
            per_sent[key] = {"contra": round(c, 4),
                             "contra_raw": round(st.mean(v[4] for v in vals), 4),
                             "one_minus_ent": round(st.mean(v[1] for v in vals), 4),
                             "n_cuts": len(vals)}
            A.append(c)
            far = [v[0] for v in vals if v[2] > edge]
            if far:
                A_far.append(st.mean(far))
        out["axes"][ax] = {
            "per_sentence": per_sent, "map_rate": round(d["map_rate"], 4),
            "n_sentences": len(d["rows"]), "n_sentences_scored": len(A),
            "n_sentences_no_cut": sum(1 for v in per_sent.values() if v is None),
            "n_cuts": sum(len(r["cuts"]) for r in d["rows"].values()),
            "cut_kinds": kinds,
            # 주지표 — 무분절 문장을 0 으로 세어 전체 문장으로 나눈다
            "B_risk_per_sentence": round(sum(B) / len(B), 4),
            "B_risk_per_sentence_raw": round(sum(B_raw) / len(B_raw), 4),
            "A_boundary_mean": round(st.mean(A), 4) if A else None,
            "A_boundary_mean_far": round(st.mean(A_far), 4) if A_far else None,
            "C_risky_cuts_per_sentence": round(sum(C) / len(C), 4)}

    dst = out_dir / f"contra_eval_{tag}_{lang}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[{lang}] -> {dst}", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="ACL 60/60 커밋 경계의 소스 NLI contra")
    p.add_argument("--tag", required=True)
    p.add_argument("--langs", nargs="+", default=["de", "ja", "zh"])
    p.add_argument("--axes", nargs="+", default=AXES)
    p.add_argument("--edge-words", type=int, default=2,
                   help="문장 끝에서 이 어절 이내의 절단은 따로 분류한다 (기본 2)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-floor", action="store_true",
                   help="길이별 잡음 바닥을 빼지 않는다 (기본은 뺀다)")
    p.add_argument("--results-root", default=str(HERE / "results" / "ACL6060"))
    a = p.parse_args()

    from core.meaning_segmentator.autoseg.runtime import metrics
    nli = metrics.make_contradiction_backend(batch_size=a.batch_size)

    outs = {lg: run_lang(lg, a.tag, a.axes, a.edge_words, nli, Path(a.results_root),
                         use_floor=not a.no_floor)
            for lg in a.langs}

    for lg, out in outs.items():
        print(f"\n== {lg} ==  (바닥 {'뺌' if not a.no_floor else '안 뺌'})")
        print(f"{'축':<10} {'B 문장당 위험':>13} {'B raw':>8} {'A 경계평균':>10} "
              f"{f'A(끝{a.edge_words}어절 제외)':>16} {'C 위험절단/문장':>15} {'무분절 문장':>11}")
        for ax in sorted(a.axes, key=lambda x: out["axes"][x]["B_risk_per_sentence"]):
            d = out["axes"][ax]
            print(f"{ax:<10} {d['B_risk_per_sentence']:>13.4f} "
                  f"{d['B_risk_per_sentence_raw']:>8.4f} {d['A_boundary_mean']:>10.4f} "
                  f"{d['A_boundary_mean_far']:>16.4f} "
                  f"{d['C_risky_cuts_per_sentence']:>15.4f} {d['n_sentences_no_cut']:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
