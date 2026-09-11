# -*- coding: utf-8 -*-
"""벤치 결과 JSON 기계 점검. 표준 라이브러리만 쓴다.

사용:
    python3 audit_results.py <결과JSON 또는 디렉터리> [...]

각 JSON 은 {"summary": {...}, "rows": [...]} 형식을 가정한다(smoke_client 출력).
언어별로 묶어 백엔드끼리 비교하고, 점검마다 PASS / FAIL / WARN / SKIP 을 찍는다.
FAIL 이 하나라도 있으면 종료 코드 1 - 그 수치를 보고서에 싣지 않는다.
"""
from __future__ import print_function
import json
import os
import re
import sys
import unicodedata

DIGIT = re.compile(r"\d")
VERDICT = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}


def say(verdict, name, detail=""):
    VERDICT[verdict] = VERDICT.get(verdict, 0) + 1
    mark = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[skip]"}[verdict]
    print("%s %-34s %s" % (mark, name, detail))


def median(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def norm(s):
    s = unicodedata.normalize("NFKC", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def load(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith(".json"):
                        out.append(os.path.join(root, f))
        else:
            out.append(p)
    runs = []
    for p in out:
        try:
            d = json.load(open(p))
        except Exception as e:                                   # noqa: BLE001
            print("  (건너뜀: %s — %s)" % (p, e))
            continue
        if not isinstance(d, dict) or "rows" not in d or "summary" not in d:
            continue
        s = d["summary"]
        runs.append({"path": p, "summary": s, "rows": d["rows"],
                     "backend": s.get("backend", os.path.basename(p)),
                     "lang": s.get("lang", "?"),
                     "unit": s.get("unit", "cer" if s.get("lang") == "ko" else "wer")})
    return runs


def err_values(run):
    u = run["unit"]
    return [r[u] for r in run["rows"] if r.get(u) is not None]


def latency_key(run):
    for k in ("laal_first_ms", "laal_ca_ms", "laal_ms"):
        if run["rows"] and run["rows"][0].get(k) is not None:
            return k
    return None


# ── 점검들 ────────────────────────────────────────────────────────────────
def check_same_clips(group, lang):
    ids = {r["backend"]: [x.get("file_id") for x in r["rows"]] for r in group}
    durs = {b: round(sum(x.get("audio_sec", 0) for x in r["rows"]), 1)
            for b, r in ((r["backend"], r) for r in group)}
    first = list(ids.values())[0]
    if all(v == first for v in ids.values()):
        say("PASS", "[%s] 같은 클립" % lang,
            "%d클립 / 총 %.1fs" % (len(first), list(durs.values())[0]))
    else:
        say("FAIL", "[%s] 같은 클립" % lang,
            "개수 %s · 총길이 %s" % ({b: len(v) for b, v in ids.items()}, durs))


def check_complete(run):
    s = run["summary"]
    n_ok, n_total = s.get("n_ok"), s.get("n_total")
    empty = s.get("empty_transcripts")
    tag = "%s/%s" % (run["backend"], run["lang"])
    if n_ok is not None and n_total is not None and n_ok != n_total:
        say("FAIL", "완주 " + tag, "%s/%s 클립만 성공" % (n_ok, n_total))
    elif empty:
        say("FAIL", "완주 " + tag, "빈 전사 %s건 — 원인 규명 전까지 수치 사용 금지" % empty)
    else:
        say("PASS", "완주 " + tag, "%s클립, 빈 전사 0" % n_ok)


def check_conditions(group, lang):
    keys = ["peak_normalize", "laal_unit", "unit"]
    diffs = []
    for k in keys:
        vals = {r["backend"]: r["summary"].get(k) for r in group}
        if len(set(map(str, vals.values()))) > 1:
            diffs.append("%s=%s" % (k, vals))
    if diffs:
        say("FAIL", "[%s] 조건 통일" % lang, " · ".join(diffs))
    else:
        say("PASS", "[%s] 조건 통일" % lang,
            "peak=%s, 단위=%s" % (group[0]["summary"].get("peak_normalize"),
                                  group[0]["summary"].get("laal_unit")))


def check_warmup(run):
    k = latency_key(run)
    tag = "%s/%s" % (run["backend"], run["lang"])
    if not k:
        say("SKIP", "워밍업 오염 " + tag, "지연 필드 없음")
        return
    v = [r[k] for r in run["rows"] if r.get(k) is not None]
    if len(v) < 5:
        say("SKIP", "워밍업 오염 " + tag, "표본 부족")
        return
    med = median(v[1:])
    if med and v[0] > 2 * med:
        say("FAIL", "워밍업 오염 " + tag,
            "클립1 %.0fms vs 중앙값 %.0fms — 모델 로딩이 섞였다" % (v[0], med))
    else:
        say("PASS", "워밍업 오염 " + tag, "클립1 %.0fms / 중앙값 %.0fms" % (v[0], med))


def check_tail(run):
    ratios = []
    tag = "%s/%s" % (run["backend"], run["lang"])
    for r in run["rows"]:
        ref, hyp = norm(r.get("reference")), norm(r.get("transcript"))
        if not ref:
            continue
        if run["lang"] == "ko":
            ratios.append(len(hyp.replace(" ", "")) / float(max(len(ref.replace(" ", "")), 1)))
        else:
            ratios.append(len(hyp.split()) / float(max(len(ref.split()), 1)))
    m = median(ratios)
    if m is None:
        say("SKIP", "꼬리 잘림 " + tag)
    elif 0.95 <= m <= 1.05:
        say("PASS", "꼬리 잘림 " + tag, "길이비 중앙 %.3f" % m)
    else:
        say("FAIL", "꼬리 잘림 " + tag,
            "길이비 중앙 %.3f — 발화 앞뒤를 놓치거나 덧붙이고 있다" % m)


def check_mean_domination(run):
    v = err_values(run)
    tag = "%s/%s" % (run["backend"], run["lang"])
    if not v:
        say("SKIP", "평균 지배 " + tag)
        return
    total = sum(v)
    worst = max(v)
    share = worst / total if total else 0
    if share > 0.30:
        say("WARN", "평균 지배 " + tag,
            "최악 1클립이 평균의 %.0f%% — 중앙값(%.4f)도 같이 볼 것" % (100 * share, median(v)))
    else:
        say("PASS", "평균 지배 " + tag,
            "평균 %.4f / 중앙 %.4f" % (total / len(v), median(v)))


def check_itn(run):
    tag = "%s/%s" % (run["backend"], run["lang"])
    ref_d = [r for r in run["rows"] if DIGIT.search(r.get("reference") or "")]
    hyp_d = [r for r in run["rows"] if DIGIT.search(r.get("transcript") or "")]
    if not ref_d:
        say("SKIP", "숫자 표기 " + tag, "참조에 숫자 없음")
        return
    u = run["unit"]
    with_d = [r[u] for r in ref_d if r.get(u) is not None]
    without = [r[u] for r in run["rows"]
               if not DIGIT.search(r.get("reference") or "") and r.get(u) is not None]
    gap = ""
    if with_d and without:
        gap = " · 숫자클립 %.4f vs 그 외 %.4f" % (sum(with_d) / len(with_d),
                                                  sum(without) / len(without))
    if not hyp_d:
        say("FAIL", "숫자 표기 " + tag,
            "참조 %d클립에 숫자가 있는데 가설엔 0건 — 재채점으로 해결(GPU 불필요)%s"
            % (len(ref_d), gap))
    else:
        say("PASS", "숫자 표기 " + tag, "가설 %d클립에 숫자 출력%s" % (len(hyp_d), gap))


def check_pieces(group, lang):
    """클립당 출력 조각 수. 확정만 1개대인데 다른 백엔드가 10+ 면 채널 누락 의심.

    미확정 채널을 이미 수집한 런은 요약의 partials_per_clip(또는 행의 n_partial)에
    값이 있다. 그 경우 합쳐서 보고 경고하지 않는다.
    """
    pieces, shown = {}, {}
    for r in group:
        n = [x.get("num_finals", 0) for x in r["rows"]]
        fin = sum(n) / float(max(len(n), 1))
        par = r["summary"].get("partials_per_clip")
        if par is None:
            pn = [x.get("n_partial") for x in r["rows"] if x.get("n_partial") is not None]
            par = (sum(pn) / float(len(pn))) if pn else None
        pieces[r["backend"]] = fin + (par or 0)
        shown[r["backend"]] = ("확정 %.1f" % fin) if par is None \
            else ("미확정 %.1f + 확정 %.1f" % (par, fin))
    if not pieces:
        return
    mx = max(pieces.values())
    thin = [b for b, v in pieces.items() if v < 1.5 and mx >= 5]
    detail = " · ".join("%s: %s" % (b, shown[b]) for b in sorted(shown))
    if thin:
        say("WARN", "[%s] 출력 채널" % lang,
            "%s 는 클립당 출력이 1개대 — 미확정 채널을 빼먹었는지 확인할 것 (%s)"
            % (", ".join(thin), detail))
    else:
        say("PASS", "[%s] 출력 채널" % lang, detail)


def check_repro(group, lang):
    by = {}
    for r in group:
        by.setdefault(r["backend"], []).append(r)
    done = False
    for b, runs in by.items():
        if len(runs) < 2:
            continue
        done = True
        a, c = runs[0], runs[1]
        u = a["unit"]
        mism = sum(1 for x, y in zip(a["rows"], c["rows"])
                   if abs((x.get(u) or 0) - (y.get(u) or 0)) > 1e-9)
        if mism:
            say("FAIL", "[%s] 재현성 %s" % (lang, b), "클립별 불일치 %d건" % mism)
        else:
            say("PASS", "[%s] 재현성 %s" % (lang, b), "두 런 클립별 완전 일치")
    if not done:
        say("SKIP", "[%s] 재현성" % lang, "같은 백엔드 런이 하나뿐")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__)
        return 2
    runs = load(args)
    if not runs:
        print("결과 JSON 을 찾지 못했다.")
        return 2

    print("=" * 78)
    print("대상 %d개" % len(runs))
    for r in runs:
        print("  %-18s %-3s  %s" % (r["backend"], r["lang"], r["path"]))
    print("=" * 78)

    langs = {}
    for r in runs:
        langs.setdefault(r["lang"], []).append(r)

    for lang, group in sorted(langs.items()):
        print("\n--- %s ---" % lang)
        check_same_clips(group, lang)
        check_conditions(group, lang)
        check_pieces(group, lang)
        check_repro(group, lang)
        for r in group:
            check_complete(r)
            check_warmup(r)
            check_tail(r)
            check_mean_domination(r)
            check_itn(r)

    print("\n" + "=" * 78)
    print("PASS %d · WARN %d · FAIL %d · SKIP %d"
          % (VERDICT["PASS"], VERDICT["WARN"], VERDICT["FAIL"], VERDICT["SKIP"]))
    if VERDICT["FAIL"]:
        print("FAIL 이 있다. 이 수치를 보고서에 싣지 말 것.")
        return 1
    if VERDICT["WARN"]:
        print("WARN 은 사람이 판단할 것. 특히 '출력 채널' 경고는 SKILL.md 2단계로.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
