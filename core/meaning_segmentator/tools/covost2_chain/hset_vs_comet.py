"""루프의 자(`H_set`)와 참조 기반 자(COMET)가 같은 순서를 내는가.

루프는 `H_set` 을 최대화하도록 프롬프트를 고쳤고 dev 에서 +0.0187 을 얻었는데, 같은
분절을 참조 COMET 으로 재면 채택본과 초기본 차이가 ±0.005 다. 목적함수가 참조 기반
지표를 대리하는지 확인된 적이 없다 — 비교군까지 같은 자에 올려 두 축을 맞대 본다.

**지연을 통제하지 않으면 순위상관은 의미가 없다.** 두 축 모두 지연이 늘수록 올라가므로
전체 상관은 "둘 다 지연을 반영한다" 는 것만 말한다. 그래서 셋을 나눠 본다:

1. 전체 순위상관 — 참고용(지연 교란 포함)
2. **같은 지연대 안에서의 순위상관** — 실측 LAAL 을 구간으로 묶어 구간마다 계산
3. **정책 쌍 비교** — 같은 지연에서 A 가 B 보다 낫다는 판정이 두 축에서 일치하는 비율

    python3 <이 파일> [--bins 6]
"""
import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import mean

A = Path(__file__).resolve().parents[2] / "experiment" / "artifacts" / "en2x" / "covost2"
FAMS = ("alignatt", "mu_prefix", "causal_align", "syntax")

p = argparse.ArgumentParser()
p.add_argument("--bins", type=int, default=6, help="지연 구간 수")
p.add_argument("--tol-ms", type=float, default=120,
               help="정책 쌍 비교에서 '같은 지연' 으로 볼 최대 차이")
a = p.parse_args()

rec = json.loads((A / "full_judge44" / "hset_conditions.json").read_text())
rows = []
for name, d in rec.items():
    short = name.split("/")[-1]
    if short in ("mechanical_8",):
        continue
    fam = next((f for f in FAMS if short == f or short.startswith(f + "_T")), None)
    if fam is None:
        fam = ("ours-adopted" if name.startswith("full_j44best") else
               "ours-initial" if name.startswith("full_j44v0") else "unsegmented")
    rows.append({"name": name, "fam": fam, "ms": d["laal_ms"], "hset": d["hset"],
                 "comet": mean(d["comet"].values())})
rows.sort(key=lambda r: r["ms"])
print(f"조건 {len(rows)}  지연 {rows[0]['ms']:.0f}~{rows[-1]['ms']:.0f}ms")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                       # 동점은 평균 순위
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = mean(rx), mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den if den else float("nan")


if len(rows) < 4:
    print("\n조건이 4개 미만이라 여기서 멈춘다 — 측정이 끝난 뒤 다시 실행할 것.")
    raise SystemExit(0)

print(f"\n[1] 전체 순위상관 (지연 교란 포함): "
      f"{spearman([r['hset'] for r in rows], [r['comet'] for r in rows]):+.3f}")

lo, hi = rows[0]["ms"], rows[-1]["ms"]
w = (hi - lo) / a.bins
print(f"\n[2] 지연 구간별 순위상관 (구간 폭 {w:.0f}ms)")
print(f"  {'구간(ms)':>16s} {'조건':>4s} {'Spearman':>9s}")
for b in range(a.bins):
    s, e = lo + b * w, lo + (b + 1) * w
    grp = [r for r in rows if s <= r["ms"] < e or (b == a.bins - 1 and r["ms"] == e)]
    if len(grp) < 4:
        print(f"  {s:6.0f}~{e:6.0f} {len(grp):4d}  (조건 부족)")
        continue
    rho = spearman([r["hset"] for r in grp], [r["comet"] for r in grp])
    print(f"  {s:6.0f}~{e:6.0f} {len(grp):4d} {rho:+9.3f}")

print(f"\n[3] 같은 지연(±{a.tol_ms:.0f}ms)의 정책 쌍에서 두 축의 판정이 일치하는가")
agree = disagree = 0
flips = []
for x, y in combinations(rows, 2):
    if x["fam"] == y["fam"] or abs(x["ms"] - y["ms"]) > a.tol_ms:
        continue
    dh, dc = x["hset"] - y["hset"], x["comet"] - y["comet"]
    if dh == 0 or dc == 0:
        continue
    if (dh > 0) == (dc > 0):
        agree += 1
    else:
        disagree += 1
        flips.append((abs(dh), x, y, dh, dc))
tot = agree + disagree
if tot:
    print(f"  쌍 {tot}  일치 {agree} ({100*agree/tot:.1f}%)  뒤집힘 {disagree}")
else:
    print("  비교할 쌍이 없다 (조건이 아직 덜 찼다)")
flips.sort(reverse=True, key=lambda t: t[0])
for _, x, y, dh, dc in flips[:8]:
    print(f"    {x['name'].split('/')[-1]:>18s} vs {y['name'].split('/')[-1]:<18s}"
          f" ΔH {dh:+.4f}  ΔCOMET {dc:+.4f}")

print("\n[4] 정책별 대표값 (지연 1,200~1,600ms 구간 평균)")
print(f"  {'정책':>14s} {'조건':>4s} {'H_set':>8s} {'COMET':>8s}")
for f in ("ours-initial", "ours-adopted", "syntax", "causal_align", "alignatt", "mu_prefix"):
    grp = [r for r in rows if r["fam"] == f and 1200 <= r["ms"] <= 1600]
    if not grp:
        continue
    print(f"  {f:>14s} {len(grp):4d} {mean(r['hset'] for r in grp):8.4f} "
          f"{mean(r['comet'] for r in grp):8.4f}")
