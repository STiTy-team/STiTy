"""run14_v* 의 bleu/{tgt}.json → 버전 × T COMET 표 + dev score 와의 순위상관."""
import json, sys, statistics as st
from pathlib import Path
R = Path("/home/mobility/STiTy/core/meaning_segmentator/experiment/artifacts/en2x/en-multi")
hist = {h["version"]: h for h in json.load(open(R / "run14/history.json"))}
tgts = ["de", "ja", "zh"]; Ts = ["4", "6", "8", "12"]
L = ["# run14 버전별 참조 기반 COMET (test 100, FLEURS multi_loop405)", "",
     "T 는 목표값, 괄호는 실현 평균 조각 크기. `dev` 는 루프의 dev score(effective 격자 평균), `adopted` 는 채택 여부.", ""]
per = {}
for v in range(6):
    row = {}
    for t in tgts:
        f = R / f"run14_v{v}/bleu/{t}.json"
        if not f.exists(): continue
        C = json.load(open(f))["conditions"]
        row[t] = {T: (C.get(f"auto_T{T}", {}).get("comet"), C.get(f"auto_T{T}", {}).get("piece_units")) for T in Ts}
        row[t]["unseg"] = (C.get("unsegmented", {}).get("comet"), None)
    per[v] = row
def rank_corr(x, y):
    n = len(x); rx = sorted(range(n), key=lambda i: x[i]); ry = sorted(range(n), key=lambda i: y[i])
    px = [0]*n; py = [0]*n
    for r, i in enumerate(rx): px[i] = r
    for r, i in enumerate(ry): py[i] = r
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(px, py)) / (n * (n * n - 1)) if n > 2 else None
for t in tgts:
    L += [f"## en→{t}", "", "| v | adopted | dev | " + " | ".join(f"T{T}" for T in Ts) + " | unseg |", "|---|---|---|" + "---|" * (len(Ts) + 1)]
    for v in range(6):
        r = per.get(v, {}).get(t)
        if not r: continue
        cells = [("—" if r[T][0] is None else f"{r[T][0]:.4f} ({r[T][1]:.1f})") for T in Ts]
        L.append(f"| {v} | {'O' if hist[v]['adopted'] else 'X'} | {hist[v]['score_dev']} | " + " | ".join(cells) + f" | {r['unseg'][0]:.4f} |")
    L.append("")
# 타깃 평균 & 상관
L += ["## 타깃 3개 평균 COMET vs dev score", "", "| v | dev | " + " | ".join(f"T{T}" for T in Ts) + " | 격자평균 |", "|---|---|" + "---|" * (len(Ts) + 1)]
mean_grid, devs = [], []
for v in range(6):
    if not all(t in per.get(v, {}) for t in tgts): continue
    vals = {T: st.mean(per[v][t][T][0] for t in tgts) for T in Ts if all(per[v][t][T][0] is not None for t in tgts)}
    g = st.mean(vals.values())
    mean_grid.append(g); devs.append(hist[v]["score_dev"])
    L.append(f"| {v} | {hist[v]['score_dev']} | " + " | ".join(f"{vals[T]:.4f}" for T in Ts) + f" | {g:.4f} |")
rc = rank_corr(devs, mean_grid)
L += ["", f"버전 간 Spearman(dev effective, test COMET 격자평균) = **{rc:.2f}** (n={len(devs)})" if rc is not None else "", ""]
(R / "run14/version_comet.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
