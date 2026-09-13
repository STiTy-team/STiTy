"""run23_gold_<정책>/bleu/{comet_seg_<tgt>.json, <tgt>.json} → 정책 × T COMET, llm_k3 대비 Δ [쌍체 부트스트랩 95% CI].
출력: run23/gold_test.md"""
import json, random, statistics as st, sys
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
R = RUNS_DIR / 'en2x/en-multi'
POL = ['llm_k3', 'A_contra_adqLR', 'B_srcnli_adqLR', 'D_adqLR']
TG = ['de', 'ja', 'zh']; TS = ['4', '6', '8', '12']

def boot(a, b, iters=2000, seed=12345):
    rng = random.Random(seed); n = len(a); d = [x - y for x, y in zip(a, b)]
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return st.mean(d), bs[int(0.025 * iters)], bs[int(0.975 * iters) - 1]

seg, meta = {}, {}
for p in POL:
    for t in TG:
        f = R / f'run23_gold_{p}/bleu/comet_seg_{t}.json'
        g = R / f'run23_gold_{p}/bleu/{t}.json'
        if f.exists() and g.exists():
            seg[(p, t)] = json.load(open(f)); meta[(p, t)] = json.load(open(g))['conditions']
L = ['# 라벨 B(소스 NLI contra) 절단의 gold 검증 — test 100, FLEURS multi_loop405 참조, madlad, COMET-da', '',
     '정책: llm_k3 = run23 v0 k=3 (실사용) / A = 현행 라벨 (번역 NLI contra × adq) / B = 소스 NLI contra × adq / D = adq 만.',
     'Δ 는 llm_k3 대비 쌍체 부트스트랩 95% CI. 굵게 = CI 가 0 을 안 품음.', '']
grid_mean = {p: {} for p in POL}
for t in TG:
    L += [f'## en→{t}', '', '| 정책 | T | k | laal_w | COMET | ΔCOMET vs llm_k3 [95% CI] |', '|---|---|---|---|---|---|']
    for T in TS:
        for p in POL:
            if (p, t) not in seg: continue
            c = seg[(p, t)].get(f'auto_T{T}'); mm = meta[(p, t)].get(f'auto_T{T}', {})
            if not c: continue
            base = seg[('llm_k3', t)].get(f'auto_T{T}') if ('llm_k3', t) in seg else None
            grid_mean[p].setdefault(T, []).append(st.mean(c))
            if p == 'llm_k3' or not base or len(base) != len(c):
                d = '—'
            else:
                m, lo, hi = boot(c, base); s = f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]'
                d = f'**{s}**' if lo > 0 or hi < 0 else s
            L.append(f"| {p} | {T} | {mm.get('k', '')} | {mm.get('laal_words', '')} | {st.mean(c):.4f} | {d} |")
    L.append('')
L += ['## 타깃 3개 평균 COMET', '', '| 정책 | ' + ' | '.join(f'T={T}' for T in TS) + ' | 격자 평균 |', '|---|' + '---|' * (len(TS) + 1)]
for p in POL:
    if not grid_mean[p]: continue
    vals = [st.mean(grid_mean[p][T]) for T in TS if T in grid_mean[p]]
    L.append(f'| {p} | ' + ' | '.join(f'{v:.4f}' for v in vals) + f' | {st.mean(vals):.4f} |')
out = R / 'run23/gold_test.md'
out.write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L)); print('->', out)
