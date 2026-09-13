"""gold 비교표 — 변형 프롬프트 vs run24 v0, 부트스트랩 95% CI. 라벨 B 오라클을 상한으로 같이 둔다."""
import json, random, statistics as st, sys
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
R = RUNS_DIR / 'en2x/en-multi'
tag = sys.argv[1]                      # 예: min_tgt_k3
TGT = ('de', 'ja', 'zh')
runs = {'변형': f'run24_gold_{tag}', 'run24 v0': 'run24_gold_llm_k3',
        '라벨 B 오라클': 'run23_gold_B_srcnli_adqLR', 'run23 v0': 'run23_gold_llm_k3'}
seg = {}
for k, p in runs.items():
    try:
        seg[k] = {t: json.load(open(R / p / 'bleu' / f'comet_seg_{t}.json')) for t in TGT}
    except FileNotFoundError as e:
        print(f'   (없음: {k} — {e.filename})')
def boot(a, b, iters=2000, seed=1):
    rng = random.Random(seed); n = len(a); d = [x - y for x, y in zip(a, b)]
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return st.mean(d), bs[int(.025 * iters)], bs[int(.975 * iters) - 1]
cols = [k for k in runs if k in seg]
out = [f'# run24 test gold — 변형 `{tag}` vs v0', '',
       '| T | ' + ' | '.join(cols) + ' | Δ(변형−v0) 3타깃 합동 [95% CI] |',
       '|---' * (len(cols) + 2) + '|']
gm = {k: [] for k in cols}
for T in ('4', '6', '8', '12'):
    v = {k: st.mean(st.mean(seg[k][t][f'auto_T{T}']) for t in TGT) for k in cols}
    for k in cols: gm[k].append(v[k])
    x = sum((seg['변형'][t][f'auto_T{T}'] for t in TGT), [])
    y = sum((seg['run24 v0'][t][f'auto_T{T}'] for t in TGT), [])
    m, lo, hi = boot(x, y); s = f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]'
    if lo > 0 or hi < 0: s = f'**{s}**'
    out.append(f'| {T} | ' + ' | '.join(f'{v[k]:.4f}' for k in cols) + f' | {s} |')
out.append('| 격자 평균 | ' + ' | '.join(f'{st.mean(gm[k]):.4f}' for k in cols) + ' | |')
# 무절단 상한과 유지율
for k in cols:
    un = st.mean(st.mean(seg[k][t]['unsegmented']) for t in TGT)
    out.append('') if k == cols[0] else None
    out.append(f'- {k}: 무절단 COMET {un:.4f}')
txt = '\n'.join(out)
print(txt)
(R / f'run24_gold_{tag}' / 'compare.md').write_text(txt + '\n', encoding='utf-8')
