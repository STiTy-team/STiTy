"""정책이 **실제로 고른 자리**의 소스 contra 를 잰다 — gold 가 못 보는 조기 방출.

gold 는 조각 번역을 이어 붙여 채점한다. 앞 조각이 오해를 만들어도 뒤 조각이 메우면
합본은 통과하므로, "사용자가 그 시점에 틀린 것을 읽었나" 는 gold 에 안 잡힌다.
cohesion(G)도 합본을 보므로 같은 맹점이다. 여기서 따로 센다. LLM·GPU 0.

    .venv/bin/python -m ...probe.contra_at_cuts [--split test]
"""
import argparse, json, statistics as st, sys
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop_distill import _seg_with, kept_positions
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import labels as L
from core.meaning_segmentator.autoseg.runtime.pipeline import boundaries

p = argparse.ArgumentParser()
p.add_argument('--split', default='test')
p.add_argument('--run-id', default='en2x/en-multi/run24')
a = p.parse_args()
R = RUNS_DIR / a.run_id
cfg = json.load(open(R / 'config.json')); sp, mg = cfg['spaced'], cfg['min_gap']
grid = cfg['final_t_grid'] if a.split == 'test' else cfg['t_grid']
sents = json.load(open(R / f'data/{a.split}.json'))
lab = json.load(open(R / f'oracle_labels_{a.split}.json')); TG = list(lab)
pre = json.load(open(R / f'pseudoref_{a.split}.json'))

def m(key, i, j): return st.mean(lab[t][i][key][j - 1] for t in TG)
def adq(i, j): return (m('adq_l', i, j) + m('adq_r', i, j)) / 2
def coh(i, j): return st.mean(pre[str(i)][str(j)][t][1] for t in TG)
LABELS = {
    'B (1−contra)×adq': lambda i, j: (1 - m('contra', i, j)) * adq(i, j),
    'D adq 만':          lambda i, j: adq(i, j),
    'G cohesion':       lambda i, j: coh(i, j),
    'H coh×(1−contra)': lambda i, j: coh(i, j) * (1 - m('contra', i, j)),
}
ROWS = {}
for tag, fn in (('정책 v0', 'test_rows.json'), ('정책 min_tgt', 'test_rows_min_tgt_k3.json')):
    p_ = R / fn
    if a.split == 'test' and p_.exists():
        ROWS[tag] = {r['id']: r for r in json.loads(p_.read_text(encoding='utf-8')) if r.get('scores')}

def kept_of(u, c, sc, T):
    return kept_positions(_seg_with(u, {j: int(round(sc[j] * 10000)) for j in c}, sp), T, sp, mg)

print(f'{a.split} {len(sents)}문장 / T {grid} / 소스 contra 전체 평균 '
      f'{st.mean(m("contra", i, j) for i, s in enumerate(sents) for j in range(1, len(L.units_of(s["text"], sp))) ):.4f}')
print(f'\n{"정책/라벨":20s} ' + ' '.join(f'T{T:<12d}' for T in grid) + '  전체')
print(f'{"":20s} ' + ' '.join(f'{"평균 / >0.5":13s}' for _ in grid))
for name in list(LABELS) + list(ROWS):
    cells, allv = [], []
    for T in grid:
        vals = []
        for i, s in enumerate(sents):
            u = L.units_of(s['text'], sp); c = list(range(mg, len(u) - mg + 1))
            if not c or boundaries(s['text'], T, sp) <= 0: continue
            if name in LABELS:
                kept = kept_of(u, c, {j: LABELS[name](i, j) for j in c}, T)
            else:
                r = ROWS[name].get(s['id'])
                if not r: continue
                kept = kept_of(u, c, {j: x / 100 for j, x in zip(r['positions'], r['scores'])}, T)
            vals += [m('contra', i, j) for j in kept]
        cells.append(f'{st.mean(vals):.3f} / {sum(v > 0.5 for v in vals) / len(vals):.3f}' if vals else '—')
        allv += vals
    print(f'{name:20s} ' + ' '.join(f'{c:13s}' for c in cells) + f'  {st.mean(allv):.3f}')
