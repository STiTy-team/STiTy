"""소스 NLI contra 프로토타입 — dev 215. premise = 영어 원문 전체, hypothesis = prefix (units[:j]).
번역을 안 거치는 '뒤가 앞을 뒤집나' 라벨. 번역 contra(3타깃) 와 비교하고, 라벨의 contra 를 이것으로
바꿨을 때 타깃 간 일치·LLM overlap 이 어떻게 되나 본다. LLM 호출 0, GPU 만 든다.
산출물: run23/src_nli_dev.json (위치별 contra·1−entail, 문장별 후보 위치)."""
import json, statistics as st, sys, itertools, collections, time
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop_distill import _seg_with, kept_positions
from core.meaning_segmentator.autoseg.runtime import labels as L, metrics
from core.meaning_segmentator.autoseg.runtime.metrics import _spearman
from core.meaning_segmentator.autoseg.runtime.pipeline import boundaries
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
A = RUNS_DIR / 'en2x/en-multi/run23'
cfg = json.load(open(A / 'config.json')); spaced = cfg['spaced']; mg = cfg['min_gap']; grid = cfg['t_grid']
lab = json.load(open(A.parent / 'run22/oracle_labels_dev.json')); tg = list(lab)
dev = json.load(open(A / 'data/dev.json'))
rows = {r['id']: r for r in json.load(open(A / 'iter_00/dev_rows.json')) if r.get('scores')}
out_path = A / 'src_nli_dev.json'

if out_path.exists():
    src = json.load(open(out_path))
else:
    nli = metrics.make_contradiction_backend()
    prem, hyp, key = [], [], []
    for i, s in enumerate(dev):
        u = L.units_of(s['text'], spaced)
        for j in range(mg, len(u) - mg + 1):
            prem.append(s['text']); hyp.append(L.join_units(u[:j], spaced)); key.append((i, j))
    t0 = time.time()
    contra, one_minus_ent = nli.score_dual(prem, hyp)
    print(f'NLI {len(prem)}쌍 {time.time()-t0:.0f}s', flush=True)
    src = collections.defaultdict(dict)
    for (i, j), c, e in zip(key, contra, one_minus_ent):
        src[str(i)][str(j)] = [round(c, 4), round(e, 4)]
    json.dump(src, open(out_path, 'w'), ensure_ascii=False)
    print('->', out_path, flush=True)

def tr(t, i, j, k):
    d = lab[t][i]
    return d[k][j - 1]
def tr_label(t, i, j): return (1 - tr(t, i, j, 'contra')) * (tr(t, i, j, 'adq_l') + tr(t, i, j, 'adq_r')) / 2
def s_contra(i, j): return src[str(i)][str(j)][0]
def s_nent(i, j): return src[str(i)][str(j)][1]

# (a) 소스 contra vs 번역 contra 평균 / 각 타깃 — 문장 내 순위상관
acc = collections.defaultdict(list)
for i, s in enumerate(dev):
    u = L.units_of(s['text'], spaced); c = list(range(mg, len(u) - mg + 1))
    if len(c) < 3: continue
    trm = [st.mean(tr(t, i, j, 'contra') for t in tg) for j in c]
    for name, v in (('src_contra', [s_contra(i, j) for j in c]), ('src_1-ent', [s_nent(i, j) for j in c])):
        w = _spearman(v, trm); acc[f'{name} vs 번역contra평균'].append(w if w is not None else 0)
        for t in tg:
            w = _spearman(v, [tr(t, i, j, 'contra') for j in c]); acc[f'{name} vs {t[:2]}'].append(w if w is not None else 0)
    for a, b in itertools.combinations(tg, 2):
        w = _spearman([tr(a, i, j, 'contra') for j in c], [tr(b, i, j, 'contra') for j in c]); acc['번역contra 타깃쌍'].append(w if w is not None else 0)
print('\n(a) 문장 내 순위상관 평균'); [print(f'   {k:32s} {st.mean(v):+.3f}') for k, v in acc.items()]
print('   소스 contra 평균값', round(st.mean(s_contra(i, j) for i in src for j in src[i]), 3), '/ 번역 contra 평균값',
      round(st.mean(tr(t, i, j, 'contra') for i, s in enumerate(dev) for t in tg for j in range(mg, len(L.units_of(s['text'], spaced)) - mg + 1)), 3))

# (b) 라벨 정의별 타깃 간 일치 (각 타깃의 라벨 = contra항 × 그 타깃 adq) 와 (c) LLM overlap
defs = {
    'A 번역contra × adq (현행)': lambda t, i, j: tr_label(t, i, j),
    'B 소스contra × adq':       lambda t, i, j: (1 - s_contra(i, j)) * (tr(t, i, j, 'adq_l') + tr(t, i, j, 'adq_r')) / 2,
    'C 소스(1−ent) × adq':      lambda t, i, j: (1 - s_nent(i, j)) * (tr(t, i, j, 'adq_l') + tr(t, i, j, 'adq_r')) / 2,
    'D adq 만':                 lambda t, i, j: (tr(t, i, j, 'adq_l') + tr(t, i, j, 'adq_r')) / 2,
    'E 소스contra 만':          lambda t, i, j: 1 - s_contra(i, j),
}
def kept_of(u, c, sc, T): return kept_positions(_seg_with(u, {j: int(round(sc[j] * 10000)) for j in c}, spaced), T, spaced, mg)
print('\n(b)(c) 라벨 정의별: 타깃 간 절단 겹침 / 타깃 순위상관 / LLM(v0,k=3) overlap / 구두점-only overlap')
pun = collections.defaultdict(list)
trn = json.load(open(A / 'data/train.json')); labtr = json.load(open(A.parent / 'run22/oracle_labels_train.json'))
for i, s in enumerate(trn):
    u = L.units_of(s['text'], spaced)
    for j in range(mg, len(u) - mg + 1): pun[u[j-1][-1] if u[j-1][-1] in ',;:.)"\'' else '-'].append(L.label_value(labtr, i, j))
pm = {k: st.mean(v) for k, v in pun.items()}
for name, f in defs.items():
    pair_ov, pair_sp, llm_ov, pun_ov = [], [], [], []
    for i, s in enumerate(dev):
        u = L.units_of(s['text'], spaced); c = list(range(mg, len(u) - mg + 1))
        if not c: continue
        per = {t: {j: f(t, i, j) for j in c} for t in tg}; mean = {j: st.mean(per[t][j] for t in tg) for j in c}
        r = rows.get(s['id'])
        if len(c) >= 3:
            for a, b in itertools.combinations(tg, 2):
                w = _spearman([per[a][j] for j in c], [per[b][j] for j in c]); pair_sp.append(w if w is not None else 0)
        for T in grid:
            if boundaries(s['text'], T, spaced) <= 0: continue
            ko = kept_of(u, c, mean, T)
            if not ko: continue
            km_t = {t: kept_of(u, c, per[t], T) for t in tg}
            for a, b in itertools.combinations(tg, 2): pair_ov.append(len(set(km_t[a]) & set(km_t[b])) / len(km_t[b]))
            if r:
                km = kept_of(u, c, {j: x / 100 for j, x in zip(r['positions'], r['scores'])}, T); llm_ov.append(len(set(km) & set(ko)) / len(ko))
            kp = kept_of(u, c, {j: pm.get(u[j-1][-1] if u[j-1][-1] in ',;:.)"\'' else '-', 0.6) + 1e-6 * (len(c) - j) for j in c}, T); pun_ov.append(len(set(kp) & set(ko)) / len(ko))
    print(f'   {name:26s} 타깃겹침 {st.mean(pair_ov):.3f}  타깃상관 {st.mean(pair_sp):+.3f}  LLM {st.mean(llm_ov):.3f}  구두점 {st.mean(pun_ov):.3f}')
