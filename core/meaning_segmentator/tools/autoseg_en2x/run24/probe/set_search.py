"""경계를 따로 재도 되나 — 그리디(경계별 상위 k) vs 집합 탐색 비교.

라벨은 경계 하나를 **조각 2개** 상태에서 잰다. 실제 절단은 k+1조각을 만들고, 그때 각
조각의 양옆은 이웃 절단이 정한다 — 라벨이 본 적 없는 그림이다. 그 불일치의 크기를 잰다.

    H_set(S) = CometKiwi(원문 전체, 조각 번역들을 순서대로 이어붙임) × (1 − max contra(S))

조각은 후보 자리 사이의 연속 구간뿐이라 O(m²)개다. 한 번 번역해 두면 어떤 집합이든
이어붙이기만 하면 되므로, 집합마다 드는 것은 QE 1회다. LLM 호출 0.

    .venv/bin/python -m ...probe.set_search --n 100
"""
from __future__ import annotations
import argparse, itertools, json, statistics as st, sys, time
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop import target_is_spaced
from core.meaning_segmentator.autoseg.loop_distill import _seg_with, kept_positions
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import labels as L, metrics
from core.meaning_segmentator.autoseg.runtime.pipeline import (JsonCache, LocalTranslator,
                                                               boundaries, capacity,
                                                               to_lang_code)

p = argparse.ArgumentParser()
p.add_argument('--split', default='dev')
p.add_argument('--n', type=int, default=100, help='문장 수 (앞에서부터)')
p.add_argument('--m', type=int, default=8, help='경계별 H 상위 m개로 가지치기')
p.add_argument('--max-sets', type=int, default=200, help='(문장,T)당 열거 상한. 넘으면 그리디+교환')
p.add_argument('--batch-size', type=int, default=32)
p.add_argument('--emit', action='store_true',
               help='탐색 최적 절단집합을 gold 용 prompt_eval 로 내보낸다 (run24_gold_S_search)')
a = p.parse_args()

R24 = RUNS_DIR / 'en2x/en-multi/run24'; R25 = RUNS_DIR / 'en2x/en-multi/run25'
cfg = json.load(open(R24 / 'config.json')); sp, mg = cfg['spaced'], cfg['min_gap']
grid = cfg['final_t_grid'] if a.split == 'test' else cfg['t_grid']
sents = json.load(open(R25 / f'data/{a.split}.json'))[:a.n]
lab = json.load(open(R25 / f'oracle_labels_{a.split}.json')); TG = list(lab)
def coh(i, j): return st.mean(lab[t][i]['adq_l'][j - 1] for t in TG)
def contra(i, j): return lab[TG[0]][i]['contra'][j - 1]
def Hb(i, j): return coh(i, j) * (1 - contra(i, j))

def valid(cut, n):
    """min_gap 을 지키는 절단집합인가 — 조각이 전부 mg 이상."""
    prev = 0
    for j in sorted(cut):
        if j - prev < mg: return False
        prev = j
    return n - prev >= mg

# ── 1) 문장·T 마다 후보 집합 열거 ────────────────────────────────────────
jobs = []          # (i, T, k, greedy_set, [집합들], 가지치기 자리)
spans = set()      # 번역이 필요한 (i, a, b)
t0 = time.time()
for i, s in enumerate(sents):
    u = L.units_of(s['text'], sp); n = len(u)
    cand = list(range(mg, n - mg + 1))
    if not cand: continue
    hb = {j: Hb(i, j) for j in cand}
    seg = _seg_with(u, {j: int(round(hb[j] * 10000)) for j in cand}, sp)
    for T in grid:
        k = min(boundaries(s['text'], T, sp), capacity(s['text'], mg, sp))
        if k < 1: continue
        greedy = tuple(sorted(kept_positions(seg, T, sp, mg)))
        if len(greedy) != k: k = len(greedy)
        if k < 1: continue
        pruned = sorted(set(sorted(cand, key=lambda j: -hb[j])[:a.m]) | set(greedy))
        sets = [c for c in itertools.combinations(pruned, k) if valid(c, n)]
        if len(sets) > a.max_sets:                      # 그리디+교환으로 대체
            sets = [greedy]
            cur = set(greedy)
            for _ in range(3):
                nb = []
                for out_ in list(cur):
                    for in_ in pruned:
                        if in_ in cur: continue
                        c = tuple(sorted((cur - {out_}) | {in_}))
                        if valid(c, n): nb.append(c)
                sets += nb[:a.max_sets]
                break
        if greedy not in sets: sets.append(greedy)
        jobs.append((i, T, k, greedy, sets, pruned))
        for c in sets:
            prev = 0
            for j in list(c) + [n]:
                spans.add((i, prev, j)); prev = j
print(f'{a.split} {len(sents)}문장 / (문장,T) {len(jobs)} / 집합 {sum(len(j[4]) for j in jobs)} / '
      f'번역할 조각 {len(spans)} ({time.time()-t0:.0f}s)', flush=True)

# ── 2) 조각 번역 ─────────────────────────────────────────────────────────
span_list = sorted(spans)
texts_of = {}
for i, s in enumerate(sents):
    texts_of[i] = L.units_of(s['text'], sp)
src_of = {sp_: L.join_units(texts_of[sp_[0]][sp_[1]:sp_[2]], sp) for sp_ in span_list}
tr_of = {}
for tgt in TG:
    code = to_lang_code(tgt)
    tr = LocalTranslator(tgt_code=code, cache=JsonCache(R24 / 'cache' / f'translate_{code}.json'))
    t1 = time.time()
    outs = tr.full([src_of[k_] for k_ in span_list])
    tr_of[tgt] = dict(zip(span_list, outs))
    print(f'  [{tgt}] 조각 번역 {len(span_list)}건 {time.time()-t1:.0f}s', flush=True)
from core.meaning_segmentator.autoseg.runtime import pipeline as P
P._MT_SHARED.clear()
try:
    import torch, gc; gc.collect(); torch.cuda.empty_cache()
except Exception: pass

# ── 3) 집합마다 H_set ────────────────────────────────────────────────────
qe = metrics.make_adequacy_backend('cometkiwi', batch_size=a.batch_size)
srcs, hyps, key = [], [], []
for idx, (i, T, k, greedy, sets, pruned) in enumerate(jobs):
    n = len(texts_of[i])
    for c in sets:
        for tgt in TG:
            join = ' ' if target_is_spaced(tgt) else ''
            prev = 0; parts = []
            for j in list(c) + [n]:
                parts.append(tr_of[tgt][(i, prev, j)]); prev = j
            srcs.append(sents[i]['text']); hyps.append(join.join(parts)); key.append((idx, c, tgt))
print(f'QE {len(srcs)}건', flush=True)
t1 = time.time()
sc = qe.score(srcs, hyps)
print(f'  QE {time.time()-t1:.0f}s', flush=True)
per = {}
for (idx, c, tgt), v in zip(key, sc):
    per.setdefault((idx, c), {})[tgt] = v

# ── 4) 비교 ─────────────────────────────────────────────────────────────
rows = []
for idx, (i, T, k, greedy, sets, pruned) in enumerate(jobs):
    def hset(c):
        q = st.mean(per[(idx, c)][t] for t in TG)
        return q * (1 - max(contra(i, j) for j in c))
    best = max(sets, key=hset)
    rows.append({'i': i, 'T': T, 'k': k, 'greedy': hset(greedy), 'best': hset(best),
                 'same': tuple(best) == tuple(greedy), 'n_sets': len(sets),
                 'frontier': any(j not in greedy and j == pruned[-1] for j in best)})
if a.emit:
    import shutil
    from core.meaning_segmentator.autoseg.runtime.pipeline import split_segments
    best_of = {(x['i'], x['T']): x for x in []}
    bset = {}
    for idx, (i, T, k, greedy, sets, pruned) in enumerate(jobs):
        def hs(c):
            q = st.mean(per[(idx, c)][t] for t in TG)
            return q * (1 - max(contra(i, j) for j in c))
        bset[(i, T)] = max(sets, key=hs)
    odir = R24.parent / 'run24_gold_S_search'
    (odir / 'prompt_eval').mkdir(parents=True, exist_ok=True)
    for fn in ('config.json', 'measured_profile.json'):
        if not (odir / fn).exists():
            shutil.copy(R24 / fn, odir / fn)
    if not (odir / 'cache').exists():
        (odir / 'cache').symlink_to(Path('..') / 'run24' / 'cache')
    erows = []
    for i, s_ in enumerate(sents):
        u = texts_of[i]
        by_T = {}
        for T in grid:
            c = bset.get((i, T))
            if c is None:
                seg = s_['text']
            else:
                seg = _seg_with(u, {j: 100 for j in c}, sp)
            pieces = split_segments(seg) or [s_['text']]
            by_T[str(T)] = {'seg_text': seg, 'k': len(pieces), 'missing_boundaries': 0,
                            'pieces_src': pieces}
        erows.append({'id': s_['id'], 'text': s_['text'], 'seg_text': s_['text'], 'valid': True,
                      'full_trans': None, 'by_T': by_T})
    (odir / 'prompt_eval' / 'S_search_test.json').write_text(json.dumps(
        {'prompt_file': 'search:H_set', 'split': a.split, 't_grid': grid, 'min_gap': mg,
         'src_spaced': sp, 'tag_convention': 'score', 'rows': erows}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print(f'[emit] run24_gold_S_search  평균 k '
          f'{ {T: round(st.mean(r["by_T"][str(T)]["k"] for r in erows), 2) for T in grid} }', flush=True)

dump = {}
for idx, (i, T, k, greedy, sets, pruned) in enumerate(jobs):
    def hs2(c):
        q = st.mean(per[(idx, c)][t] for t in TG)
        return round(q * (1 - max(contra(i, j) for j in c)), 5)
    dump.setdefault(str(i), {})[str(T)] = {'greedy': list(greedy),
                                           'sets': {','.join(map(str, c)): hs2(c) for c in sets}}
(R25 / f'set_scores_{a.split}.json').write_text(json.dumps(dump, ensure_ascii=False), encoding='utf-8')

out = R25 / f'set_search_{a.split}.json'
out.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
print(f'\n{"T":>4s} {"n":>5s} {"그리디":>9s} {"탐색최적":>9s} {"차이":>9s} {"최적=그리디":>10s} {"집합수":>7s}')
for T in grid:
    r = [x for x in rows if x['T'] == T]
    if not r: continue
    print(f'{T:4d} {len(r):5d} {st.mean(x["greedy"] for x in r):9.4f} {st.mean(x["best"] for x in r):9.4f} '
          f'{st.mean(x["best"]-x["greedy"] for x in r):+9.4f} {sum(x["same"] for x in r)/len(r):10.2f} '
          f'{st.mean(x["n_sets"] for x in r):7.1f}')
print(f'전체 차이 평균 {st.mean(x["best"]-x["greedy"] for x in rows):+.4f} / '
      f'최적이 그리디와 같은 비율 {sum(x["same"] for x in rows)/len(rows):.2f} / -> {out.name}')
