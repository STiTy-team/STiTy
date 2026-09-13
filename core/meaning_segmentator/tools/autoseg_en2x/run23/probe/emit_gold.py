"""gold 검증용 절단 내보내기 — test 100, 정책별 `run23_gold_<정책>/prompt_eval/<정책>_test.json`.

정책 (모두 같은 후보 위치, 절단은 `truncate` 로 T 격자):
  A_contra_adqLR   현행 라벨: (1 − 번역 NLI contra 3타깃 평균) × adq 평균          ← 기준 정답
  B_srcnli_adqLR   (1 − 소스 NLI contra) × adq 평균.  premise=원문, hypothesis=prefix. 번역 무관 contra
  D_adqLR          adq 평균만 (contra 없이)
  llm_k3           run23 v0 k=3 의 test 점수 (test_rows.json)                       ← 실사용 정책

bleu_eval + comet_eval 이 읽는 형식은 `gates/oracle_label.py` 의 emit 과 같다. 번역 캐시는
run23/cache 에 심볼릭 링크 (effective 계산 때 madlad 번역이 이미 있다). LLM 호출 0, GPU 는 NLI 만.
소스 NLI 값은 run23/src_nli_test.json 에 남긴다.
"""
import json, shutil, statistics as st, sys, time
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop_distill import _seg_with
from core.meaning_segmentator.autoseg.runtime import labels as L, metrics
from core.meaning_segmentator.autoseg.runtime.pipeline import split_segments, truncate
from core.meaning_segmentator.autoseg.paths import RUNS_DIR

R = RUNS_DIR / 'en2x/en-multi'
A = R / 'run23'
cfg = json.load(open(A / 'config.json')); spaced = cfg['spaced']; mg = cfg['min_gap']; grid = cfg['final_t_grid']
test = json.load(open(A / 'data/test.json')); ids = [s['id'] for s in test]; texts = [s['text'] for s in test]
lab = json.load(open(R / 'run22/oracle_labels_test.json')); tg = list(lab)
for t in tg:
    assert [e['id'] for e in lab[t]] == ids, f'{t} 라벨 순서가 test 와 다르다'
llm = {r['id']: r for r in json.load(open(A / 'test_rows.json'))}
units = [L.units_of(t, spaced) for t in texts]

nli_path = A / 'src_nli_test.json'
if nli_path.exists():
    src = json.load(open(nli_path))
else:
    nli = metrics.make_contradiction_backend()
    prem, hyp, key = [], [], []
    for i, u in enumerate(units):
        for j in range(1, len(u)):
            prem.append(texts[i]); hyp.append(L.join_units(u[:j], spaced)); key.append((i, j))
    t0 = time.time(); contra, nent = nli.score_dual(prem, hyp)
    src = {}
    for (i, j), c, e in zip(key, contra, nent):
        src.setdefault(str(i), {})[str(j)] = [round(c, 4), round(e, 4)]
    json.dump(src, open(nli_path, 'w'), ensure_ascii=False)
    print(f'소스 NLI test {len(prem)}쌍 {time.time()-t0:.0f}s -> {nli_path.name}', flush=True)

def m(key, i, j): return st.mean(lab[t][i][key][j - 1] for t in tg)
def adq(i, j): return (m('adq_l', i, j) + m('adq_r', i, j)) / 2
policies = {
    'A_contra_adqLR': lambda i, j: (1 - m('contra', i, j)) * adq(i, j),
    'B_srcnli_adqLR': lambda i, j: (1 - src[str(i)][str(j)][0]) * adq(i, j),
    'D_adqLR':        lambda i, j: adq(i, j),
}
def seg_of(pol, i):
    u = units[i]
    if pol == 'llm_k3':
        r = llm.get(ids[i])
        if not r or not r.get('scores'):
            return texts[i]
        ts = r.get('tag_scale', 1)
        return _seg_with(u, {j: round(s * ts) for j, s in zip(r['positions'], r['scores'])}, spaced)
    f = policies[pol]
    return _seg_with(u, {j: round(f(i, j) * 10000) for j in range(1, len(u))}, spaced)

for pol in list(policies) + ['llm_k3']:
    odir = R / f'run23_gold_{pol}'
    (odir / 'prompt_eval').mkdir(parents=True, exist_ok=True)
    for fn in ('config.json', 'measured_profile.json'):
        if not (odir / fn).exists():
            shutil.copy(A / fn, odir / fn)
    if not (odir / 'cache').exists():
        (odir / 'cache').symlink_to(Path('..') / 'run23' / 'cache')
    rows = []
    for i in range(len(test)):
        seg = seg_of(pol, i)
        by_T = {}
        for T in grid:
            cut, miss = truncate(seg, T, spaced, mg)
            pieces = split_segments(cut) or [texts[i]]
            by_T[str(T)] = {'seg_text': cut, 'k': len(pieces), 'missing_boundaries': miss, 'pieces_src': pieces}
        rows.append({'id': ids[i], 'text': texts[i], 'seg_text': seg, 'valid': True, 'full_trans': None, 'by_T': by_T})
    (odir / 'prompt_eval' / f'{pol}_test.json').write_text(json.dumps({
        'prompt_file': f'gold:{pol}', 'split': 'test', 't_grid': grid, 'min_gap': mg,
        'src_spaced': spaced, 'tag_convention': 'score', 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    ks = {T: round(st.mean(r['by_T'][str(T)]['k'] for r in rows), 2) for T in grid}
    print(f'[emit] {pol:16s} -> {odir.name}/prompt_eval/{pol}_test.json  평균 k {ks}', flush=True)
