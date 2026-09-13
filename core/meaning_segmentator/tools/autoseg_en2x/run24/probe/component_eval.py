"""변형 vs 기준(run24 v0, k=3) — 라벨 성분별로 LLM 이 무엇을 보게 됐나. dev 215, 비용 0."""
import json, statistics as st, sys, collections
sys.path.insert(0,'.')
from core.meaning_segmentator.autoseg.loop_distill import _seg_with, kept_positions
from core.meaning_segmentator.autoseg.runtime import labels as L
from core.meaning_segmentator.autoseg.runtime.metrics import _spearman
from core.meaning_segmentator.autoseg.runtime.pipeline import boundaries
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
A=RUNS_DIR/'en2x/en-multi/run24'; tag=sys.argv[1]
cfg=json.load(open(A/'config.json')); spaced=cfg['spaced']; mg=cfg['min_gap']; grid=cfg['t_grid']
lab=json.load(open(A/'oracle_labels_dev.json')); tg=list(lab); dev=json.load(open(A/'data/dev.json'))
def rows_of(p): return {r['id']:r for r in json.load(open(p)) if r.get('scores')}
base=rows_of(A/'iter_00/dev_rows.json'); var=rows_of(A/f'rule_probe_{tag}_rows.json')
def comp(i,j,k): return st.mean(lab[t][i][k][j-1] for t in tg)
defs={'전체 라벨':lambda i,j:L.label_value(lab,i,j),'contra 만 (1−c)':lambda i,j:1-comp(i,j,'contra'),
      'adq 만':lambda i,j:(comp(i,j,'adq_l')+comp(i,j,'adq_r'))/2}
def kept_of(u,c,sc,T): return kept_positions(_seg_with(u,{j:int(round(sc[j]*10000)) for j in c},spaced),T,spaced,mg)
print(f'{"":14s} {"기준(v0)":>10s} {"변형":>10s}')
for name,f in defs.items():
    ov={'base':[], 'var':[]}
    for i,s in enumerate(dev):
        u=L.units_of(s['text'],spaced); c=list(range(mg,len(u)-mg+1))
        if not c: continue
        lb={j:f(i,j) for j in c}
        for T in grid:
            if boundaries(s['text'],T,spaced)<=0: continue
            ko=kept_of(u,c,lb,T)
            if not ko: continue
            for key,rows in (('base',base),('var',var)):
                r=rows.get(s['id'])
                if not r: continue
                km=kept_of(u,c,{j:x/100 for j,x in zip(r['positions'],r['scores'])},T); ov[key].append(len(set(km)&set(ko))/len(ko))
    print(f'overlap vs {name:10s} {st.mean(ov["base"]):10.4f} {st.mean(ov["var"]):10.4f}')
for name,k in (('−contra','contra'),('adq_l','adq_l'),('adq_r','adq_r')):
    sp={'base':[], 'var':[]}
    for i,s in enumerate(dev):
        for key,rows in (('base',base),('var',var)):
            r=rows.get(s['id'])
            if not r or len(r['positions'])<3: continue
            v=[(-1 if k=='contra' else 1)*comp(i,j,k) for j in r['positions']]
            w=_spearman([float(x) for x in r['scores']],v)
            if w is not None: sp[key].append(w)
    print(f'spearman vs {name:10s} {st.mean(sp["base"]):+10.3f} {st.mean(sp["var"]):+10.3f}')
