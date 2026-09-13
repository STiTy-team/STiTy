"""임의 프롬프트를 run24 test 100 에 채점하고 gold 용 prompt_eval 을 낸다.

`gold_llm.py` 는 런이 이미 낸 `test_rows.json`(v0 채택본)을 옮겨 담을 뿐이라, 손으로 쓴
변형 프롬프트는 test 채점부터 해야 한다. dev overlap 은 오라클 대비 지표라 오라클이 틀리면
같이 틀린다 — gold(실제 번역 + COMET)만 그걸 가른다.

    PYTHONPATH=. .venv/bin/python -m ...probe.gold_minimal \
        --prompt <변형.txt> --tag min_tgt_k3

캐시는 run24 것을 그대로 쓴다 (키에 프롬프트 해시가 들어가 섞이지 않는다). k=3 이라
`segment_s1/s2.json` 도 같이 쓴다. 비용은 30초마다 산출물에 덮어써 크래시해도 남는다.
"""
from __future__ import annotations
import argparse, json, shutil, statistics as st, sys, threading, time
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.infra.gateway import Gateway, add_provider_args
from core.meaning_segmentator.autoseg.loop_distill import evaluate, _seg_with
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import data, labels as L
from core.meaning_segmentator.autoseg.runtime.pipeline import JsonCache, split_segments, truncate

p = argparse.ArgumentParser()
p.add_argument('--prompt', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--run-id', default='en2x/en-multi/run24')
p.add_argument('--model', default='gpt-5-mini')
p.add_argument('--seg-reasoning-effort', default='medium')
p.add_argument('--batch-size', type=int, default=6)
p.add_argument('--workers', type=int, default=16)
p.add_argument('--budget', type=float, default=3.0)
add_provider_args(p)
a = p.parse_args()

A = RUNS_DIR / a.run_id
cfg = json.load(open(A / 'config.json'))
spaced, mg, grid = cfg['spaced'], cfg['min_gap'], cfg['final_t_grid']
prompt = Path(a.prompt).read_text(encoding='utf-8')
test = [data.Sentence(**r) for r in json.load(open(A / 'data/test.json'))]
lab = json.load(open(A / 'oracle_labels_test.json'))
gw = Gateway.from_args(a, budget=a.budget)
cache = JsonCache(A / 'cache' / 'segment.json')
odir = A.parent / f'{A.name}_gold_{a.tag}'
(odir / 'prompt_eval').mkdir(parents=True, exist_ok=True)
for fn in ('config.json', 'measured_profile.json'):
    if not (odir / fn).exists():
        shutil.copy(A / fn, odir / fn)
if not (odir / 'cache').exists():
    (odir / 'cache').symlink_to(Path('..') / A.name / 'cache')
usage_path = odir / 'usage.json'
stop = threading.Event()
def dump(extra=None):
    usage_path.write_text(json.dumps({'tag': a.tag, 'prompt': a.prompt, 'model': a.model,
        'usage': gw.usage.snapshot(), **(extra or {})}, ensure_ascii=False, indent=1), encoding='utf-8')
def ticker():
    while not stop.wait(30):
        dump(); u = gw.usage.snapshot()
        print(f'  [진행] 호출 {u["calls"]} 누적 ${u["cost"]:.4f}', flush=True)
threading.Thread(target=ticker, daemon=True).start()
t0 = time.time()
try:
    rows, m = evaluate(gw, prompt, test, lab, spaced, mg, grid, cache, a.workers,
                       a.batch_size, None if a.seg_reasoning_effort == 'none' else a.seg_reasoning_effort,
                       k_samples=cfg['k_samples'])
finally:
    stop.set(); cache.flush(); dump()
print(f'[test] {len(test)}문장 {time.time()-t0:.0f}s overlap={m["overlap"]} by_T={m["overlap_by_T"]} '
      f'sp={m["spearman_within"]} fmt={m["format_pass_rate"]}', flush=True)
(A / f'test_rows_{a.tag}.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')

by_id = {r['id']: r for r in rows}
pe = []
for s in test:
    u = L.units_of(s.text, spaced); r = by_id.get(s.id)
    seg = (_seg_with(u, {j: round(x * r.get('tag_scale', 1)) for j, x in zip(r['positions'], r['scores'])}, spaced)
           if r and r.get('scores') else s.text)
    by_T = {}
    for T in grid:
        cut, miss = truncate(seg, T, spaced, mg)
        by_T[str(T)] = {'seg_text': cut, 'k': len(split_segments(cut) or [s.text]),
                        'missing_boundaries': miss, 'pieces_src': split_segments(cut) or [s.text]}
    pe.append({'id': s.id, 'text': s.text, 'seg_text': seg, 'valid': bool(r and r.get('valid')),
               'full_trans': None, 'by_T': by_T})
(odir / f'prompt_eval/{a.tag}_test.json').write_text(json.dumps(
    {'prompt_file': a.prompt, 'split': 'test', 't_grid': grid, 'min_gap': mg, 'src_spaced': spaced,
     'tag_convention': 'score', 'rows': pe}, ensure_ascii=False, indent=1), encoding='utf-8')
dump({'test_metrics': m})
print(f'[emit] {odir.name}', {T: round(st.mean(r['by_T'][str(T)]['k'] for r in pe), 2) for T in grid}, flush=True)
