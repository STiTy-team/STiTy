"""run24 채택본(v0, 소스 NLI 문구, k=3) 의 test 절단을 gold 로 — run23 llm_k3 / 라벨 B 와 비교.
1) prompt_eval 내보내기 (run24_gold_llm_k3)  2) bleu/comet 은 셸에서  3) --compare 로 부트스트랩 표"""
import json, shutil, statistics as st, sys, random
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop_distill import _seg_with
from core.meaning_segmentator.autoseg.runtime import labels as L
from core.meaning_segmentator.autoseg.runtime.pipeline import split_segments, truncate
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
R = RUNS_DIR / 'en2x/en-multi'; A = R / 'run24'
if '--compare' not in sys.argv:
    cfg = json.load(open(A / 'config.json')); spaced = cfg['spaced']; mg = cfg['min_gap']; grid = cfg['final_t_grid']
    test = json.load(open(A / 'data/test.json')); llm = {r['id']: r for r in json.load(open(A / 'test_rows.json'))}
    odir = R / 'run24_gold_llm_k3'; (odir / 'prompt_eval').mkdir(parents=True, exist_ok=True)
    for fn in ('config.json', 'measured_profile.json'):
        if not (odir / fn).exists(): shutil.copy(A / fn, odir / fn)
    if not (odir / 'cache').exists(): (odir / 'cache').symlink_to(Path('..') / 'run24' / 'cache')
    rows = []
    for s in test:
        u = L.units_of(s['text'], spaced); r = llm.get(s['id'])
        seg = (_seg_with(u, {j: round(x * r.get('tag_scale', 1)) for j, x in zip(r['positions'], r['scores'])}, spaced)
               if r and r.get('scores') else s['text'])
        by_T = {}
        for T in grid:
            cut, miss = truncate(seg, T, spaced, mg); pieces = split_segments(cut) or [s['text']]
            by_T[str(T)] = {'seg_text': cut, 'k': len(pieces), 'missing_boundaries': miss, 'pieces_src': pieces}
        rows.append({'id': s['id'], 'text': s['text'], 'seg_text': seg, 'valid': True, 'full_trans': None, 'by_T': by_T})
    (odir / 'prompt_eval/llm_k3_test.json').write_text(json.dumps({'prompt_file': 'run24/best_prompt.txt', 'split': 'test', 't_grid': grid,
        'min_gap': mg, 'src_spaced': spaced, 'tag_convention': 'score', 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('[emit] run24_gold_llm_k3', {T: round(st.mean(r['by_T'][str(T)]['k'] for r in rows), 2) for T in grid})
else:
    def boot(a, b, iters=2000, seed=1):
        rng = random.Random(seed); n = len(a); d = [x - y for x, y in zip(a, b)]
        bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)); return st.mean(d), bs[int(.025 * iters)], bs[int(.975 * iters) - 1]
    seg = {p: {t: json.load(open(R / f'{p}/bleu/comet_seg_{t}.json')) for t in ('de', 'ja', 'zh')}
           for p in ('run24_gold_llm_k3', 'run23_gold_llm_k3', 'run23_gold_B_srcnli_adqLR', 'run23_gold_A_contra_adqLR')}
    L_ = ['# run24 v0 (소스 NLI 문구, k=3) test gold — run23 v0 / 라벨 B 대비', '', '| T | run24 llm | run23 llm | Δ(run24−run23) 3타깃 합동 [95% CI] | 라벨 B 오라클 | 라벨 A 오라클 |', '|---|---|---|---|---|---|']
    gm = {p: [] for p in seg}
    for T in ('4', '6', '8', '12'):
        vals = {p: st.mean(st.mean(seg[p][t][f'auto_T{T}']) for t in ('de', 'ja', 'zh')) for p in seg}
        for p in seg: gm[p].append(vals[p])
        x = sum((seg['run24_gold_llm_k3'][t][f'auto_T{T}'] for t in ('de', 'ja', 'zh')), []); y = sum((seg['run23_gold_llm_k3'][t][f'auto_T{T}'] for t in ('de', 'ja', 'zh')), [])
        m, lo, hi = boot(x, y); s = f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]'; s = f'**{s}**' if lo > 0 or hi < 0 else s
        L_.append(f"| {T} | {vals['run24_gold_llm_k3']:.4f} | {vals['run23_gold_llm_k3']:.4f} | {s} | {vals['run23_gold_B_srcnli_adqLR']:.4f} | {vals['run23_gold_A_contra_adqLR']:.4f} |")
    L_.append(f"| 격자 평균 | {st.mean(gm['run24_gold_llm_k3']):.4f} | {st.mean(gm['run23_gold_llm_k3']):.4f} | | {st.mean(gm['run23_gold_B_srcnli_adqLR']):.4f} | {st.mean(gm['run23_gold_A_contra_adqLR']):.4f} |")
    (A / 'gold_test.md').write_text('\n'.join(L_) + '\n', encoding='utf-8'); print('\n'.join(L_))
