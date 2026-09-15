"""라벨 후보 F/G — 절단 damage 를 **문장 하나 통째로** 잰다. NLI 도 조각 QE 도 안 쓴다.

지금 라벨(A/B)은 위치마다 세 값을 따로 재서 곱한다: (1 − contra) × (adq_l + adq_r)/2.
조각 둘을 따로 채점하므로 "합쳐 놓고 보면 말이 되나" 는 아무도 안 본다. gold 는 정확히
그걸 잰다 — 조각 번역을 이어 붙여 참조와 COMET 으로 비교한다. 라벨을 그 모양으로 맞춘다.

    F (참조 COMET)   src = 원문 전체, mt = MT(prefix) ⊕ MT(suffix), ref = MT(원문 전체)
    G (무참조 QE)    src = 원문 전체, mt = 같은 이어붙임              ← F 의 참조 없는 짝
    H = G × (1 − 소스 contra)

H 를 두는 이유: G 는 조각 번역을 **이어 붙인 뒤** 채점하므로, 앞 조각이 오해를 만들어도
뒤 조각이 메우면 통과한다. 사용자는 그 시점에 이미 틀린 것을 읽었다. 소스 contra 는
그 자리(부정·제한이 뒤에 오는 자리)를 잡으니 둘은 서로 다른 실패를 본다 — dev 실측으로
소스만 모순인 자리 84곳, 번역만 모순인 자리 174곳으로 겹치지 않는다.

참조는 **로컬 번역기가 만든 전체 번역**이다. 사람 참조가 없는 데이터셋에도 붙는다.
조각과 참조가 같은 엔진에서 나오므로 둘의 차이는 절단이 낸 손해뿐이다.

**메모: 이 방향은 `metrics.py` 가 명시적으로 기각해 둔 것이다** — "참조를 full 번역으로
두면 어순을 단조화한 좋은 분절이 감점된다". 맞는 우려고, 그래서 재는 것이다. 판정은
gold 에서 한다: F 로 자른 오라클이 B 오라클(0.7685)을 넘으면 우려보다 이득이 크다.

    .venv/bin/python -m ...probe.pseudoref --split dev            # 라벨 만들고 분석
    .venv/bin/python -m ...probe.pseudoref --split test --emit    # gold 용 절단 내보내기

LLM 호출 0. GPU 만 든다 (madlad 는 대부분 캐시 적중, COMET 두 모델).
"""
from __future__ import annotations
import argparse, json, shutil, statistics as st, sys, time
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop import target_is_spaced
from core.meaning_segmentator.autoseg.loop_distill import _seg_with, kept_positions
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import labels as L, metrics
from core.meaning_segmentator.autoseg.runtime.metrics import _spearman
from core.meaning_segmentator.autoseg.runtime.pipeline import (JsonCache, LocalTranslator,
                                                               boundaries, split_segments,
                                                               to_lang_code, truncate)

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--split', choices=('dev', 'test', 'train', 'extra'), default='dev',
               help="'extra' 는 run26 용 추가 문장 — 라벨만 만들고 분석은 건너뛴다")
p.add_argument('--emit', action='store_true', help='test 절단을 gold 용으로 내보낸다')
p.add_argument('--run-id', default='en2x/en-multi/run24')
p.add_argument('--mt-cache-from', default='en2x/en-multi/run21',
               help='prefix/suffix 번역 캐시를 가진 런 (라벨을 만든 런)')
p.add_argument('--comet', default='Unbabel/wmt22-comet-da')
p.add_argument('--targets', nargs='+', default=None,
               help='cohesion 을 잴 언어들. 기본은 라벨 파일에 있는 것. 여기 없던 언어를 넣으면 '
                    '조각 번역부터 새로 한다 (contra 는 소스 NLI 라 타깃과 무관하다)')
p.add_argument('--batch-size', type=int, default=32)
a = p.parse_args()

R = RUNS_DIR / a.run_id
cfg = json.load(open(R / 'config.json'))
spaced, mg, grid = cfg['spaced'], cfg['min_gap'], cfg['final_t_grid']
sents = json.load(open(R / f'data/{a.split}.json'))
ids = [s['id'] for s in sents]; texts = [s['text'] for s in sents]
units = [L.units_of(t, spaced) for t in texts]
lab = json.load(open(R / f'oracle_labels_{a.split}.json'))
TG = a.targets or list(lab)
mt_cache_dir = RUNS_DIR / a.mt_cache_from / 'cache'
out_path = R / f'pseudoref_{a.split}.json'

# ── 1) 번역 (캐시 적중이 대부분) ────────────────────────────────────────
if out_path.exists():
    blob = json.loads(out_path.read_text(encoding='utf-8'))
    print(f'[skip] {out_path.name} 이미 있음')
else:
    pre_src, suf_src, owner, pos = [], [], [], []
    for i, u in enumerate(units):
        for j in range(1, len(u)):
            pre_src.append(L.join_units(u[:j], spaced)); suf_src.append(L.join_units(u[j:], spaced))
            owner.append(i); pos.append(j)
    print(f'{a.split} {len(sents)}문장 / 경계 {len(pre_src)} / 타깃 {TG}', flush=True)
    cat: dict[str, list[str]] = {}
    full_tr: dict[str, list[str]] = {}
    for tgt in TG:
        code = to_lang_code(tgt); tsp = target_is_spaced(tgt)
        tr = LocalTranslator(tgt_code=code, cache=JsonCache(mt_cache_dir / f'translate_{code}.json'))
        t0 = time.time()
        full_tr[tgt] = tr.full(texts)
        pre_tr = tr.full(pre_src); suf_tr = tr.full(suf_src)
        join = ' ' if tsp else ''
        cat[tgt] = [f'{x}{join}{y}' for x, y in zip(pre_tr, suf_tr)]
        print(f'  [{tgt}] 번역 {time.time()-t0:.0f}s', flush=True)
    from core.meaning_segmentator.autoseg.runtime import pipeline as P
    P._MT_SHARED.clear()
    try:
        import torch, gc; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass

    # ── 2) F: 참조 COMET / G: 무참조 QE ────────────────────────────────
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(a.comet))
    scores: dict[str, dict[str, list[float]]] = {}
    for tgt in TG:
        t0 = time.time()
        batch = [{'src': texts[i], 'mt': m, 'ref': full_tr[tgt][i]} for i, m in zip(owner, cat[tgt])]
        out = model.predict(batch, batch_size=a.batch_size, gpus=1, progress_bar=False, num_workers=0)
        scores.setdefault(tgt, {})['F'] = [float(x) for x in out.scores]
        print(f'  [{tgt}] F(참조 COMET) {len(batch)}건 {time.time()-t0:.0f}s', flush=True)
    del model
    try:
        import torch, gc; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass
    qe = metrics.make_adequacy_backend('cometkiwi', batch_size=a.batch_size)
    for tgt in TG:
        t0 = time.time()
        scores[tgt]['G'] = qe.score([texts[i] for i in owner], cat[tgt])
        print(f'  [{tgt}] G(무참조 QE) {time.time()-t0:.0f}s', flush=True)

    blob = {}
    for k, (i, j) in enumerate(zip(owner, pos)):
        blob.setdefault(str(i), {})[str(j)] = {t: [round(scores[t]['F'][k], 4), round(scores[t]['G'][k], 4)]
                                               for t in TG}
    out_path.write_text(json.dumps(blob, ensure_ascii=False), encoding='utf-8')
    print(f'-> {out_path}', flush=True)
if a.split == 'extra':
    sys.exit(0)

# ── 3) 라벨 정의들 ──────────────────────────────────────────────────────
TG_LAB = [t for t in TG if t in lab]        # adq/contra 라벨이 있는 타깃만 (새로 넣은 언어는 없다)


def m(key, i, j): return st.mean(lab[t][i][key][j - 1] for t in TG_LAB)
def adq(i, j): return (m('adq_l', i, j) + m('adq_r', i, j)) / 2
def pr(i, j, k): return st.mean(blob[str(i)][str(j)][t][k] for t in TG)
DEFS = {
    'B 소스contra × adq (현행)': lambda i, j: (1 - m('contra', i, j)) * adq(i, j),
    'D adq 만':                lambda i, j: adq(i, j),
    'F 참조 COMET (이어붙임)':    lambda i, j: pr(i, j, 0),
    'G 무참조 QE (이어붙임)':     lambda i, j: pr(i, j, 1),
    'F × B':                   lambda i, j: pr(i, j, 0) * (1 - m('contra', i, j)) * adq(i, j),
    'H G × (1−소스contra)':      lambda i, j: pr(i, j, 1) * (1 - m('contra', i, j)),
}
PER_TGT = {
    'B 소스contra × adq (현행)': lambda t, i, j: (1 - lab[t][i]['contra'][j-1]) * (lab[t][i]['adq_l'][j-1] + lab[t][i]['adq_r'][j-1]) / 2,
    'D adq 만':                lambda t, i, j: (lab[t][i]['adq_l'][j-1] + lab[t][i]['adq_r'][j-1]) / 2,
    'F 참조 COMET (이어붙임)':    lambda t, i, j: blob[str(i)][str(j)][t][0],
    'G 무참조 QE (이어붙임)':     lambda t, i, j: blob[str(i)][str(j)][t][1],
}
LAB_BASED = {'B 소스contra × adq (현행)', 'D adq 만'}   # lab 을 읽으므로 TG_LAB 만 돈다

def kept_of(u, c, sc, T):
    return kept_positions(_seg_with(u, {j: int(round(sc[j] * 10000)) for j in c}, spaced), T, spaced, mg)

if a.emit:
    # ── test: F/G 오라클 절단을 gold 로 ────────────────────────────────
    for name, key in (('F_pseudoref', 'F 참조 COMET (이어붙임)'), ('G_qeconcat', 'G 무참조 QE (이어붙임)'),
                      ('H_qeXcontra', 'H G × (1−소스contra)')):
        f = DEFS[key]
        odir = R.parent / f'{R.name}_gold_{name}'
        (odir / 'prompt_eval').mkdir(parents=True, exist_ok=True)
        for fn in ('config.json', 'measured_profile.json'):
            if not (odir / fn).exists():
                shutil.copy(R / fn, odir / fn)
        # 링크 대상은 gitignore 라 새로 받은 기계에는 없다. 끊긴 링크면 bleu_eval 의 mkdir 이
        # FileExistsError 로 죽으므로 대상부터 만든다.
        (R / 'cache').mkdir(exist_ok=True)
        if not (odir / 'cache').is_symlink() and not (odir / 'cache').exists():
            (odir / 'cache').symlink_to(Path('..') / R.name / 'cache')
        rows = []
        for i, s in enumerate(sents):
            u = units[i]
            seg = _seg_with(u, {j: round(f(i, j) * 10000) for j in range(1, len(u))}, spaced)
            by_T = {}
            for T in grid:
                cut, miss = truncate(seg, T, spaced, mg)
                pieces = split_segments(cut) or [texts[i]]
                by_T[str(T)] = {'seg_text': cut, 'k': len(pieces), 'missing_boundaries': miss, 'pieces_src': pieces}
            rows.append({'id': ids[i], 'text': texts[i], 'seg_text': seg, 'valid': True,
                         'full_trans': None, 'by_T': by_T})
        (odir / 'prompt_eval' / f'{name}_test.json').write_text(json.dumps(
            {'prompt_file': f'gold:{name}', 'split': 'test', 't_grid': grid, 'min_gap': mg,
             'src_spaced': spaced, 'tag_convention': 'score', 'rows': rows}, ensure_ascii=False, indent=1),
            encoding='utf-8')
        print(f'[emit] {name} -> {odir.name}  평균 k '
              f'{ {T: round(st.mean(r["by_T"][str(T)]["k"] for r in rows), 2) for T in grid} }', flush=True)
    sys.exit(0)

# ── 4) dev 분석 ─────────────────────────────────────────────────────────
llm_rows = {}
for tag, path in ((('v0 (k=3)', R / 'iter_00/dev_rows.json'),
                   ('minimal_tgt', R / 'rule_probe_minimal_tgt_k3_rows.json'))
                  if a.split == 'dev' else ()):        # 채점 행은 dev 것만 있다
    if path.exists():
        llm_rows[tag] = {r['id']: r for r in json.loads(path.read_text(encoding='utf-8')) if r.get('scores')}

print(f'\n{a.split} {len(sents)}문장 / T {cfg["t_grid"]}')
hdr = f'{"라벨 정의":26s} {"타깃겹침":>8s} {"타깃상관":>8s}' + ''.join(f' {k:>12s}' for k in llm_rows)
print(hdr)
for name, f in DEFS.items():
    pair_ov, pair_sp = [], []
    llm_ov = {k: [] for k in llm_rows}
    for i, s in enumerate(sents):
        u = units[i]; c = list(range(mg, len(u) - mg + 1))
        if not c: continue
        agg = {j: f(i, j) for j in c}
        per_t = ({t: {j: PER_TGT[name](t, i, j) for j in c} for t in (TG_LAB if name in LAB_BASED else TG)}
                 if name in PER_TGT else None)
        for T in cfg['t_grid']:
            if boundaries(s['text'], T, spaced) <= 0: continue
            ko = kept_of(u, c, agg, T)
            if not ko: continue
            if per_t:
                tl = list(per_t)
                ks = [kept_of(u, c, per_t[t], T) for t in tl]
                for x in range(len(tl)):
                    for y in range(x + 1, len(tl)):
                        pair_ov.append(len(set(ks[x]) & set(ks[y])) / max(1, len(ks[x])))
            for tag, rows in llm_rows.items():
                r = rows.get(s['id'])
                if not r: continue
                km = kept_of(u, c, {j: x / 100 for j, x in zip(r['positions'], r['scores'])}, T)
                llm_ov[tag].append(len(set(km) & set(ko)) / len(ko))
        if per_t and len(c) >= 3:
            tl = list(per_t)
            for x in range(len(tl)):
                for y in range(x + 1, len(tl)):
                    w = _spearman([per_t[tl[x]][j] for j in c], [per_t[tl[y]][j] for j in c])
                    if w is not None: pair_sp.append(w)
    po = f'{st.mean(pair_ov):8.3f}' if pair_ov else f'{"—":>8s}'
    ps = f'{st.mean(pair_sp):+8.3f}' if pair_sp else f'{"—":>8s}'
    print(f'{name:26s} {po} {ps}' +
          ''.join(f' {(st.mean(v) if v else float("nan")):12.4f}' for v in llm_ov.values()))

# 정의끼리 얼마나 다른가 — 순위상관
print('\n정의 간 문장 내 순위상관')
keys = list(DEFS)
for x in range(len(keys)):
    for y in range(x + 1, len(keys)):
        acc = []
        for i, s in enumerate(sents):
            u = units[i]; c = list(range(mg, len(u) - mg + 1))
            if len(c) < 3: continue
            w = _spearman([DEFS[keys[x]](i, j) for j in c], [DEFS[keys[y]](i, j) for j in c])
            if w is not None: acc.append(w)
        print(f'   {keys[x]:26s} vs {keys[y]:26s} {st.mean(acc):+.3f}')
