"""run26 — judge 루프용 3분할: train(사례·예시 출처) / test-A(채택 판정) / test-B(체크포인트·확인).

run25 는 dev-A 150 에서 사례를 뽑고 같은 dev-A 에서 채택을 판정했다. 실측 예시(사례 문장 +
라벨 순위)가 프롬프트에 들어간 채 그 문장에서 채점되니 그 문장만 정답을 보고 푸는 셈이었다
(judge09 iter 2: dev-A Δ +0.0105 중 +0.006 이 예시 3문장 몫). 그래서 출처와 판정을 가른다.

    train  200 = run25 train                                   사례·실측 예시
    test_a 200 = run25 dev[:150] + 추가 50                      채택 판정 (선별 50 + 본채점)
    test_b 200 = run25 dev[150:] (65) + run25 test (100) + 추가 35   체크포인트·유망 확인·최종 표
    최종 홀드아웃 test 는 없다 — test_b 가 그 자리를 겸한다 (판정에 안 쓰고 확인에만 쓴다).

추가 85문장은 어느 분할에도 없고 프롬프트 재료(spare[:28], v0 프로파일·예시)도 아닌 문장에서
시드로 뽑는다. 라벨은 run25 와 같은 절차: contra·번역 QE 는 `labels.compute_labels`, cohesion 은
`pseudoref --split extra`, 둘을 합쳐 `cohesion × (1 − contra)` 꼴로 굽는다. LLM 0콜, GPU 만.

    .venv-autoseg/bin/python -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.make_run26 --phase labels
    .venv-autoseg/bin/python -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split extra \
        --run-id en2x/en-multi/run24 --mt-cache-from en2x/en-multi/run24
    .venv-autoseg/bin/python -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.make_run26 --phase build
"""
import argparse, json, random, shutil, statistics as st, sys
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop import target_is_spaced
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import data, labels as L, metrics

R24 = RUNS_DIR / 'en2x/en-multi/run24'
R25 = RUNS_DIR / 'en2x/en-multi/run25'
R26 = RUNS_DIR / 'en2x/en-multi/run26'
N_EXTRA, N_A, SEED, N_MATERIAL = 85, 50, 20260915, 28

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--phase', choices=('labels', 'build'), required=True)
a = p.parse_args()

cfg = json.loads((R25 / 'config.json').read_text(encoding='utf-8'))
spaced, targets = cfg['spaced'], cfg['targets']
split = {s: json.loads((R25 / f'data/{s}.json').read_text(encoding='utf-8')) for s in ('dev', 'train', 'test')}
pool = {r['id'] for s in split.values() for r in s}
allx = data.load(cfg['dataset'])
spare = [x for x in allx if x.id not in pool]
rng = random.Random(SEED)
cand = spare[N_MATERIAL:]
rng.shuffle(cand)
extra = [x.to_dict() for x in cand[:N_EXTRA]]
extra_path = R24 / 'data/extra.json'

if a.phase == 'labels':
    lens = [len(x['text'].split()) for x in extra]
    print(f'[extra] {len(extra)}문장 / 어절 평균 {st.mean(lens):.1f} 최대 {max(lens)} 최소 {min(lens)}')
    assert max(lens) <= 60, '이상치 — 라벨 전에 데이터 위생 확인'
    if extra_path.exists():
        assert json.loads(extra_path.read_text(encoding='utf-8')) == extra, 'extra.json 이 다른 표본이다'
    else:
        extra_path.write_text(json.dumps(extra, ensure_ascii=False, indent=1), encoding='utf-8')
    adequacy = metrics.make_adequacy_backend(cfg.get('adequacy_backend', 'cometkiwi'), batch_size=32)
    contradiction = metrics.make_contradiction_backend()
    L.compute_labels(R24, 'extra', [x['id'] for x in extra], [x['text'] for x in extra], targets,
                     spaced, adequacy, contradiction, cfg['local_mt_model'], target_is_spaced,
                     contra_source='source')
    print(f'-> {R24}/oracle_labels_extra.json')
    sys.exit(0)

# ── build ────────────────────────────────────────────────────────────────
pre = json.loads((R24 / 'pseudoref_extra.json').read_text(encoding='utf-8'))
lab24 = json.loads((R24 / 'oracle_labels_extra.json').read_text(encoding='utf-8'))
base = next(iter(lab24.values()))
lab_extra = {}
for tgt in targets:
    per = lab24.get(tgt, base)
    rows = []
    for i, d in enumerate(per):
        assert d['id'] == extra[i]['id']
        coh = [pre[str(i)][str(j)][tgt][1] for j in range(1, len(d['contra']) + 1)]
        assert len(coh) == len(d['contra'])
        rows.append({'id': d['id'], 'contra': d['contra'], 'ent': d['ent'],
                     'contra_floor': d['contra_floor'], 'adq_l': coh, 'adq_r': coh,
                     'hyp_units': d['hyp_units'], 'contra_source': 'source',
                     'label_form': 'cohesion x (1 - contra)'})
    lab_extra[tgt] = rows
lab25 = {s: json.loads((R25 / f'oracle_labels_{s}.json').read_text(encoding='utf-8')) for s in ('dev', 'train', 'test')}

R26.mkdir(parents=True, exist_ok=True)
(R26 / 'data').mkdir(exist_ok=True)
new_splits = {
    'train': (split['train'], {t: lab25['train'][t] for t in targets}),
    'test_a': (split['dev'][:150] + extra[:N_A],
               {t: lab25['dev'][t][:150] + lab_extra[t][:N_A] for t in targets}),
    'test_b': (split['dev'][150:] + split['test'] + extra[N_A:],
               {t: lab25['dev'][t][150:] + lab25['test'][t] + lab_extra[t][N_A:] for t in targets}),
}
for name, (sents, lab) in new_splits.items():
    for t in targets:
        assert [d['id'] for d in lab[t]] == [s['id'] for s in sents], f'{name}/{t} 순서 불일치'
    (R26 / f'data/{name}.json').write_text(json.dumps(sents, ensure_ascii=False, indent=1), encoding='utf-8')
    (R26 / f'oracle_labels_{name}.json').write_text(json.dumps(lab, ensure_ascii=False), encoding='utf-8')
    v = [(1 - r['contra'][k]) * st.mean(lab[t][i]['adq_l'][k] for t in targets)
         for i, r in enumerate(lab[targets[0]]) for k in range(len(r['contra']))]
    print(f'[run26] {name}: {len(sents)}문장 / 경계 {len(v)} / 라벨 평균 {st.mean(v):.4f}')
ids = [s['id'] for name in new_splits for s in new_splits[name][0]]
assert len(ids) == len(set(ids)), '분할 사이에 겹치는 문장'
for fn in ('measured_profile.json', 'language_profile.json'):
    if (R25 / fn).exists():
        shutil.copy(R25 / fn, R26 / fn)
cfg26 = {**cfg, 'run_id': 'run26', 'split_from': 'en2x/en-multi/run25', 'split_scheme': 'train/test_a/test_b',
         'extra_from': 'run24 data/extra.json (seed %d)' % SEED}
(R26 / 'config.json').write_text(json.dumps(cfg26, ensure_ascii=False, indent=2), encoding='utf-8')
if not (R26 / 'cache').exists():
    (R26 / 'cache').symlink_to(Path('..') / 'run24' / 'cache')
print(f'-> {R26}')
