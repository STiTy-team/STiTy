"""run27 — run26 의 3분할에 **최종 홀드아웃 test 200** 을 더한다.

run26 은 test-B 가 체크포인트·유망 확인·최종 표를 겸해, 완전히 안 본 집합이 없었다. 채택본과
v0 의 차이를 아무 판정에도 안 쓴 문장에서 재야 "이 런이 올린 폭" 이 남는다.

    train  200 / test_a 200 / test_b 200 = run26 그대로
    test   200 = 어느 분할에도 없고 v0 프로파일 재료(분할 밖 앞 20문장)도 아닌 285문장에서 시드로

라벨은 run26 추가분과 같은 절차: contra 는 소스 NLI(`labels.compute_labels`, contra_source=source),
cohesion 은 `pseudoref --split test`, 둘을 `cohesion × (1 − contra)` 꼴로 합친다. LLM 0콜, GPU 만.
번역 캐시는 run24 것을 **복사해** 쓴다 — 같은 파일을 judge 런이 동시에 쓰고 있어 심볼릭 링크면
서로 덮어쓴다.

    PY=.venv/bin/python
    $PY -m core.meaning_segmentator.tools.autoseg_en2x.run27.make_run27 --phase build     # 분할·복사
    $PY -m core.meaning_segmentator.tools.autoseg_en2x.run27.make_run27 --phase labels    # contra·QE
    $PY -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split test \
        --run-id en2x/en-multi/run27 --mt-cache-from en2x/en-multi/run27
    $PY -m core.meaning_segmentator.tools.autoseg_en2x.run27.make_run27 --phase merge     # 라벨 합침
"""
import argparse, json, random, shutil, statistics as st, sys
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop import target_is_spaced
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import data, labels as L, metrics

R24 = RUNS_DIR / 'en2x/en-multi/run24'
R26 = RUNS_DIR / 'en2x/en-multi/run26'
R27 = RUNS_DIR / 'en2x/en-multi/run27'
N_TEST, N_PROFILE, SEED = 200, 20, 20260916

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--phase', choices=('build', 'labels', 'merge'), required=True)
a = p.parse_args()

cfg = json.loads((R26 / 'config.json').read_text(encoding='utf-8'))
spaced, targets = cfg['spaced'], cfg['targets']
splits = {s: json.loads((R26 / f'data/{s}.json').read_text(encoding='utf-8'))
          for s in ('train', 'test_a', 'test_b')}
pool = {r['id'] for s in splits.values() for r in s}
spare = [x for x in data.load(cfg['dataset']) if x.id not in pool]     # 데이터셋 순서 — loop_judge 의 spare 와 같다
cand = spare[N_PROFILE:]            # 앞 20 은 Writer 프로파일 재료
rng = random.Random(SEED)
rng.shuffle(cand)
test = [x.to_dict() for x in cand[:N_TEST]]
test_path = R27 / 'data/test.json'

if a.phase == 'build':
    assert len(cand) >= N_TEST, f'분할 밖 문장 {len(cand)} < {N_TEST}'
    lens = [len(x['text'].split()) for x in test]
    print(f'[test] {len(test)}문장 / 어절 평균 {st.mean(lens):.1f} 최대 {max(lens)} 최소 {min(lens)}')
    assert max(lens) <= 60, '이상치 — 라벨 전에 데이터 위생 확인'
    (R27 / 'data').mkdir(parents=True, exist_ok=True)
    for s in splits:
        shutil.copy(R26 / f'data/{s}.json', R27 / f'data/{s}.json')
        shutil.copy(R26 / f'oracle_labels_{s}.json', R27 / f'oracle_labels_{s}.json')
    for fn in ('measured_profile.json', 'language_profile.json'):
        if (R26 / fn).exists():
            shutil.copy(R26 / fn, R27 / fn)
    if test_path.exists():
        assert json.loads(test_path.read_text(encoding='utf-8')) == test, 'test.json 이 다른 표본이다'
    else:
        test_path.write_text(json.dumps(test, ensure_ascii=False, indent=1), encoding='utf-8')
    if not (R27 / 'cache').exists():
        (R27 / 'cache').mkdir()
        for f in R24.glob('cache/translate_*.json'):
            shutil.copy(f, R27 / 'cache' / f.name)
    cfg27 = {**cfg, 'run_id': 'run27', 'split_from': 'en2x/en-multi/run26',
             'split_scheme': 'train/test_a/test_b/test',
             'test_from': 'dataset minus run26 splits minus first %d spare (seed %d)' % (N_PROFILE, SEED)}
    (R27 / 'config.json').write_text(json.dumps(cfg27, ensure_ascii=False, indent=2), encoding='utf-8')
    ids = [r['id'] for s in splits.values() for r in s] + [r['id'] for r in test]
    assert len(ids) == len(set(ids)), '분할 사이에 겹치는 문장'
    print(f'-> {R27} (test {len(test)}, 캐시 복사 {len(list((R27 / "cache").glob("*.json")))}개)')
    sys.exit(0)

if a.phase == 'labels':
    adequacy = metrics.make_adequacy_backend(cfg.get('adequacy_backend', 'cometkiwi'), batch_size=32)
    contradiction = metrics.make_contradiction_backend()
    L.compute_labels(R27, 'test', [x['id'] for x in test], [x['text'] for x in test], targets,
                     spaced, adequacy, contradiction, cfg['local_mt_model'], target_is_spaced,
                     contra_source='source')
    print(f'-> {R27}/oracle_labels_test.json (contra 소스 NLI, QE 조각별 — cohesion 은 pseudoref 뒤 merge)')
    sys.exit(0)

# ── merge: pseudoref 의 cohesion 을 라벨에 합친다 (make_run26 build 와 같은 꼴) ─────────
pre = json.loads((R27 / 'pseudoref_test.json').read_text(encoding='utf-8'))
lab = json.loads((R27 / 'oracle_labels_test.json').read_text(encoding='utf-8'))
assert all(d.get('label_form') != 'cohesion x (1 - contra)' for d in next(iter(lab.values()))), '이미 합쳐진 라벨'
base = next(iter(lab.values()))
out = {}
for tgt in targets:
    per = lab.get(tgt, base)
    rows = []
    for i, d in enumerate(per):
        assert d['id'] == test[i]['id']
        coh = [pre[str(i)][str(j)][tgt][1] for j in range(1, len(d['contra']) + 1)]
        assert len(coh) == len(d['contra'])
        rows.append({'id': d['id'], 'contra': d['contra'], 'ent': d['ent'],
                     'contra_floor': d['contra_floor'], 'adq_l': coh, 'adq_r': coh,
                     'hyp_units': d['hyp_units'], 'contra_source': 'source',
                     'label_form': 'cohesion x (1 - contra)'})
    out[tgt] = rows
(R27 / 'oracle_labels_test.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
v = [(1 - r['contra'][k]) * st.mean(out[t][i]['adq_l'][k] for t in targets)
     for i, r in enumerate(out[targets[0]]) for k in range(len(r['contra']))]
print(f'[run27] test: {len(test)}문장 / 경계 {len(v)} / 라벨 평균 {st.mean(v):.4f}')
for s in ('train', 'test_a', 'test_b'):
    lab_s = json.loads((R27 / f'oracle_labels_{s}.json').read_text(encoding='utf-8'))
    vs = [(1 - r['contra'][k]) * st.mean(lab_s[t][i]['adq_l'][k] for t in targets)
          for i, r in enumerate(lab_s[targets[0]]) for k in range(len(r['contra']))]
    print(f'[run27] {s}: {len(lab_s[targets[0]])}문장 / 경계 {len(vs)} / 라벨 평균 {st.mean(vs):.4f}')
print(f'-> {R27}')
