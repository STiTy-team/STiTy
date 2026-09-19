"""run28 — run27 에 **v0 후보 선별용 분할과 넓힌 최종 홀드아웃**을 더한다.

run27 의 최종 test 는 200문장이라 짝 비교 반폭이 0.013 이었다. 실측한 분절 비결정론이
그 중 0.0103 을 차지하고 프롬프트 간 진짜 차이는 0.0080 이라, 200문장으로는 루프가
만드는 폭을 잡음과 못 가른다 (`artifacts/en2x/en-multi/noise/sample_noise.json`).

    sel   300  v0 후보 선별 — 여기서 고르고 test 에서 보고한다 (승자의 저주 차단)
    test  660  run27 test 200 + 새 460. 반폭 0.0072
    train / test_a / test_b   run27 그대로 (루프의 사례·dev)

새 문장은 `fleurs_nway_en-de_multi_new760.jsonl` 에서 온다 — FLEURS en train+dev 를
**de 와의 2언어 교집합**까지만 좁힌 1,580문장 풀에서 run27 이 쓴 820(4분할 800 + Writer
프로파일 20)을 뺀 760문장이다. 4언어 교집합(1,405)을 고집할 이유가 없다: 오라클 라벨은
참조 번역을 안 쓴다 (번역 madlad, cohesion CometKiwi, contra 소스 NLI — LLM 0콜).

**test 660 의 라벨은 앞 200 을 포함해 전부 새로 만든다.** run27 것을 이어붙이면 인덱스가
어긋나기 쉽고, 겹치는 200문장이 run27 라벨과 얼마나 같은지가 그대로 드리프트 점검이 된다.

    PY=.venv-autoseg/bin/python
    $PY -m core.meaning_segmentator.tools.autoseg_en2x.run28.make_run28 --phase build
    $PY -m core.meaning_segmentator.tools.autoseg_en2x.run28.make_run28 --phase labels --split sel
    $PY -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split sel \
        --run-id en2x/en-multi/run28 --mt-cache-from en2x/en-multi/run28
    $PY -m core.meaning_segmentator.tools.autoseg_en2x.run28.make_run28 --phase merge --split sel
    (test 도 같은 세 단계)
"""
import argparse
import json
import shutil
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop import target_is_spaced
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import labels as L, metrics

R27 = RUNS_DIR / 'en2x/en-multi/run27'
R28 = RUNS_DIR / 'en2x/en-multi/run28'
MANIFEST = Path('evaluation/ast/manifests/fleurs_nway_en-de_multi_new760.jsonl')
N_SEL, N_TEST_NEW = 300, 460

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--phase', choices=('build', 'labels', 'merge'), required=True)
p.add_argument('--split', choices=('sel', 'test'))
a = p.parse_args()

cfg = json.loads((R27 / 'config.json').read_text(encoding='utf-8'))
spaced, targets = cfg['spaced'], cfg['targets']


def new_rows() -> list[dict]:
    """매니페스트 순서 그대로. 빌더가 이미 길이 층화 순서로 깔아 두므로 앞에서 연속으로
    떼어도 길이 분포가 치우치지 않는다."""
    rows = [json.loads(l) for l in MANIFEST.open(encoding='utf-8')]
    return [{'id': r['utt_id'], 'text': r['src_text']} for r in rows]


if a.phase == 'build':
    assert MANIFEST.exists(), f'매니페스트 없음: {MANIFEST}'
    new = new_rows()
    assert len(new) >= N_SEL + N_TEST_NEW, f'문장 부족 {len(new)}'
    old_test = json.loads((R27 / 'data/test.json').read_text(encoding='utf-8'))
    sel = new[:N_SEL]
    test = old_test + new[N_SEL:N_SEL + N_TEST_NEW]

    for name, rows in (('sel', sel), ('test', test)):
        lens = [len(r['text'].split()) for r in rows]
        print(f'[{name}] {len(rows)}문장 / 어절 평균 {st.mean(lens):.1f} '
              f'최대 {max(lens)} 최소 {min(lens)}')
        assert max(lens) <= 60, f'{name}: 이상치 — 라벨 전에 데이터 위생 확인'

    (R28 / 'data').mkdir(parents=True, exist_ok=True)
    for s in ('train', 'test_a', 'test_b'):
        shutil.copy(R27 / f'data/{s}.json', R28 / f'data/{s}.json')
        shutil.copy(R27 / f'oracle_labels_{s}.json', R28 / f'oracle_labels_{s}.json')
    for fn in ('measured_profile.json', 'language_profile.json'):
        if (R27 / fn).exists():
            shutil.copy(R27 / fn, R28 / fn)
    for name, rows in (('sel', sel), ('test', test)):
        (R28 / f'data/{name}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                               encoding='utf-8')
    # 번역 캐시는 **복사한다.** 심볼릭 링크로 묶으면 judge 런과 동시에 돌 때 서로 덮어쓴다.
    if not (R28 / 'cache').exists():
        (R28 / 'cache').mkdir()
        for f in R27.glob('cache/translate_*.json'):
            shutil.copy(f, R28 / 'cache' / f.name)

    ids = [r['id'] for s in ('train', 'test_a', 'test_b')
           for r in json.loads((R28 / f'data/{s}.json').read_text(encoding='utf-8'))]
    ids += [r['id'] for r in sel] + [r['id'] for r in test]
    assert len(ids) == len(set(ids)), '분할 사이에 겹치는 문장'
    cfg28 = {**cfg, 'run_id': 'run28', 'split_from': 'en2x/en-multi/run27',
             'split_scheme': 'train/test_a/test_b/sel/test',
             'sel_from': f'{MANIFEST.name} [0:{N_SEL}]',
             'test_from': f'run27 test 200 + {MANIFEST.name} [{N_SEL}:{N_SEL + N_TEST_NEW}]'}
    (R28 / 'config.json').write_text(json.dumps(cfg28, ensure_ascii=False, indent=2),
                                     encoding='utf-8')
    print(f'-> {R28} (sel {len(sel)} / test {len(test)})')
    sys.exit(0)

assert a.split, '--split 이 필요하다'
rows = json.loads((R28 / f'data/{a.split}.json').read_text(encoding='utf-8'))

if a.phase == 'labels':
    adequacy = metrics.make_adequacy_backend(cfg.get('adequacy_backend', 'cometkiwi'),
                                             batch_size=32)
    contradiction = metrics.make_contradiction_backend()
    L.compute_labels(R28, a.split, [x['id'] for x in rows], [x['text'] for x in rows], targets,
                     spaced, adequacy, contradiction, cfg['local_mt_model'], target_is_spaced,
                     contra_source='source')
    print(f'-> {R28}/oracle_labels_{a.split}.json (contra 소스 NLI, cohesion 은 merge 에서)')
    sys.exit(0)

# ── merge: pseudoref 의 cohesion 을 라벨에 합친다
pre = json.loads((R28 / f'pseudoref_{a.split}.json').read_text(encoding='utf-8'))
lab = json.loads((R28 / f'oracle_labels_{a.split}.json').read_text(encoding='utf-8'))
assert all(d.get('label_form') != 'cohesion x (1 - contra)'
           for d in next(iter(lab.values()))), '이미 합쳐진 라벨'
base = next(iter(lab.values()))
out = {}
for tgt in targets:
    per = lab.get(tgt, base)
    made = []
    for i, d in enumerate(per):
        assert d['id'] == rows[i]['id']
        coh = [pre[str(i)][str(j)][tgt][1] for j in range(1, len(d['contra']) + 1)]
        assert len(coh) == len(d['contra'])
        made.append({'id': d['id'], 'contra': d['contra'], 'ent': d['ent'],
                     'contra_floor': d['contra_floor'], 'adq_l': coh, 'adq_r': coh,
                     'hyp_units': d['hyp_units'], 'contra_source': 'source',
                     'label_form': 'cohesion x (1 - contra)'})
    out[tgt] = made
(R28 / f'oracle_labels_{a.split}.json').write_text(json.dumps(out, ensure_ascii=False),
                                                   encoding='utf-8')
v = [(1 - r['contra'][k]) * st.mean(out[t][i]['adq_l'][k] for t in targets)
     for i, r in enumerate(out[targets[0]]) for k in range(len(r['contra']))]
print(f'[run28] {a.split}: {len(rows)}문장 / 경계 {len(v)} / 라벨 평균 {st.mean(v):.4f}')

# 겹치는 200문장이 run27 라벨과 얼마나 같은지 — 드리프트 점검
if a.split == 'test':
    old = json.loads((R27 / 'oracle_labels_test.json').read_text(encoding='utf-8'))
    o0, n0 = old[targets[0]], out[targets[0]]
    diffs = [abs(a_ - b_) for i in range(len(o0))
             for a_, b_ in zip(o0[i]['contra'], n0[i]['contra'])]
    assert o0[0]['id'] == n0[0]['id'], '앞 200 이 run27 test 순서와 다르다'
    print(f'[드리프트] run27 과 겹치는 {len(o0)}문장 contra 절대차: '
          f'평균 {st.mean(diffs):.5f} 최대 {max(diffs):.5f}')
print(f'-> {R28}')
