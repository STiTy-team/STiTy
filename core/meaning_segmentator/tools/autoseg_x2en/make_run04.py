"""x2en run04 — 판정 분할을 영어 트랙(run30) 해상도로 키운다. **라벨은 다시 계산하지 않는다.**

run03 은 200×4 + spare 105 였다. 영어는 train 500 / dev 500 / test 560 이고, 판정력이 거기서
갈린다 — dev 200 이면 한 벌 sd 가 0.0055, 500 이면 0.0035 다. 소스 문장은 있다:
소스 train+dev ∩ en train+dev, 25자 이상으로 de 1,580 / zh 1,618 / ja 1,430
(`extend_pool.py` 가 앞 905행을 건드리지 않고 뒤에 붙여 만든다).

**ja 만 1,430 이라 500/500/560(=1,560)이 안 들어간다.** 450/450/480 으로 줄인다 — dev 450 의
sd 가 0.0037 로 500(0.0035)과 실질 차이가 없다. FLEURS test 스플릿을 넣으면 1,715 가 되지만
그 분할은 STT·AST 평가용으로 봉인돼 있어 쓰지 않는다.

**라벨을 아끼는 것이 이 스크립트의 요점이다.** 오라클 라벨은 `{타깃: [{id, contra, adq_l, ...}]}`
꼴이라 문장 단위이고 이미 병합까지 끝나 있다(`label_form: cohesion x (1 - contra)`). 그래서
run03 이 라벨한 905문장은 **그대로 옮겨 담고** 새 문장만 계산한다 — de 655 / zh 655 / ja 475.
전부 다시 돌리면 언어당 GPU 2~3시간이 더 든다.

분할 구성도 역할을 지킨다. 판정에 쓰던 문장이 학습 재료로 내려오지 않게:

    train = run03 train(200) + spare(105) + 새 문장
    dev   = run03 test_a(200) + test_b(200) + 새 문장      (둘 다 판정용이었다)
    test  = run03 test(200) + 새 문장                       (최종 홀드아웃 그대로)

    PY=.venv-autoseg/bin/python; M=core.meaning_segmentator.tools.autoseg_x2en.make_run04
    $PY -m $M --lang de --phase build        # 분할·config (라벨 계산 없음)
    $PY -m $M --lang de --phase labels       # data/new.json 만 계산
    $PY -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split new \\
        --run-id x2en/de-multi/run04 --mt-cache-from x2en/de-multi/run04
    $PY -m $M --lang de --phase merge        # 새 문장 라벨 병합
    $PY -m $M --lang de --phase compose      # run03 라벨 + 새 라벨 → train/dev/test
"""
import argparse, json, shutil, sys
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import data

LANGS = {'de': dict(src_lang='German',   pool=1580, sizes=(500, 500, 560)),
         'ja': dict(src_lang='Japanese', pool=1430, sizes=(450, 450, 480)),
         'zh': dict(src_lang='Chinese',  pool=1618, sizes=(500, 500, 560))}
ALL_TARGETS = ['English', 'Chinese', 'Japanese', 'German', 'Spanish']
OLD = ('train', 'test_a', 'test_b', 'test', 'spare')
NEW_SPLIT = 'new'

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--lang', choices=sorted(LANGS), required=True)
p.add_argument('--phase', choices=('build', 'labels', 'merge', 'compose'), required=True)
p.add_argument('--from-run', default='run03')
p.add_argument('--run', default='run04')
a = p.parse_args()

spec = LANGS[a.lang]
R = RUNS_DIR / f'x2en/{a.lang}-multi/{a.run}'
R03 = RUNS_DIR / f'x2en/{a.lang}-multi/{a.from_run}'
MAN = f'evaluation/ast/manifests/fleurs_nway_{a.lang}-en_multi2en_loop{spec["pool"]}.jsonl'
targets = [t for t in ALL_TARGETS if t != spec['src_lang']]


def old_split(name):
    return json.loads((R03 / f'data/{name}.json').read_text(encoding='utf-8'))


if a.phase == 'build':
    allx = data.load(MAN)
    assert len(allx) == spec['pool'], f'{MAN}: {len(allx)}문장 (설정은 {spec["pool"]})'
    old = {s: old_split(s) for s in OLD}
    seen = {r['id'] for rows in old.values() for r in rows}
    assert len(seen) == sum(len(v) for v in old.values()), 'run03 분할 사이에 겹치는 문장'
    fresh = [x for x in data.stratified_order([x for x in allx if x.id not in seen], 20260916)]
    n_tr, n_dev, n_te = spec['sizes']
    take = [n_tr - len(old['train']) - len(old['spare']),
            n_dev - len(old['test_a']) - len(old['test_b']),
            n_te - len(old['test'])]
    assert min(take) >= 0, f'분할이 기존보다 작다: {take}'
    assert sum(take) <= len(fresh), f'새 문장 {len(fresh)} < 필요 {sum(take)}'
    # **길이를 맞춰 나눈다.** 새 문장이 옛 905보다 길고(zh 평균 73자 대 50자) 분할마다 받는
    # 비율이 다르다(train 39% / dev 20% / test 64%). 그대로 쓰면 홀드아웃이 dev 보다 길어져
    # 두 Δ 를 나란히 못 놓는다 — 문장이 길수록 절단 자리가 많아 H_set 의 성질이 달라진다.
    # 긴 것부터 **지금 평균이 가장 낮은 분할**에 하나씩 주면 세 평균이 함께 올라가 모인다.
    # 목표는 세 분할의 **평균 길이를 같게** 두는 것이다. 전체 평균을 구해 분할마다 목표 합을
    # 정하고, 긴 문장부터 **남은 자리당 부족분이 가장 큰 분할**에 준다. 단순히 "평균이 낮은
    # 쪽" 에 주면 근시안이라 마지막에 남는 짧은 문장이 한 분할에 몰린다(실측: de test 129자).
    base = [old['train'] + old['spare'], old['test_a'] + old['test_b'], old['test']]
    used = fresh[:sum(take)]
    size = [len(b) + take[k] for k, b in enumerate(base)]
    mean = (sum(len(r['text']) for b in base for r in b) + sum(len(x.text) for x in used)) / sum(size)
    tot = [sum(len(r['text']) for r in b) for b in base]
    want = [mean * size[k] for k in range(3)]
    cut = [[], [], []]
    for x in sorted(used, key=lambda x: -len(x.text)):
        j = max((k for k in range(3) if len(cut[k]) < take[k]),
                key=lambda k: (want[k] - tot[k]) / (take[k] - len(cut[k])))
        cut[j].append(x.to_dict()); tot[j] += len(x.text)
    i = sum(take)
    splits = {'train': old['train'] + old['spare'] + cut[0],
              'dev':   old['test_a'] + old['test_b'] + cut[1],
              'test':  old['test'] + cut[2]}
    new_rows = [r for c in cut for r in c]
    spare_left = len(fresh) - i
    (R / 'data').mkdir(parents=True, exist_ok=True)
    (R / 'cache').mkdir(exist_ok=True)
    for s, rows in {**splits, NEW_SPLIT: new_rows}.items():
        lens = [len(r['text']) for r in rows]
        print(f'[{a.lang}/{s}] {len(rows)}문장 / 자 평균 {sum(lens)//len(lens)} 최대 {max(lens)}')
        (R / f'data/{s}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[{a.lang}] 기존 라벨 재사용 {len(seen)} / 새로 계산 {len(new_rows)} / 분할 밖 {spare_left}')
    assert spare_left >= 20, f'프로파일 재료 20문장이 안 남는다 ({spare_left})'
    cfg = json.loads((R03 / 'config.json').read_text(encoding='utf-8'))
    cfg.update({'dataset': MAN, 'run_id': a.run, 'split_scheme': 'train/dev/test',
                'split_from': None, 'labels_from': f'x2en/{a.lang}-multi/{a.from_run}',
                'split_note': f'run03 200x4+spare105 를 옮겨 담고 새 문장 {len(new_rows)} 추가 '
                              f'(train={spec["sizes"][0]}/dev={spec["sizes"][1]}/test={spec["sizes"][2]}); '
                              f'새 문장은 stratified_order(20260916)'})
    (R / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    for f in ('measured_profile.json',):
        if (R03 / f).exists() and not (R / f).exists():
            shutil.copy(R03 / f, R / f)
    print(f'-> {R}')
    sys.exit(0)

if a.phase in ('labels', 'merge'):
    # **run03 에 위임하지 않는다.** 그쪽은 자기 905 매니페스트에서 분할을 다시 만들므로 `new` 를
    # 알 수 없다(실측으로 거기서 죽었다). 절차는 같고 대상만 `data/new.json` 이다.
    import statistics as st
    from core.meaning_segmentator.autoseg.loop import target_is_spaced
    from core.meaning_segmentator.autoseg.runtime import labels as L, metrics
    cfg = json.loads((R / 'config.json').read_text(encoding='utf-8'))
    spaced = cfg['spaced']
    rows = json.loads((R / f'data/{NEW_SPLIT}.json').read_text(encoding='utf-8'))

    if a.phase == 'labels':
        adequacy = metrics.make_adequacy_backend(cfg.get('adequacy_backend', 'cometkiwi'), batch_size=32)
        contradiction = metrics.make_contradiction_backend()
        L.compute_labels(R, NEW_SPLIT, [x['id'] for x in rows], [x['text'] for x in rows], targets,
                         spaced, adequacy, contradiction, cfg['local_mt_model'], target_is_spaced,
                         contra_source='source', translators={})
        print(f'-> {R}/oracle_labels_{NEW_SPLIT}.json (cohesion 은 pseudoref 뒤 merge)', flush=True)
        sys.exit(0)

    lab_path = R / f'oracle_labels_{NEW_SPLIT}.json'
    lab = json.loads(lab_path.read_text(encoding='utf-8'))
    if all(d.get('label_form') == 'cohesion x (1 - contra)' for d in next(iter(lab.values()))):
        print(f'[{a.lang}/{NEW_SPLIT}] 이미 합쳐진 라벨 — 건너뜀'); sys.exit(0)
    pre = json.loads((R / f'pseudoref_{NEW_SPLIT}.json').read_text(encoding='utf-8'))
    base = next(iter(lab.values()))
    out = {}
    for tgt in targets:
        per = lab.get(tgt, base)
        merged = []
        for i, d in enumerate(per):
            assert d['id'] == rows[i]['id']
            coh = [pre[str(i)][str(j)][tgt][1] for j in range(1, len(d['contra']) + 1)]
            assert len(coh) == len(d['contra'])
            merged.append({'id': d['id'], 'contra': d['contra'], 'ent': d['ent'],
                           'contra_floor': d['contra_floor'], 'adq_l': coh, 'adq_r': coh,
                           'hyp_units': d['hyp_units'], 'contra_source': 'source',
                           'label_form': 'cohesion x (1 - contra)'})
        out[tgt] = merged
    lab_path.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    v = [(1 - r['contra'][k]) * st.mean(out[t][i]['adq_l'][k] for t in targets)
         for i, r in enumerate(out[targets[0]]) for k in range(len(r['contra']))]
    print(f'[{a.lang}/{NEW_SPLIT}] {len(rows)}문장 / 경계 {len(v)} / 라벨 평균 {st.mean(v):.4f}')
    sys.exit(0)

# ── compose: run03 라벨(문장 단위) + 새 라벨 → train/dev/test ────────────────────────
pool: dict[tuple[str, str], dict] = {}
for s in (*OLD, ):
    f = R03 / f'oracle_labels_{s}.json'
    if not f.exists():
        print(f'  {f.name} 없음 — 건너뜀'); continue
    for tgt, rows in json.loads(f.read_text(encoding='utf-8')).items():
        for r in rows:
            pool[(tgt, r['id'])] = r
fnew = R / f'oracle_labels_{NEW_SPLIT}.json'
if fnew.exists():
    for tgt, rows in json.loads(fnew.read_text(encoding='utf-8')).items():
        for r in rows:
            pool[(tgt, r['id'])] = r
print(f'[{a.lang}] 문장 단위 라벨 {len(pool) // max(1, len(targets))}문장 × 타깃 {len(targets)}')
for s in ('train', 'dev', 'test'):
    ids = [r['id'] for r in json.loads((R / f'data/{s}.json').read_text(encoding='utf-8'))]
    miss = [i for i in ids if (targets[0], i) not in pool]
    assert not miss, f'{s}: 라벨 없는 문장 {len(miss)}개 (예: {miss[:3]})'
    out = {t: [pool[(t, i)] for i in ids] for t in targets}
    (R / f'oracle_labels_{s}.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print(f'-> {R}/oracle_labels_{s}.json ({len(ids)}문장 × {len(targets)}타깃)')
