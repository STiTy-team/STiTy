"""x2en run03 — judge 루프(loop_judge)용 **4분할 + 오라클 라벨** (de/ja/zh 소스, 다타깃).

타깃은 judge13(en → zh/ja/de/es)과 같은 꼴로 **소스를 뺀 4개**: zh → en/de/ja/es, ja → en/zh/de/es,
de → en/zh/ja/es. contra 는 소스 NLI 라 타깃과 무관하고, cohesion(QE)은 타깃마다 재서 평균한다.

en2x 의 run27 과 같은 꼴을 소스 언어별로 만든다. run02(옛 distill 루프)는 405문장뿐이고
오라클 라벨·`spaced`·`targets` 가 없어 `loop_judge --from-run` 이 못 읽는다.

    train  200 (사례·실측 예시) / test_a 200 (판정) / test_b 200 (체크포인트·확인) / test 200 (최종 홀드아웃)
    나머지 105 = 분할 밖 spare. loop_judge 는 그중 데이터셋 순서 앞 20 을 Writer 프로파일 재료로 쓴다.

분할은 `fleurs-{lang}-en-x`(loop905) 를 `data.stratified_order(seed)` 로 섞어 앞에서 200씩 뗀다.
FLEURS BLEU 홀드아웃(eval500)은 매니페스트에 아예 없다.

라벨은 run27 과 같은 절차: contra 는 소스 NLI(`labels.compute_labels`, contra_source=source),
cohesion 은 `pseudoref --split <s>`, 둘을 `cohesion × (1 − contra)` 꼴로 합친다. LLM 0콜, GPU 만.

    PY=.venv/bin/python; M=core.meaning_segmentator.tools.autoseg_x2en.make_run03
    $PY -m $M --lang zh --phase build      # 분할·config·measured_profile
    $PY -m $M --lang zh --phase labels     # contra·QE (기본 train,test_a,test_b — test 는 --splits test 로 나중에)
    for s in train test_a test_b; do
      $PY -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split $s \\
          --run-id x2en/zh-multi/run03 --mt-cache-from x2en/zh-multi/run03; done
    $PY -m $M --lang zh --phase merge      # 라벨 합침

체인은 `make_run03.sh` (tmux).
"""
import argparse, json, statistics as st, shutil, sys
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop import target_is_spaced
from core.meaning_segmentator.autoseg.paths import RUNS_DIR
from core.meaning_segmentator.autoseg.runtime import data, labels as L, metrics

# min_gap·격자는 run02 가 강제정렬 발화속도에서 유도한 값 그대로. judge 는 min_gap 1 로 덮어쓰고
# k 축으로 판정하므로 여기 값은 라벨 지표(evaluate)·pseudoref 분석에만 쓰인다.
LANGS = {
    'de': dict(src_lang='German',   spaced=True,  min_gap=3, t_grid=[4, 6, 12],  final_t_grid=[4, 6, 8, 12],  main_t=6,  max_units=60),
    'ja': dict(src_lang='Japanese', spaced=False, min_gap=7, t_grid=[9, 14, 27], final_t_grid=[9, 14, 18, 27], main_t=14, max_units=150),
    'zh': dict(src_lang='Chinese',  spaced=False, min_gap=6, t_grid=[8, 12, 24], final_t_grid=[8, 12, 16, 24], main_t=12, max_units=150),
}
SPLITS = ('train', 'test_a', 'test_b', 'test')
N_SPLIT, N_PROFILE, SEED = 200, 20, 20260916
ALL_TARGETS = ['English', 'Chinese', 'Japanese', 'German', 'Spanish']   # English 이 targets[0]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--lang', choices=sorted(LANGS), required=True)
p.add_argument('--phase', choices=('build', 'labels', 'merge'), required=True)
p.add_argument('--run', default='run03')
p.add_argument('--splits', default='train,test_a,test_b',
               help='labels/merge 할 분할 (쉼표). build 는 항상 4개 다 만든다. test(최종 홀드아웃)는 '
                    '판정에 안 쓰이므로 나중에 --splits test 로 따로 라벨링한다')
a = p.parse_args()

lang, spec = a.lang, LANGS[a.lang]
spaced = spec['spaced']
dataset = f'fleurs-{lang}-en-x'
targets = [t for t in ALL_TARGETS if t != spec['src_lang']]
R = RUNS_DIR / f'x2en/{lang}-multi/{a.run}'
R02 = RUNS_DIR / f'x2en/{lang}-en/run02'

allx = data.load(dataset)
order = data.stratified_order(allx, SEED)
assert len(order) == len(allx) == 905, f'{dataset}: {len(allx)}문장 (905 이어야 한다)'
splits = {s: [x.to_dict() for x in order[i * N_SPLIT:(i + 1) * N_SPLIT]] for i, s in enumerate(SPLITS)}
pool = {r['id'] for s in splits.values() for r in s}
spare = [x for x in allx if x.id not in pool]
# **분할 밖 문장도 이름 있는 분할로 라벨한다.** 판정 4분할(200×4)을 떼면 105문장이 남는데
# 지금까지 라벨이 없었다 — 이 코퍼스에서 유일하게 남은 새 문장이 그것뿐이다(정렬 1,405 중
# 0~739·1240~1404 가 loop905, 740~1239 는 BLEU 홀드아웃이라 건드리지 않는다).
# `pool`·`spare` 를 센 **뒤에** 넣는다 — 분할 겹침 검사와 프로파일 재료 검사가 4분할 기준이다.
# `loop_judge` 는 자기가 읽은 분할로만 `pool_ids` 를 만들므로 이 파일이 생겨도 프로파일 재료
# (분할 밖 문장 앞 20)는 움직이지 않는다.
SPARE = 'spare'
splits[SPARE] = [x.to_dict() for x in order[N_SPLIT * len(SPLITS):]]
todo = [s for s in (*SPLITS, SPARE) if s in a.splits.split(',')]
assert todo, f'--splits {a.splits!r}: {(*SPLITS, SPARE)} 중 하나여야 한다'


def n_units(text):
    return len(L.units_of(text, spaced))


if a.phase == 'build':
    assert len(pool) == N_SPLIT * len(SPLITS), '분할 사이에 겹치는 문장'
    assert {r['id'] for r in splits[SPARE]} == {x.id for x in spare}, 'spare 가 분할 밖 문장과 다르다'
    assert len(spare) >= N_PROFILE, f'분할 밖 문장 {len(spare)} < 프로파일 재료 {N_PROFILE}'
    unit = '어절' if spaced else '자'
    for s, rows in splits.items():
        lens = [n_units(r['text']) for r in rows]
        print(f'[{lang}/{s}] {len(rows)}문장 / {unit} 평균 {st.mean(lens):.1f} 최대 {max(lens)} 최소 {min(lens)}')
        assert max(lens) <= spec['max_units'], f'{s}: 이상치 {max(lens)}{unit} — 라벨 전에 데이터 위생 확인'
    print(f'[{lang}] spare {len(spare)} (프로파일 재료 앞 {N_PROFILE})')
    (R / 'data').mkdir(parents=True, exist_ok=True)
    (R / 'cache').mkdir(exist_ok=True)
    for s, rows in splits.items():
        path = R / f'data/{s}.json'
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8')) == rows, f'{s}.json 이 다른 표본이다'
        else:
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    # 실측 프로파일(단위·구두점 통계)은 run02 것 — 같은 코퍼스, 같은 언어. Writer 의 measured_facts 재료.
    if not (R / 'measured_profile.json').exists():
        shutil.copy(R02 / 'measured_profile.json', R / 'measured_profile.json')
    cfg = {
        'dataset': dataset, 'src_lang': spec['src_lang'],
        'pair_id': f'x2en/{lang}-multi', 'run_id': a.run,
        'model': 'gpt-5-mini', 'provider': 'openai', 'base_url': None,
        'seg_reasoning_effort': 'medium', 'agent_reasoning_effort': 'medium',
        'batch_size': 6, 'contra_source': 'source', 'k_samples': 3, 'workers': 16,
        'local_mt_model': 'google/madlad400-3b-mt', 'adequacy_backend': 'cometkiwi', 'comet_batch_size': 64,
        'min_gap': spec['min_gap'],
        'units_per_sec_source': f'alignment:fleurs_nway_{lang}-en_multi2en_loop405_unittimes.json',
        't_grid': spec['t_grid'], 'final_t_grid': spec['final_t_grid'], 'main_t': spec['main_t'],
        'seed': SEED, 'targets': targets, 'spaced': spaced, 'mode': 'judge',
        'translator_id': 'local:google/madlad400-3b-mt:en:ctx=False',
        'split_scheme': 'train/test_a/test_b/test (+spare)', 'split_from': None, 'labels_from': None,
        'split_note': f'stratified_order(seed {SEED}) over {dataset} (905) → 200×4, spare {len(spare)}',
        'measured_profile_from': f'x2en/{lang}-en/run02',
    }
    (R / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'-> {R}')
    sys.exit(0)

if a.phase == 'labels':
    cfg = json.loads((R / 'config.json').read_text(encoding='utf-8'))
    adequacy = metrics.make_adequacy_backend(cfg.get('adequacy_backend', 'cometkiwi'), batch_size=32)
    contradiction = metrics.make_contradiction_backend()
    translators: dict = {}          # 분할 사이에 번역기를 공유 — 모델을 네 번 올리지 않는다
    for s in todo:
        rows = splits[s]
        L.compute_labels(R, s, [x['id'] for x in rows], [x['text'] for x in rows], targets,
                         spaced, adequacy, contradiction, cfg['local_mt_model'], target_is_spaced,
                         contra_source='source', translators=translators)
        print(f'-> {R}/oracle_labels_{s}.json (contra 소스 NLI, QE 조각별 — cohesion 은 pseudoref 뒤 merge)', flush=True)
    sys.exit(0)

# ── merge: pseudoref 의 cohesion 을 라벨에 합친다 (make_run27 merge 와 같은 꼴, 분할 4개) ─────
for s in todo:
    rows = splits[s]
    lab_path = R / f'oracle_labels_{s}.json'
    lab = json.loads(lab_path.read_text(encoding='utf-8'))
    if all(d.get('label_form') == 'cohesion x (1 - contra)' for d in next(iter(lab.values()))):
        print(f'[{lang}/{s}] 이미 합쳐진 라벨 — 건너뜀')
        continue
    pre = json.loads((R / f'pseudoref_{s}.json').read_text(encoding='utf-8'))
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
    print(f'[{lang}/{s}] {len(rows)}문장 / 경계 {len(v)} / 라벨 평균 {st.mean(v):.4f}')
print(f'-> {R}')
