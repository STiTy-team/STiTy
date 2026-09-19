"""run28 분할을 최종 배분으로 다시 깐다 — 라벨은 이미 있는 것을 잘라 붙인다 (GPU 0).

    train 200                                사례·실측 예시 (run27 그대로)
    sel   300 = train 200 + 새 [0:100]       v0 후보 선별
    dev   600 = test_a 200 + test_b 200 + 새 [100:300]   개정 채택 판정
    test  660 = run27 test 200 + 새 [300:760]            최종 홀드아웃 (이미 완성)

라벨을 런 사이에 섞어도 되는 근거: `label_value` 는 `contra` 와 `adq_l/adq_r` 만 읽는다.
런마다 다른 값은 `contra_floor` 하나인데 채점에 안 쓰인다. 그리고 run27 과 겹치는
200문장을 run28 에서 다시 만들어 본 결과 contra 절대차가 평균 0.00000 / 최대 0.00200
이었다 — 계산이 재현된다.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.paths import RUNS_DIR

R27 = RUNS_DIR / 'en2x/en-multi/run27'
R28 = RUNS_DIR / 'en2x/en-multi/run28'
N_SEL_NEW = 100


def rows(run, name):
    return json.loads((run / f'data/{name}.json').read_text(encoding='utf-8'))


def labs(run, name):
    return json.loads((run / f'oracle_labels_{name}.json').read_text(encoding='utf-8'))


def cat_labels(parts: list[dict]) -> dict:
    tgts = list(parts[0])
    assert all(set(p) == set(tgts) for p in parts), '타깃 집합이 다르다'
    return {t: [r for p in parts for r in p[t]] for t in tgts}


def write(name, rs, ls):
    (R28 / f'data/{name}.json').write_text(json.dumps(rs, ensure_ascii=False, indent=1),
                                           encoding='utf-8')
    (R28 / f'oracle_labels_{name}.json').write_text(json.dumps(ls, ensure_ascii=False),
                                                    encoding='utf-8')
    t0 = next(iter(ls))
    assert len(ls[t0]) == len(rs), f'{name}: 라벨 {len(ls[t0])} != 문장 {len(rs)}'
    for i, r in enumerate(rs):
        assert ls[t0][i]['id'] == r['id'], f'{name}: {i}번째 라벨 id 어긋남'
    print(f'[{name}] {len(rs)}문장')


new_rows, new_labs = rows(R28, 'sel'), labs(R28, 'sel')          # 새 [0:300]
tr_rows, tr_labs = rows(R27, 'train'), labs(R27, 'train')
a_rows, a_labs = rows(R27, 'test_a'), labs(R27, 'test_a')
b_rows, b_labs = rows(R27, 'test_b'), labs(R27, 'test_b')

# 새 sel 을 덮어쓰기 전에 원본을 남긴다 — 다시 만들려면 GPU 70분이다
shutil.copy(R28 / 'data/sel.json', R28 / 'data/_new300.json')
shutil.copy(R28 / 'oracle_labels_sel.json', R28 / 'oracle_labels__new300.json')

def cut(d, a, b):
    return {t: per[a:b] for t, per in d.items()}

write('sel', tr_rows + new_rows[:N_SEL_NEW],
      cat_labels([tr_labs, cut(new_labs, 0, N_SEL_NEW)]))
write('dev', a_rows + b_rows + new_rows[N_SEL_NEW:],
      cat_labels([a_labs, b_labs, cut(new_labs, N_SEL_NEW, None)]))
shutil.copy(R27 / 'data/train.json', R28 / 'data/train.json')
shutil.copy(R27 / 'oracle_labels_train.json', R28 / 'oracle_labels_train.json')

ids = {}
for name in ('train', 'sel', 'dev', 'test'):
    ids[name] = [r['id'] for r in rows(R28, name)]
print(f"[겹침] sel∩dev {len(set(ids['sel']) & set(ids['dev']))} / "
      f"sel∩test {len(set(ids['sel']) & set(ids['test']))} / "
      f"dev∩test {len(set(ids['dev']) & set(ids['test']))} / "
      f"train⊂sel {set(ids['train']) <= set(ids['sel'])}")
cfg = json.loads((R28 / 'config.json').read_text(encoding='utf-8'))
cfg['split_scheme'] = 'train/sel/dev/test'
cfg['sel_from'] = 'run27 train 200 + new760 [0:100]'
cfg['dev_from'] = 'run27 test_a 200 + test_b 200 + new760 [100:300]'
(R28 / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
print('-> config.json 갱신')
