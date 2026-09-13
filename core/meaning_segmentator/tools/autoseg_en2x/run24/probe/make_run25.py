"""새 오라클(`cohesion × (1 − 소스 contra)`)을 라벨 파일 형식으로 굽고 run25 를 만든다.

`rule_probe`·`loop_distill` 은 런 디렉토리의 `oracle_labels_<split>.json` 을 정답으로 읽고
`labels.label_value` 로 `(1 − contra) × (adq_l + adq_r)/2` 를 계산한다. 그래서 **adq_l 과
adq_r 자리에 cohesion 을 넣으면** 그 식이 그대로 `(1 − contra) × cohesion` 이 된다. 읽는
쪽 코드를 안 고치고 새 라벨로 갈아끼우는 방법이다 (`adq_l`/`adq_r` 을 따로 쓰는 곳은
`component_eval` 같은 분석 도구뿐이고, 거기서는 둘이 같은 값으로 보인다).

분할·설정은 run24 것을 그대로 복사하고 캐시는 심볼릭 링크한다 — 캐시 키에 프롬프트 해시가
들어가므로 기존 프롬프트(v0, minimal_tgt)의 dev 채점은 전부 적중해 공짜다.

    .venv/bin/python -m ...probe.make_run25
"""
import json, shutil, statistics as st, sys
from pathlib import Path
sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.paths import RUNS_DIR

SRC = RUNS_DIR / 'en2x/en-multi/run24'
DST = RUNS_DIR / 'en2x/en-multi/run25'
DST.mkdir(parents=True, exist_ok=True)
for fn in ('config.json', 'measured_profile.json', 'language_profile.json'):
    if (SRC / fn).exists():
        shutil.copy(SRC / fn, DST / fn)
(DST / 'data').mkdir(exist_ok=True)
for split in ('train', 'dev', 'test'):
    shutil.copy(SRC / f'data/{split}.json', DST / f'data/{split}.json')
if not (DST / 'cache').exists():
    (DST / 'cache').symlink_to(Path('..') / 'run24' / 'cache')

for split in ('dev', 'test', 'train'):
    pre_path = SRC / f'pseudoref_{split}.json'
    if not pre_path.exists():
        print(f'[skip] {pre_path.name} 없음'); continue
    pre = json.loads(pre_path.read_text(encoding='utf-8'))
    lab = json.loads((SRC / f'oracle_labels_{split}.json').read_text(encoding='utf-8'))
    out = {}
    n_pos = 0
    for tgt, per in lab.items():
        rows = []
        for i, d in enumerate(per):
            coh = [pre[str(i)][str(j)][tgt][1] for j in range(1, len(d['contra']) + 1)]
            assert len(coh) == len(d['contra']), f'{tgt} {d["id"]}: {len(coh)} vs {len(d["contra"])}'
            rows.append({'id': d['id'], 'contra': d['contra'], 'ent': d['ent'],
                         'contra_floor': d['contra_floor'], 'adq_l': coh, 'adq_r': coh,
                         'hyp_units': d['hyp_units'], 'contra_source': 'source',
                         'label_form': 'cohesion x (1 - contra)'})
            n_pos += len(coh)
        out[tgt] = rows
    (DST / f'oracle_labels_{split}.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    v = [ (1 - r['contra'][k]) * st.mean(out[t][i]['adq_l'][k] for t in out)
          for i, r in enumerate(out[list(out)[0]]) for k in range(len(r['contra'])) ]
    print(f'[run25] {split}: 경계 {n_pos // len(out)} / 라벨 평균 {st.mean(v):.4f} '
          f'(min {min(v):.3f} max {max(v):.3f})')
print(f'-> {DST}')
