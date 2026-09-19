"""짧은 지연에서 무엇이 어긋나는지 — 순위를 깊이별로 가른다. API·GPU 0.

judge20 여섯 후보의 개선이 매번 긴 지연(≤99)에 몰리고 짧은 지연(≤3)은 제자리였다. 이유가
둘 중 하나다:

    (a) 프롬프트가 **자리를 잘못 고른다** — 오라클과 다른 위치를 자른다
    (b) 자리는 비슷한데 **그 자리들이 애초에 나쁘다** — 3어절 조각의 한계

가르는 방법: 긴 지연은 순위 맨 위 1~2개만 쓰고, 짧은 지연은 중간·아래까지 소비한다.
그러니 **순위 깊이별 정확도**를 보면 된다. 깊은 곳에서만 무너지면 (a), 전 구간 고르게
낮으면 (b) 쪽이다.

점수는 judge20 의 분절 캐시에서 읽는다(v0-A, k=3 병합). 라벨은 run28 dev.

    PYTHONPATH=. .venv-autoseg/bin/python -m core.meaning_segmentator.tools.autoseg_en2x.diag.rank_depth
"""
import argparse
import hashlib
import json
import shutil
import statistics as st
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.runtime import agents_distill as ad, hset, labels as L

A = Path('core/meaning_segmentator/experiment/artifacts/en2x/en-multi')
R28, J20 = A / 'run28', A / 'judge20'
SPACED, MIN_GAP, MIN_CHUNK, MAX_K = True, 1, 2, 99
BINS = ('≤3', '≤5', '≤7', '≤10', '≤99')


def key(*p):
    return hashlib.sha256('\x1f'.join(p).encode('utf-8')).hexdigest()[:32]


def load_cache(cache_dir, name):
    """런이 아직 돌고 있으면 JsonCache 가 파일을 통째로 다시 쓴다 — 쓰는 중에 읽으면
    잘린 JSON 을 만난다. 복사본을 만들어 읽는다."""
    src = Path(cache_dir) / name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        shutil.copy(src, tmp.name)
        return json.loads(Path(tmp.name).read_text(encoding='utf-8'))


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 3:
        return None
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else None


def main() -> int:
    ap = argparse.ArgumentParser(description='순위를 깊이별로 가른다 (API·GPU 0)')
    ap.add_argument('--run', default=str(R28), help='config·data·라벨을 읽을 런 디렉토리')
    ap.add_argument('--cache-dir', default=str(J20 / 'cache'), help='분절 캐시가 있는 디렉토리')
    ap.add_argument('--prompt', default=str(J20 / 'prompt_v0.txt'), help='깊이를 잴 프롬프트 파일')
    ap.add_argument('--caches', default='segment.json,segment_s1.json,segment_s2.json',
                    help='벌별 캐시 파일 이름을 쉼표로. 여러 벌을 주면 순위 평균(k벌 병합)으로 본다')
    ap.add_argument('--split', default='dev')
    ap.add_argument('--tag', default='rank_depth', help='산출물 이름 (diag/<tag>.json)')
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    sents = json.loads((run / f'data/{args.split}.json').read_text(encoding='utf-8'))
    lab = json.loads((run / f'oracle_labels_{args.split}.json').read_text(encoding='utf-8'))
    prompt = Path(args.prompt).read_text(encoding='utf-8')
    ph = key(prompt)
    names = [x.strip() for x in args.caches.split(',') if x.strip()]
    caches = [load_cache(args.cache_dir, f) for f in names]
    print(f'[설정] 런 {run.name} / {args.split} {len(sents)}문장 / 프롬프트 '
          f'{Path(args.prompt).name} / 벌 {len(names)}개 {names}')

    hit = 0
    per_bin_overlap = {b: [] for b in BINS}
    per_bin_k = {b: [] for b in BINS}
    rho_all, rho_top, rho_bottom = [], [], []
    depth_acc = {}      # 순위 깊이 d 에서 맞춘 비율

    for i, r in enumerate(sents):
        u = L.units_of(r['text'], SPACED)
        cand = ad.candidate_positions(len(u), MIN_GAP)
        if not cand:
            continue
        marked = ad.mark_candidates(u, cand)
        k = key('seg6', ph, cfg['model'], cfg['seg_reasoning_effort'], str(cfg['batch_size']),
                marked)
        vals = [c.get(k) for c in caches]
        if not all(v and (v[0] or '').strip() for v in vals):
            continue
        hit += 1
        mr = ad.mean_rank_scores([ad.scores_of(v[0]) for v in vals])
        p_score = {j: float(x) / 100.0 for j, x in zip(cand, mr)}
        l_score = {j: L.label_value(lab, i, j) for j in cand}

        ps = [p_score[j] for j in cand]
        ls = [l_score[j] for j in cand]
        rho = spearman(ps, ls)
        if rho is not None:
            rho_all.append(rho)
            # 라벨 기준 상위 절반 / 하위 절반 안에서의 순위 일치
            order = sorted(cand, key=lambda j: -l_score[j])
            half = max(3, len(order) // 2)
            for sub, acc in ((order[:half], rho_top), (order[half:], rho_bottom)):
                if len(sub) >= 3:
                    rr = spearman([p_score[j] for j in sub], [l_score[j] for j in sub])
                    if rr is not None:
                        acc.append(rr)

        # 순위 깊이별 적중 — 프롬프트 상위 d 개가 라벨 상위 d 개와 얼마나 겹치나
        p_order = sorted(cand, key=lambda j: -p_score[j])
        l_order = sorted(cand, key=lambda j: -l_score[j])
        for d in range(1, min(13, len(cand)) + 1):
            depth_acc.setdefault(d, []).append(
                len(set(p_order[:d]) & set(l_order[:d])) / d)

        for kk in hset.k_range(len(u), MIN_CHUNK, MAX_K):
            pc = hset.top_k_cuts(u, p_score, kk, MIN_GAP)
            oc = hset.top_k_cuts(u, l_score, kk, MIN_GAP)
            if len(pc) != kk or len(oc) != kk:
                continue
            b = None
            n_units = len(u)
            for name, hi in zip(BINS, (3, 5, 7, 10, 10 ** 9)):
                if hset.chunk_len(n_units, kk) <= hi:
                    b = name
                    break
            if b is None:
                continue
            per_bin_overlap[b].append(len(set(pc) & set(oc)) / kk)
            per_bin_k[b].append(kk)

    print(f'[data] 문장 {hit}/{len(sents)} (캐시 적중)\n')
    print('구간별 — 프롬프트 절단집합이 오라클과 겹치는 비율')
    print(f"{'구간':6} {'겹침':>7} {'평균 k':>7} {'짝':>7}")
    for b in BINS:
        if per_bin_overlap[b]:
            print(f'{b:6} {st.mean(per_bin_overlap[b]):7.3f} '
                  f'{st.mean(per_bin_k[b]):7.2f} {len(per_bin_overlap[b]):7d}')
    print()
    print('순위 일치 (Spearman, 문장 평균)')
    print(f'  전체 위치        {st.mean(rho_all):+.3f}  (문장 {len(rho_all)})')
    if rho_top:
        print(f'  라벨 상위 절반   {st.mean(rho_top):+.3f}')
    if rho_bottom:
        print(f'  라벨 하위 절반   {st.mean(rho_bottom):+.3f}')
    print()
    print('순위 깊이별 적중 — 프롬프트 상위 d 개 ∩ 라벨 상위 d 개')
    print(f"{'d':>3} {'적중률':>7} {'무작위 기대':>10}")
    for d in sorted(depth_acc):
        exp = st.mean(min(1.0, d / max(1, len(ad.candidate_positions(
            len(L.units_of(s['text'], SPACED)), MIN_GAP)))) for s in sents[:200])
        print(f'{d:3d} {st.mean(depth_acc[d]):7.3f} {exp:10.3f}')
    out = A / 'diag'
    out.mkdir(exist_ok=True)
    (out / f'{args.tag}.json').write_text(json.dumps({
        'prompt': str(args.prompt), 'caches': names, 'split': args.split,
        'n_sentences': hit,
        'overlap_by_bin': {b: round(st.mean(v), 4) for b, v in per_bin_overlap.items() if v},
        'mean_k_by_bin': {b: round(st.mean(v), 2) for b, v in per_bin_k.items() if v},
        'spearman_all': round(st.mean(rho_all), 4),
        'spearman_top_half': round(st.mean(rho_top), 4) if rho_top else None,
        'spearman_bottom_half': round(st.mean(rho_bottom), 4) if rho_bottom else None,
        'depth_hit': {d: round(st.mean(v), 4) for d, v in sorted(depth_acc.items())},
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n-> {out / (args.tag + ".json")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
