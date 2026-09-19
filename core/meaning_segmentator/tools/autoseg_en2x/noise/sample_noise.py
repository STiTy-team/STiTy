"""분절 비결정론을 H_set 단위로 잰다 — API 호출 0.

run27 test 200문장에 대한 v0 프롬프트의 분절이 캐시에 세 벌 있다
(`segment.json` / `segment_s1.json` / `segment_s2.json`). 세 벌을 각각 따로
채점하면 **같은 프롬프트를 k=1 로 세 번 잰 것**과 같다. 진짜 차이가 0 인 비교라,
거기서 나오는 폭이 곧 k=1 채점의 잡음이다.

번역은 로컬 madlad, 일관성은 CometKiwi — GPU 만 쓰고 돈은 안 든다.

    tmux new-session -d -s segnoise -c <저장소> \
      ".venv-autoseg/bin/python -u -m core.meaning_segmentator.tools.autoseg_en2x.noise.sample_noise"
"""
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.loop import target_is_spaced
from core.meaning_segmentator.autoseg.runtime import agents_distill as ad
from core.meaning_segmentator.autoseg.runtime import hset, labels as L, metrics
from core.meaning_segmentator.autoseg.runtime.pipeline import JsonCache, LocalTranslator

A = Path('core/meaning_segmentator/experiment/artifacts/en2x/en-multi')
R27 = A / 'run27'
OUT = A / 'noise'
SPACED, MIN_GAP, MIN_CHUNK, MAX_K = True, 1, 2, 99


def to_lang_code(name: str) -> str:
    return {'Chinese': 'zh', 'Japanese': 'ja', 'German': 'de', 'Spanish': 'es'}[name]


def cache_key(*parts: str) -> str:
    return hashlib.sha256('\x1f'.join(parts).encode('utf-8')).hexdigest()[:32]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((R27 / 'config.json').read_text(encoding='utf-8'))
    targets = cfg['targets']
    test = json.loads((R27 / 'data/test.json').read_text(encoding='utf-8'))
    lab = json.loads((R27 / 'oracle_labels_test.json').read_text(encoding='utf-8'))

    prompt = (A / 'judge19/prompt_v0.txt').read_text(encoding='utf-8')
    ph = cache_key(prompt)
    caches = [json.load(open(R27 / 'cache' / f, encoding='utf-8'))
              for f in ('segment.json', 'segment_s1.json', 'segment_s2.json')]

    # ── 샘플별 절단집합
    sets = [{}, {}, {}]
    merged = {}
    for i, r in enumerate(test):
        u = L.units_of(r['text'], SPACED)
        cand = ad.candidate_positions(len(u), MIN_GAP)
        if not cand:
            continue
        marked = ad.mark_candidates(u, cand)
        k = cache_key('seg6', ph, cfg['model'], cfg['seg_reasoning_effort'],
                      str(cfg['batch_size']), marked)
        hits = [c.get(k) for c in caches]
        if not all(h and (h[0] or '').strip() for h in hits):
            print(f'[skip] 캐시 미스 {r["id"]}')
            continue
        samples = [ad.scores_of(h[0]) for h in hits]
        per = [{j: float(x) / 100.0 for j, x in zip(cand, s)} for s in samples]
        mr = ad.mean_rank_scores(samples)
        scm = {j: float(x) / 100.0 for j, x in zip(cand, mr)}
        for kk in hset.k_range(len(u), MIN_CHUNK, MAX_K):
            for s in range(3):
                cut = hset.top_k_cuts(u, per[s], kk, MIN_GAP)
                if len(cut) == kk:
                    sets[s][(i, kk)] = cut
            cm = hset.top_k_cuts(u, scm, kk, MIN_GAP)
            if len(cm) == kk:
                merged[(i, kk)] = cm

    keys = sorted(set(sets[0]) & set(sets[1]) & set(sets[2]) & set(merged))
    print(f'[data] 짝 {len(keys)} / 문장 {len({i for i, _ in keys})}', flush=True)

    # ── 채점기 (GPU, 돈 안 듦). 캐시는 run27 것을 그대로 쓴다 — 조각 번역이 대부분 이미 있다.
    translators = {t: LocalTranslator(tgt_code=to_lang_code(t),
                                      cache=JsonCache(R27 / 'cache' /
                                                      f'translate_{to_lang_code(t)}.json'))
                   for t in targets}
    scorer = hset.HsetScorer(
        translators=translators,
        qe=metrics.make_adequacy_backend(cfg.get('adequacy_backend', 'cometkiwi'),
                                         batch_size=cfg.get('comet_batch_size', 64)),
        spaced=SPACED, target_spaced={t: target_is_spaced(t) for t in targets})

    texts = [s['text'] for s in test]

    def contra_of(i, j):
        return lab[targets[0]][i]['contra'][j - 1]

    hs = []
    for tag, S in [('s0', sets[0]), ('s1', sets[1]), ('s2', sets[2]), ('k3', merged)]:
        vals = scorer.score(texts, [(i, S[(i, T)]) for i, T in keys], contra_of)
        h = dict(zip(keys, vals))
        hs.append(h)
        for tr in translators.values():
            if tr.cache is not None:
                tr.cache.flush()
        print(f'[{tag}] H_set {st.mean(h.values()):.4f}', flush=True)

    # ── 잡음 폭: 같은 프롬프트의 두 샘플을 짝 비교하면 진짜 차이가 0 이다
    rep = {'n_pairs': len(keys), 'n_clusters': len({i for i, _ in keys}),
           'means': {t: round(st.mean(h.values()), 4)
                     for t, h in zip(('s0', 's1', 's2', 'k3'), hs)}}
    clusters = [i for i, _ in keys]
    null = []
    for a_i, b_i in ((0, 1), (0, 2), (1, 2)):
        b = hset.paired_bootstrap([hs[a_i][k] for k in keys], [hs[b_i][k] for k in keys],
                                  clusters=clusters)
        null.append(b)
        print(f'[null s{a_i} vs s{b_i}] Δ {b["mean"]:+.4f} [{b["lo"]:+.4f}, {b["hi"]:+.4f}]',
              flush=True)
    rep['null_k1'] = null
    rep['null_k1_halfwidth_mean'] = round(st.mean((b['hi'] - b['lo']) / 2 for b in null), 4)

    # 문장·지연축별 샘플 간 표준편차 = σ_w
    sw = [st.pstdev([hs[s][k] for s in range(3)]) for k in keys]
    rep['sigma_w_per_pair'] = round(st.mean(sw), 4)
    print(f'[σ_w] 짝 단위 샘플 간 표준편차 평균 {rep["sigma_w_per_pair"]:.4f}', flush=True)

    (OUT / 'sample_noise.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                           encoding='utf-8')
    (OUT / 'h_by_sample.json').write_text(
        json.dumps({t: [[i, k, round(v, 5)] for (i, k), v in h.items()]
                    for t, h in zip(('s0', 's1', 's2', 'k3'), hs)}, ensure_ascii=False),
        encoding='utf-8')
    print('[done]', OUT / 'sample_noise.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
