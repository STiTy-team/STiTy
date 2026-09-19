"""배치 크기가 추론 토큰과 벽시계 시간에 무엇을 하는지 잰다.

분절 비용의 99% 가 출력(추론) 토큰이다. 배치를 6 에서 1 로 내리면 요청 하나가 짧아져
같은 동시성에서 벽시계가 줄지만, **문장을 혼자 줄 때 추론 토큰이 늘어나면** 그 이득을
돈으로 물어내게 된다. 그 방향과 크기는 재봐야 안다.

같은 프롬프트(judge v0)·같은 문장으로 배치만 바꿔 한 벌씩 돌린다. 문장은 어느 분할에도
쓰인 적 없는 새 760 에서 앞 120 — 양쪽 다 캐시 미적중이라 공정하다 (캐시 키에
`batch_size` 가 들어 있어 서로 섞이지도 않는다).

usage 는 팔마다 **끝나는 즉시** 파일에 붙인다. 중간에 죽어도 그때까지의 지출이 남는다.

    tmux new-session -d -s probebatch -c <저장소> \
      "bash core/meaning_segmentator/tools/autoseg_en2x/probe_batch/run.sh"
"""
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.infra.gateway import Gateway
from core.meaning_segmentator.autoseg.runtime import agents_distill as ad, labels as L
from core.meaning_segmentator.autoseg.runtime.pipeline import JsonCache, segment_batch

A = Path('core/meaning_segmentator/experiment/artifacts/en2x/en-multi')
OUT = A / 'probe_batch'
MANIFEST = Path('evaluation/ast/manifests/fleurs_nway_en-de_multi_new760.jsonl')
N, SPACED, MIN_GAP = 120, True, 1
ARMS = [(6, 128), (1, 600)]          # (batch_size, workers)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prompt = (A / 'judge19/prompt_v0.txt').read_text(encoding='utf-8')
    rows = [json.loads(l) for l in MANIFEST.open(encoding='utf-8')][:N]
    texts = [r['src_text'] for r in rows]
    units = [L.units_of(t, SPACED) for t in texts]
    cand = [ad.candidate_positions(len(u), MIN_GAP) for u in units]
    idx = [i for i, c in enumerate(cand) if c]
    marked = [ad.mark_candidates(units[i], cand[i]) for i in idx]
    print(f'[data] {len(marked)}문장 / 어절 평균 '
          f'{st.mean(len(u) for u in units):.1f}', flush=True)

    rec_path = OUT / 'usage.jsonl'
    report = {}
    for batch_size, workers in ARMS:
        gw = Gateway(provider='openai', model='gpt-5-mini', budget=5.0, reasoning_effort='medium',
                     max_connections=max(16, workers),
                     api_keys=[k for k in (os.environ.get('OPENAI_API_KEY'),
                                           os.environ.get('OPENAI_API_KEY_2')) if k])
        cache = JsonCache(OUT / f'cache_b{batch_size}.json')
        # 캐시를 데운다 — 같은 접두를 동시에 던지면 첫 파도가 통째로 캐시를 못 맞는다.
        gw.chat(system=prompt, user=marked[0], max_tokens=64,
                reasoning_effort='medium', purpose='warmup')
        t0 = time.perf_counter()
        outs, ok1 = segment_batch(
            gw, prompt, marked, cache=cache, workers=workers,
            validate_fn=lambda t, o: ad.validate_scored('', t, o, SPACED),
            normalize_fn=ad.normalize_scored, reasoning_effort='medium',
            batch_size=batch_size, first_pass_sink=[],
            realign_fn=lambda t, o: ad.realign_tags(t, o, SPACED))
        wall = time.perf_counter() - t0
        cache.flush()
        u = gw.usage.snapshot()
        seg = u['by_purpose'].get('segment', {})
        valid = sum(1 for i, o in zip(idx, outs) if not ad.validate_scored('', marked[idx.index(i)], o, SPACED))
        arm = {'batch_size': batch_size, 'workers': workers, 'wall_sec': round(wall, 1),
               'sentences': len(marked), 'calls': seg.get('calls'),
               'prompt_tokens': seg.get('prompt_tokens'),
               'cached_tokens': seg.get('cached_tokens'),
               'completion_tokens': seg.get('completion_tokens'),
               'reasoning_tokens': seg.get('reasoning_tokens'),
               'cost': round(u['cost'], 4),
               'first_pass_rate': round(sum(ok1) / len(ok1), 3),
               'valid_outputs': valid}
        arm['reasoning_per_sentence'] = round((seg.get('reasoning_tokens') or 0) / len(marked), 1)
        arm['cost_per_sentence'] = round(u['cost'] / len(marked), 5)
        with rec_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(arm, ensure_ascii=False) + '\n')
        report[f'b{batch_size}'] = arm
        print(f"[batch {batch_size}] 벽시계 {arm['wall_sec']}초 / 호출 {arm['calls']} / "
              f"추론 {arm['reasoning_tokens']} (문장당 {arm['reasoning_per_sentence']}) / "
              f"비용 ${arm['cost']} / 1차통과 {arm['first_pass_rate']}", flush=True)

    b6, b1 = report['b6'], report['b1']
    report['delta'] = {
        'reasoning_ratio': round(b1['reasoning_per_sentence'] / b6['reasoning_per_sentence'], 3),
        'cost_ratio': round(b1['cost'] / b6['cost'], 3),
        'wall_ratio': round(b1['wall_sec'] / b6['wall_sec'], 3)}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                     encoding='utf-8')
    d = report['delta']
    print(f"[결론] batch1/batch6 — 추론 토큰 {d['reasoning_ratio']}배 / "
          f"비용 {d['cost_ratio']}배 / 벽시계 {d['wall_ratio']}배", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
