"""End-to-end probe of the brig pipeline through the real conversation server.

Streams one northstar scene over the WebSocket at wall-clock speed, as a viewer whose
langMap asks for both directions, and reports what reached the client: finals per
target, folds and whether each fold's segmentId matches a final it can edit — the
exact contract the mobile app relies on.

usage: (server running on :8765) uv run --project server python scripts/brig_ws_probe.py \
         [--wav .../scene1.wav] [--url ws://127.0.0.1:8765/api/v1/ws]"""
import argparse
import asyncio
import json
import time
import wave

import websockets

ap = argparse.ArgumentParser()
ap.add_argument("--wav", default="/home/mobility/stity-datasets/northstar-260928/audio/scene1.wav")
ap.add_argument("--url", default="ws://127.0.0.1:8765/api/v1/ws")
ap.add_argument("--chunk-ms", type=int, default=100)
a = ap.parse_args()


async def main():
    with wave.open(a.wav, "rb") as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1 and f.getsampwidth() == 2
        pcm = f.readframes(f.getnframes())
    duration = len(pcm) / 2 / 16000
    chunk = 2 * 16 * a.chunk_ms

    finals, folds, others = [], [], []
    async with websockets.connect(a.url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello
        await ws.send(json.dumps({
            "type": "start", "lang": "auto",
            "langMap": {"ko": ["en"], "en": ["ko"]},
        }))

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                t = round(time.perf_counter() - t0, 2)
                if m["type"] == "final":
                    finals.append((t, m))
                elif m["type"] == "fold":
                    folds.append((t, m))
                elif m["type"] not in ("partial",):
                    others.append((t, m))

        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "ready":
                break
        t0 = time.perf_counter()
        reading = asyncio.create_task(reader())
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i + chunk])
            target = t0 + (i + chunk) / 2 / 16000
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        await ws.send(json.dumps({"type": "finish"}))
        await asyncio.sleep(5.0)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(reading, timeout=5.0)
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            reading.cancel()

    by_target = {}
    for _, m in finals:
        by_target[m.get("targetLang") or "(전사)"] = by_target.get(m.get("targetLang") or "(전사)", 0) + 1
    final_ids = {m.get("segmentId") for _, m in finals}
    matched = sum(1 for _, m in folds if m.get("segmentId") in final_ids)
    print(f"\n오디오 {duration:.0f}s, final {len(finals)}개 {by_target}, fold {len(folds)}개 "
          f"(segmentId 가 final 과 맞는 것 {matched}/{len(folds)})")
    print("\n처음 10줄:")
    for t, m in finals[:10]:
        tag = m.get("targetLang") or m.get("language") or "?"
        print(f"  {t:6.2f}s [{tag:>4}] #{m.get('segmentId')} {m['translation'] or m['original']}")
    print("\nfold 표본:")
    for t, m in folds[:6]:
        print(f"  {t:6.2f}s [{m['targetLang']}] #{m['segmentId']} '{m['original'][:30]}' -> '{m['replacement'][:30]}'")
    if others:
        print("\n기타:", [(t, m["type"]) for t, m in others[:5]])
    print("PROBE_DONE")


asyncio.run(main())
