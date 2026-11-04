# CLAUDE.md

**STiTy** — real-time multilingual speech translation (SKKU SKKAI Audio AI lab). Streaming ASR
(Qwen3-ASR) → translation → mobile client over WebSocket.

```
Audio in → VAD
         → Qwen3-ASR (streaming)
         → commit trigger (SEG token / VAD / dot)
         → translation (a local LLM, Qwen3.5-4B in the shipped pipelines)
         → client via WebSocket
```

The pipeline is one YAML file in `configs/pipelines/`, assembled from `core/components/` by
`core/pipeline/`. The conversation server (`server/`) and the benchmark (`bench/`) read the same file,
so what bench measures is what the server runs.

**One server is one model**, and nothing in it picks a model by language — the English and Korean
finetunes are two pipelines, and a client picks its language by picking the server.

## Commands

```bash
make server PIPELINE=asr.qwen-seg-en+mt.qwen3.5-4b HOST=0.0.0.0   # conversation server
make bench-manager     # bench experiments: queue runs on GPU machines, configs, compare, replay (:29140)
cd mobile && pnpm install && pnpm start                             # mobile app
```

## Where things are

This file is the map; every measurement and every step-by-step procedure lives next to the code it
describes.

| Path | What, and what to read |
|---|---|
| `server/` | The conversation server — one pipeline for many clients, shared conversations. [server/README.md](server/README.md) |
| `mobile/` | The mobile app. [mobile/README.md](mobile/README.md) |
| `bench/` | Config-driven benchmark of the same pipeline, no WebSocket. Run experiments from the manager website (`bench/manager/`, `make bench-manager`), which queues them on a GPU machine's worker; `make bench`·`make bench-batch` run on this machine and are allowed but not preferred. [bench/README.md](bench/README.md), naming rules in [configs/README.md](configs/README.md) |
| `core/` | Pipeline components, the local translator and segmentation research — [core/CLAUDE.md](core/CLAUDE.md). Which local model to load and how much context it can use: [core/translator/LOCAL_TRANSLATION.md](core/translator/LOCAL_TRANSLATION.md) |
| `Qwen3-ASR/` | Vendored upstream ASR (QwenLM/Qwen3-ASR), tracked directly — **not** a submodule. Finetuning lives in `finetuning/`. [Qwen3-ASR/CLAUDE.md](Qwen3-ASR/CLAUDE.md) |
| `models/` | Finetuned weights: `Qwen3-ASR-1.7B-{ko-silence-v4c900,en-silence-c80}-merged/` and `Qwen3-ASR-1.7B-en-dailytalk-seg/` |
| [docs/WEBSOCKET_PROTOCOL.md](docs/WEBSOCKET_PROTOCOL.md) | Every message type, `langMap` semantics, `commitReason` values |

## Key configuration

- **Backend deps** (`Qwen3-ASR/pyproject.toml`): `transformers==4.57.6`, `accelerate`, `librosa`, `gradio`, `flask`; optional `vllm==0.14.0`. `server/` and `bench/` each pin their own environment (`uv run --project server|bench`).
- **Protocol:** connect → `hello` → `start` → `ready`, then raw PCM up (s16le, 16 kHz, mono) and JSON `partial`/`final` messages back.
