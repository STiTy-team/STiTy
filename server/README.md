# server/ — conversation server

A FastAPI server that runs one pipeline from `configs/pipelines/` for many clients at once.
Clients speak the mobile app's WebSocket protocol on `/api/v1/ws`
([docs/WEBSOCKET_PROTOCOL.md](../docs/WEBSOCKET_PROTOCOL.md)). Clients that send the same
`conversationId` in `start` share one **conversation**:

- **Transcription runs once per conversation.** Every client that sends audio is a channel.
  The channels are mixed into one stream, and that stream goes through the pipeline's
  VAD + ASR once.
- **Translation runs once per target language.** Each committed segment is translated once
  into each language some client in the conversation reads. Every client that wants that
  language gets the same result.
- A client that never sends audio is a listener. It gets the same partials and finals.

One server holds one model, as in production. The English and Korean finetunes are two pipelines,
and a client picks its language by picking the server.

## Launch

The card needs about 22 GB free (RTX 4090, 24 GB). The ASR engine takes
`gpu_memory_utilization: 0.30` of the card and the Qwen3.5-4B translator takes `0.52`. Both
numbers come from the pipeline file. The server process holds the rest. Nothing else may share
the GPU: stop any bench run first.

```bash
cd <repo root>
make server PIPELINE=asr.qwen-seg-en+mt.qwen3.5-4b HOST=0.0.0.0  # English speakers
# or:  make server PIPELINE=asr.qwen-seg-ko+mt.qwen3.5-4b HOST=0.0.0.0  # Korean speakers
# or:  make server                                                    # mock pipeline, no GPU
```

- The translator's vLLM JIT-compiles kernels with nvcc and finds it through `CUDA_HOME`. Set it
  in `.env` (copy `.env.example`); without one the Makefile falls back to `/usr/local/cuda`. An old
  `/usr/bin/nvcc` fails with `Unknown option '-generate-dependencies-with-compile'`. `make env`
  shows what is in use.
- `PIPELINE` is a file name in `configs/pipelines/`. Without it the server runs `stity.pipeline`
  from `server/configs/application.yml`, which is `mock`. `HOST` defaults to 127.0.0.1, and a
  phone needs `0.0.0.0`. `PORT` defaults to 8765. The Makefile turns these into
  `STITY_STITY__PIPELINE`, `STITY_SERVER__HOST` and `STITY_SERVER__PORT`. Those variables, or a
  gitignored `server/configs/application.local.yml`, work too.
- **Translation server.** You do not start it yourself. At startup the `qwen3.5` translator
  checks `http://127.0.0.1:8100/v1`. If nothing is there, it starts `vllm serve Qwen/Qwen3.5-4B`
  from its own uv project (`core/components/translation/local_qwen_3_5_4b/`) and stops it when
  the server exits. Its log is `/tmp/qwen3.5-vllm.log`. If a vLLM server for that model is
  already on 8100, it is reused.
- **Startup time.** The first start on this machine's HDD took 13 min (780 s). The weights, the
  torch.compile cache and the two uv environments are all read cold. Later starts take a few
  minutes. The server is ready when the log prints `Application startup complete` and
  `GET /api/v1/healthz` answers `{"status": "ok", ...}`.
- **Environment.** `.env` at the repository root is loaded if it exists. These pipelines need no
  API keys. The models come from the HuggingFace cache (`Doo12/...`, `Qwen/Qwen3.5-4B`).
- The server's uv environment has the same ASR stack as `bench/`: `qwen-asr[vllm]` with vLLM
  0.14, silero-vad. The translator's environment is separate (vLLM ≥ 0.17), because the two
  vLLM versions cannot share one environment.

Run it in tmux so it outlives your terminal (`.claude/rules/long-jobs-in-tmux.md`).

## Connect a client

**Scripted client** (the tests). This is the quickest way to see the server work, and it needs
FLEURS in `$STITY_DATA_ROOT` (see `bench/README.md`):

```bash
PYTHONPATH=server uv run --project server python server/scripts/client.py \
  --data-root $STITY_DATA_ROOT --lang en \
  --server-log <server log> --translator-log /tmp/qwen3.5-vllm.log
```

`--lang` is the language of the loaded model (`en` or `ko`). Pass the server's log to check that
each stage ran once. It prints PASS/FAIL per check and exits non-zero on a failure.

**Mobile app.** The server address is hard-coded as `RUNPOD_SERVER_URL` in
`STiTy-Mobile/src/context/WebSocketContext.tsx`. Set it to `ws://<this machine's LAN IP>:8765`.
The app appends `/api/v1/ws` itself. Then run `npm start` in `STiTy-Mobile/`. The phone must
reach that IP (same Wi-Fi, no client isolation). The app sends no `conversationId`, so each phone
gets its own conversation.

**Web demo** (`STiTy-Mobile/demo-web/partial_demo/`). Run its proxy in plain relay mode with
every route pointing at this server:

```bash
S=ws://127.0.0.1:8765/api/v1/ws
uv run --project server python STiTy-Mobile/demo-web/partial_demo/demo_proxy.py 8080 \
  --route en=$S,ko=$S --default $S
```

Open `http://localhost:8080/show.html?src=en&tgt=ko&layout=tag&pick=ko` and press the microphone. To use a
recorded clip instead, put a 16 kHz mono wav at `web/partial_test_en.wav` (gitignored) and open
`?auto=en`. This server sends one `translation` per client, the one for its `targetLang`. The
page's lane layouts (the default) draw one lane per language and expect the old proxy's
multi-target `translations`, so the lanes of languages nobody translated into stay empty. Use
`layout=tag&pick=<target>`: one column with the translation large and the original under it. Leave out the proxy's `--dual`, `--lid*`,
`--translate-url` and `--targets` flags: they drive the old per-language ASR servers.

## Demo script (5 minutes)

1. Start the English server (above) in tmux. Wait for `Application startup complete`, then check
   `curl localhost:8765/api/v1/healthz`.
2. **One speaker.** Open the web demo with `?src=en&tgt=ko&layout=tag&pick=ko` and speak English. Grey
   partials appear while you talk. Each sentence then commits and turns into Korean about half a
   second after you stop. The scripted client measured a median of 0.4 s from the end of the
   audio to the final.
3. **A shared conversation.** Run `server/scripts/client.py --cases join` (with the flags above) in a second terminal.
   A speaker and two listeners join one named conversation, one listener reading Korean and one
   Japanese. The output shows both listeners get the same originals, each in its own language.
   Point at the server log: one `Started transcription worker` for the room, and one
   `Started <lang> translation worker` per language.
4. **Robustness.** Run `--cases lifecycle,bad`. A listener leaves and comes back, and a client
   drops mid-sentence. Bad ids and malformed frames are refused with a close code. The server
   stays healthy the whole time.
5. Stop the server with Ctrl-C. `Closing N open conversation(s)` then `Stopped STiTy server`.

## What is not there yet

- No HTTP API for conversations: a conversation exists while at least one client is in it, and
  the id is whatever the first client sends in `start` (or a new UUID).
- No join tickets. Anyone who knows an id can join it.
- No backlog: a client that joins late gets only the finals committed after it joined.
- One `targetLang` per client.

## Layout

```
app/
  main.py, __main__.py      FastAPI app and uvicorn entry (python -m app)
  config.py                 application.yml (+ application.local.yml) + STITY_* env → ServerConfig
  api/websocket.py          generic receive/send loop, close codes (4000 + HTTP status)
  api/v1/endpoints/         /api/v1/ws (conversation), /api/v1/healthz
  services/conversation/    joining, leaving, the transcription and per-language translation workers
  models/                   conversation state, inboxes and feeds, audio channel mixing
scripts/client.py           scripted client that plays the mobile app (the test suite)
```
