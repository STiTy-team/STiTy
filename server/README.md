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
- `PIPELINE` is a file name in `configs/pipelines/`. Configs live in S3: when `STITY_S3_BUCKET` is set,
  `make server PIPELINE=<name>` first pulls that one config into `configs/` (it never overwrites a
  local edit), then the server reads the file. Without `PIPELINE` the server runs `stity.pipeline`
  from `server/configs/application.yml`, which is `mock` (get it once with
  `make configs-pull ONLY=pipeline/mock`). `HOST` defaults to 127.0.0.1, and a
  phone needs `0.0.0.0`. `PORT` defaults to 8765. `scripts/server/run.sh` turns these into
  `STITY_STITY__PIPELINE`, `STITY_SERVER__HOST` and `STITY_SERVER__PORT`. Those variables, or a
  gitignored `server/configs/application.local.yml`, work too.
- `STAGE` defaults to `dev`. On `dev`, `make server` also runs `scripts/server/tunnel.sh`, which opens a
  Cloudflare quick tunnel to `PORT` and prints its `https://*.trycloudflare.com` URL in the log. Once the server answers through the tunnel, it also prints a QR code that opens the hosted web app pointed at it (see [Connect a client](#connect-a-client)). The
  URL changes on every start. Needs `cloudflared` on `PATH`. Any other `STAGE` (e.g. `STAGE=prod`,
  on the command line or in `.env`) starts the server without a tunnel.
- `server.cors_origins` and `server.cors_origin_regex` in `application.yml` list the browser
  origins that may call the REST API: Metro's `http://localhost:8081` and the hosted web app
  (`stity-mobile-preview.expo.app` and its per-deploy URLs). The WebSocket doesn't need them.
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

## Recordings

A client asks for recording with `recording: true` in `start` or `config` (the mobile app's
Developer → Record on server switch in conversation settings, development builds only), and can turn
it off again mid-conversation. The server records only when `STITY_RECORDING__ALLOWED=true`;
`scripts/server/run.sh` sets it for `STAGE=dev` unless it is already set, otherwise requests are
refused and `ready`/`config_ok` answer `recording: false`. Each time recording turns on it writes a
new folder, `server/logs/recordings/<date>/<time>-<participant id>/`. `STITY_RECORDING__DIR` moves
the folder. `server/logs/` is gitignored.

| File | What |
|---|---|
| `audio.wav` | Every PCM frame the client sent while recording, 16 kHz mono s16le |
| `events.jsonl` | Every JSON message in both directions except `ping`: `{"t", "audioSeconds", "direction": "in"\|"out", "message"}`. `t` is seconds since the connection opened, `audioSeconds` how much audio had arrived by then |
| `meta.json` | Participant id, pipeline file name, the `serverConfig` the client got in `hello`, start time; end time and audio length are added when recording stops |

The conversation id is in the `ready` message. Replaying `audio.wav` against another pipeline and
comparing its finals with the `out` messages here is the intended test use.

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

**Web app** (`mobile/`, hosted at `STITY_WEB_URL`, default `https://stity-mobile-preview.expo.app`).
With `STAGE=dev`, scan the QR code the server prints with any phone camera. It opens
`<web app>/dev-server?url=wss://….trycloudflare.com`; the app shows the address, asks before it
connects, then signs out and restarts on that server. Or enter the printed `wss://` address in
Profile → Developer → Server. The phone can be on any network. A development build of the mobile
app takes the same address in Profile → Developer → Server. Preview and production builds use the
server they were built with — see [mobile/README.md](../mobile/README.md#configuration).

## Demo script (5 minutes)

1. Start the English server (above) in tmux. Wait for `Application startup complete`, then check
   `curl localhost:8765/api/v1/healthz`.
2. **One speaker.** Open the mobile app pointed at this server, set English → Korean and speak
   English. Partials appear while you talk. Each sentence then commits and turns into Korean about half a
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
  services/recording.py     per-connection audio + message recording (server/logs/recordings/)
  models/                   conversation state, inboxes and feeds, audio channel mixing
scripts/client.py           scripted client that plays the mobile app (the test suite)
```
