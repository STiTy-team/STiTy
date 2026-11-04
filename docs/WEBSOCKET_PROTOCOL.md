# WebSocket message protocol

The contract between the conversation server (`server/`, message models in
`server/app/api/v1/schema/conversation.py`) and the mobile app (`mobile/src/api/`).


**Handshake:** connect → server `hello` → client `start` → server `ready` → audio streams.
**Binary frames (client → server):** raw PCM, s16le, 16 kHz, mono, no wrapper.

## Client → Server

| `type` | Key fields | Purpose |
|---|---|---|
| `start` | `lang`, `targetLang`, `displayMode`, `langMap`? | Begin streaming (`lang` = code or `"auto"`) |
| `config` | `langMap`, `lang`?, `targetLang`? | Change translation direction mid-stream (no restart) |
| `stop` / `finish` | — | End and close / flush final segment but stay open |
| `log` / `tts_log` | — | Client-side telemetry appended to server logs |

## Server → Client

| `type` | Key fields | Purpose |
|---|---|---|
| `hello` | `message`, `serverConfig` | Connected |
| `ready` | `message` | Streaming initialized |
| `config_ok` | `lang`, `targetLang`, `langMap` | Echo of the applied `config` |
| `partial` | `text`, `language`, `seq` | Uncommitted text so far; the next `partial` or `final` replaces it |
| `final` | `start`, `end`, `original`, `translation`, `language`, `targetLang`, `commitReason` | Committed segment, one per requested target |

`langMap` maps a **detected** language code to its translation target or target**s**
(`{"ko":"en","ja":"ko"}`, or `{"ko":["en","ja"],"en":["ko"]}` to fan one source out to several).
When present it wins over the `lang` ↔ `targetLang` pair rule, so three or
more languages can each go somewhere different. Unknown codes are dropped; an entry that points at
itself (`{"ko":"ko"}`, or `{"ko":[]}`) means **transcribe only** — the source stays open for ASR and
its segments arrive as `final`s with an empty `translation`. Its **keys** —
the source languages — also replace the pair as the ASR `allowed_languages` set (targets are output only,
so they are not opened). That set is computed per stream slot, so a mid-stream `config` change switches
translation direction at once but reaches ASR only from the next commit on.

A source with several targets produces one `final` per target for the same segment —
same `start`/`end`/`original`/`language`, differing `translation` and `targetLang`. Finals are
ordered **within each target**: a segment's `ja` final never waits for its `en` translation, so
two targets may interleave but each reads in commit order. `config_ok` echoes `langMap` with list
values (`{"ko":["en"]}`) even when the client sent plain strings. A `final` that was never
translated (nowhere to translate to) has an empty `translation` and `targetLang`.

`commitReason`: `seg` (SEG token), `vad` (silence), `dot` (sentence-ending punctuation), `always` (every chunk), `finish` (stream ended) — the pipeline's commit triggers decide which appear.

