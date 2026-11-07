# STiTy Mobile

Real-time speech translation app for Android and iOS. Expo SDK 58 · React Native 0.88 (New
Architecture) · TypeScript strict · Expo Router.

The app streams microphone audio to the ASR server over a WebSocket, shows partial and final
transcripts as they arrive, and speaks the translations the server sends. Signed-in users get each
conversation saved by the server and can read it again in Records. The wire format is in
[docs/WEBSOCKET_PROTOCOL.md](../docs/WEBSOCKET_PROTOCOL.md).

## Getting started

```bash
cp .env.example .env          # point EXPO_PUBLIC_SERVER_URL at your server
pnpm install
pnpm android                   # or: pnpm ios — builds a development client
pnpm start                    # later runs: start Metro and open the installed dev client
```

Expo Go cannot run this app: microphone capture and audio routing live in a local native module
(`modules/stity-audio`), so the app always runs as a development build. Any change under
`modules/` or `assets/fonts/` needs a rebuild; JavaScript changes reload instantly.

`pnpm web` runs the same app in a browser, with microphone and speech from the browser (see
[modules/stity-audio](modules/stity-audio/README.md)). It needs no build or install, so it works on
an iPhone, but it can't choose earphone or speaker. The microphone only works on `https` or
`localhost`, so a phone has to open it through a tunnel such as `pnpm start --tunnel`.

| Command                                           | What it does                                                      |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| `pnpm check`                                      | lint + typecheck + format check + tests — run before every commit |
| `pnpm test`                                       | Jest unit tests (`*.test.ts` next to the code)                    |
| `pnpm web`                                        | The app in a browser, served by Metro                             |
| `pnpm run doctor`                                 | `expo-doctor`: dependency and config health                       |
| `pnpm build:dev` / `build:preview` / `build:prod` | EAS cloud builds                                                  |

## Configuration

| Variable                 | Default                                   | Meaning                                                                                                                                                                                       |
| ------------------------ | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EXPO_PUBLIC_SERVER_URL` | `ws://localhost:8765` in development only | WebSocket origin (`wss://`; `ws://` in development only), no path — the app appends `/api/v1/ws`. Required for preview and production                                                         |
| `EXPO_PUBLIC_API_URL`    | the WebSocket host over `http(s)`         | REST origin for accounts and records — the app appends `/api/v1`                                                                                                                              |
| `EXPO_PUBLIC_API_MOCKS`  | everything but `live` (none in prod)      | API areas answered by mock data: a comma list, `all` or `none` — see [API layer](#api-layer-mock-or-live)                                                                                     |
| `STITY_WEB_URL`          | the hosted preview web app                | Where a phone's invite links open. Read at build time from the environment, then the repository's `.env`; EAS cloud builds don't upload that file, so set it in `eas.json` to change it there |
| `APP_VARIANT`            | `development`                             | `development` · `preview` · `production` — set by the EAS profile                                                                                                                             |
| `APP_ID`                 | `com.anonymous.stitymobile`               | Bundle id and Android package. Production builds refuse the placeholder                                                                                                                       |

Values are validated at startup in `src/config/env.ts`; a malformed URL, a URL with a path,
cleartext in a release build, an unknown API area or a mock in production fails fast. For EAS builds, set them per environment with `eas env:create`
instead of committing them.

Each variant gets its own app id (`.dev`, `.preview` suffixes), name and URL scheme (`stity-dev`,
`stity-preview`, `stity`), so all three can be installed side by side. Cleartext `ws://` and `http://` are allowed only in the development
variant.

### Feature switches (`configs/application.yml`)

Features that a platform can't run well are switched per platform in `configs/application.yml`,
the way the server uses `server/configs/application.yml`:

```yaml
features:
  speech:
    ios: true
    android: true
    web: false
```

| Feature  | Off means                                                                                                                                     |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `speech` | Translations aren't read aloud; the sound and output-mode controls are hidden. Off on web: Safari's `speechSynthesis` silences the microphone |

`app.config.ts` reads the file at build time and puts it in `extra.features`. The schema and the
per-platform choice live in `src/config/feature-switches.ts`; the app reads the result as
`features.speech` from `src/config/features.ts`. A test loads the real app config and checks the file
against the schema, so a typo fails `pnpm check` instead of the app. `app.config.ts` runs in Node and
is typechecked on its own with `tsconfig.node.json`. Metro caches the config, so restart with
`npx expo start -c` after editing it.

### Changing the server in a development build

A development build can switch servers without a rebuild. Either way the app signs out (a session
belongs to the server that issued it), saves the address in secure storage, ignores
`EXPO_PUBLIC_API_URL`, and restarts on the new server:

- **Scan** the QR code `make server` prints when it opens a Cloudflare tunnel
  ([server/README.md](../server/README.md#connect-a-client)). It opens the hosted web app at
  `/dev-server?url=wss://…`, which `src/app/dev-server.tsx` handles (a development build answers
  the same path as `stity-dev://dev-server?url=…`). Anyone can send such a link, so the app shows
  the address and asks before it connects.
- **Type** it in Profile → Developer → Server. `https://host` and a bare host are taken as
  `wss://host`. _Use the build's server_ forgets it.

`src/config/server-override.ts` holds the saved address and `changeServer` in
`features/account/account-runtime.ts` switches; `resolveEnv` uses the address only in the
development variant, so preview and production builds always talk to the server they were built
with and show no Developer section.

## Project structure

```
app.config.ts            Expo config: variants, permissions, plugins
configs/application.yml  Feature switches per platform
eas.json                 EAS build profiles (development / preview / production)
modules/stity-audio/     Local native module — mic capture, audio routing, speech (Kotlin + Swift)
src/
├── app/                 Routes only (Expo Router). (tabs)/ holds Translate, Records and Profile
├── features/
│   ├── conversation/    Home, the live conversation, its settings and add-ons, Present mode
│   │   ├── screens/       Screen composition
│   │   ├── components/    UI pieces used only by this feature
│   │   ├── hooks/         React glue (lifecycle, typewriter, capacity notice)
│   │   └── model/         State machine, controller, ports, stores, playback policy — no React
│   ├── records/         Records list, the transcript viewer and its reading settings
│   ├── account/         Sign-in, profile, account deletion, session restore
│   ├── settings/        App language and legal documents, opened from Profile
│   ├── startup/         Crash reporting, session restore and capabilities at launch
│   ├── display/         Applies the saved app language and theme
│   └── intro/           Launch animation that takes over from the native splash
├── api/                 The only code that talks to the server — see below
├── services/            Device-only side effects: mic, speech, haptics, secure storage, telemetry
├── stores/              App-wide state (languages and display, session, catalogue, server status) — zustand
├── components/          Shared UI: buttons, list rows, segmented control, header, bottom sheet
├── domain/              Core vocabulary: languages, output modes
├── i18n/                i18next setup and one JSON file per UI language
├── theme/               Light and dark colors, spacing, Pretendard type scale — styles via makeStyles
├── config/              Validated environment, app info and feature switches
└── lib/                 Small pure helpers
```

### Rules that keep it maintainable

- **Dependencies point one way:** `app → features → api / services / stores → domain / lib / theme`.
  A feature never imports another feature's internals except through its runtime entry point
  (`conversation-runtime.ts`, `account-runtime.ts`, `records-runtime.ts`).
- **State lives with its owner.** `stores/preferences-store.ts` holds only app-wide settings
  (conversation languages, app language, theme). Conversation add-ons, output mode and Save live
  in `features/conversation/model/conversation-preferences.ts`; the transcript viewer's settings
  in `features/records/model/reading-preferences.ts`. Each moved store read its first values from
  the old shared `stity.preferences` key.
- **Features import `@/api` only**, never an adapter file (`*.live.ts`, `*.mock.ts`); lint enforces it.
- **Routes are thin.** A file in `src/app/` only picks a screen; logic lives in `features/`.
- **Business logic has no React.** `ConversationController` drives a session — server check →
  live session → microphone → transcript → speech — and is tested with fakes of the interfaces in
  `model/ports.ts`. Components read stores and call controller methods.
- **One native boundary.** Everything platform-specific about audio is in `modules/stity-audio`;
  TypeScript sees a typed API and events.
- **Every user-facing string is in `src/i18n/locales/*.json`.** `en.json` is the source of truth
  for the key types; other locales fall back to English for missing keys.
- Imports across folders use `@/…` (lint enforces it); files are kebab-case.

### How a conversation flows

The conversation has one state, `phase` in the conversation store, changed only by the
controller's `transition()` along the table in `model/conversation-machine.ts`:

```
idle/failed → starting → recording ⇄ paused → resuming → recording
any → ending → idle
```

`failed` is idle with the error of the last start; a failed resume goes back to `paused`. Each
start or resume is one attempt: End and the background cancel it, a lost connection fails it,
and every step checks it after its `await`, so a step that finishes late never leaves the
microphone or a socket open.

1. `ServerMonitor` keeps a standby socket open while idle, so the Start button knows whether the
   server is up and the server does not idle-shutdown.
2. Start (`openConversation()`) starts the session before it opens the screen, so a second tap
   finds it already starting. The monitor hands over and `api.live.open()` performs `hello → start → ready`. A
   signed-in `start` carries the access token and `save`; `ready` answers with the
   `conversationId` of the record the server writes. `Microphone` then streams 100 ms PCM chunks
   (only while held, with the hold-to-talk add-on), and the output route is set from the mode.
3. `partial` messages update a dimmed live line (typed out by `useTypewriter`); a `final` takes
   over that line in place. The server records the same sentence as it sends it; the app never
   uploads a transcript.
4. The translation comes from the server and is spoken by `SpeechPlayer` on the route chosen by
   `planPlayback`. A `final` without a translation is shown as said and isn't spoken.
5. Going to the background pauses and releases the socket and microphone, and cancels a start
   or resume in flight; returning resumes and resends the same `conversationId`, so the
   conversation stays one record. A conversation already paused stays paused. Only `background`
   counts: iOS `inactive` (Control Center, a call banner) leaves the session running.
6. Closing a session sends `stop` and keeps the socket for up to 2 s, until the server closes it,
   so the sentence the server flushes last still reaches the record (and, after a background
   pause, the transcript).
7. On End, home shows "Saved · Open" linking to the record, or a sign-in prompt for a guest. The
   Save switch is a saved preference, so it stays as the user left it.

**Add-ons** (hold to talk, large text, vibrate on translation) are switches in Conversation
settings, kept in `conversation-preferences.ts`; the controller reads them per chunk and per
translation. **Present mode** (`/present`) shows the same conversation in landscape for a
mirrored TV or projector. It runs on top of the live screen, so the session never restarts; while
it shows, hold to talk is ignored. Its controls are the live screen's `SessionControls` on a dark
floating bar, so it connects, retries and resumes the same way. A tap on the screen shows or hides
them; they come up by themselves whenever recording stops. **The intro** (Profile → About) is a scripted session: `getIntroLive()` replays
`INTRO_SCRIPT` through the same `open()` as a real conversation, with no server or microphone, and
pauses at its end; resuming plays it again. The mocked `live` area uses the same script player. Turning sideways needs `expo-screen-orientation`, so the native app
allows rotation and the JavaScript side locks portrait at launch.

## API layer: mock or live

Everything the app asks of the server goes through `src/api/`. Screens and the controller see the
typed `Api` from `getApi()`; `EXPO_PUBLIC_API_MOCKS` decides per area whether the `live` or the
`mock` adapter answers, so an area switches to the real server by removing one word.

| Area            | Live adapter calls                                 | Mock does                                             |
| --------------- | -------------------------------------------------- | ----------------------------------------------------- |
| `capabilities`  | `GET /capabilities` (a 404 means streaming only)   | Reports every feature as available                    |
| `auth`          | `/auth/email/*`, `/auth/refresh`, `/auth/logout`   | Apple and Google sign in at once; email code `000000` |
| `profile`       | `GET · PATCH · DELETE /me`                         | One seeded profile                                    |
| `conversations` | `GET /conversations`, `GET · PATCH · DELETE /{id}` | Eight sample records of different lengths, ko↔en      |
| `live`          | The WebSocket (works today)                        | Replays a scripted conversation                       |

- The REST endpoints are the **proposed** contract from the improvement plan; one zod schema per
  response (`src/api/schemas.ts`) is shared by both adapters, so a mock can't drift from it.
- Mocks wait 150–600 ms per call and keep their data in AsyncStorage under `stity.mock.v1`, so a
  record survives a restart. A mocked live session writes each `final` into the mock records as
  it emits it; with `live` real and `conversations` mocked, the real session's finals are recorded
  the same way.
- Nothing on screen says an area is mocked; the app treats sample data as real. Signing in with
  email on mocked `auth` takes the code `000000`. Tests can force mocked calls to fail with
  `setMockFailure`.
- Production refuses mocks: `config/env.ts` throws at startup, and `api/modes.test.ts` checks it.
- Tokens live in the phone's secure storage (`expo-secure-store`). An expired access token is
  refreshed once, and before the WebSocket `start` when it is about to expire. The app signs out
  only when the server refuses the refresh token (401/403); a network error or 5xx keeps the
  tokens and fails the call. A refresh still in flight at sign-out can't write tokens back.
- The live Apple and Google sign-in need their native SDKs, so the live adapter offers email only
  until they are added.

## Telemetry

`services/telemetry.ts` reports uncaught errors and five events with no transcript text in them:
`session_started`, `first_translation_shown` (ms), `session_ended` (duration),
`session_error` (code) and `record_opened`. They go to the device log until a crash and analytics
vendor is plugged in with `setTelemetrySink`.

## Rooms

The server gives every unsaved conversation a short room code in `ready`. Invite in the conversation
header shows it with a link to share, and Join room on the home screen enters it on other phones.
The sheet asks `GET /api/v1/rooms/{code}`
on the conversation server first, and stays open with an error unless the room is open, so a bad code
never reaches the conversation screen. That request goes to the WebSocket's host over `http(s)`, not
`EXPO_PUBLIC_API_URL`. The code is then sent as `conversationId` in `start` with `join: true`, so a
room that closed in between is still refused (close code 4404, shown as `errors.roomNotFound`). The
code lives in `room` in the conversation store and is used again on resume.

Inviting or joining makes the conversation shared until it ends. The server mixes every phone's
microphone into one stream, so a shared conversation has no speaker labels: lines show the language
rather than "Me". Translations aren't spoken, because every phone is listening. A room closes when
the last phone leaves.

A saved conversation has no code to share, because its `conversationId` is the id of its record, so
Invite is hidden while saving.

The link is `/join?room=<code>` on the web app, or on `STITY_WEB_URL` when shared from a phone. `src/app/join.tsx` shows the room and the languages and joins on a tap, so a link alone never
starts the microphone. A phone with a server set in the app adds `&server=<url>`, and a development
build asks before it switches to that server. The installed app opens `stity://join?room=<code>`;
opening the `https` link in the app would need universal links.

## Output modes

| Mode     | Who hears what                                                      |
| -------- | ------------------------------------------------------------------- |
| Speaker  | Every translation through the loudspeaker; one shared phone         |
| Earphone | Only the peer's speech, translated into my language, in my earphone |
| Both     | Peer → my earphone; me → the loudspeaker for the peer               |

The mode defaults to Earphone when a headset is connected at launch.

## Web hosting

Every push to `main` that touches `mobile/` exports the web build and deploys it to EAS Hosting as
the production alias (`.github/workflows/deploy-web.yml`, which needs the `EXPO_TOKEN` secret). It
is a development build: it starts on `ws://localhost:8765`, takes its server from Profile →
Developer or a `/dev-server?url=…` link, and answers everything but `live` from mocks. By hand:

```bash
npx expo export --platform web
npx eas-cli deploy          # a preview URL; add --prod to move the production alias
```

## Releasing

1. Bump `version` in `app.config.ts` (build numbers are managed remotely by EAS).
2. `pnpm build:prod`, then `eas submit --profile production`.
3. Before the first store release, set `APP_ID` to the real bundle id — it cannot be changed
   once published, and production builds refuse the placeholder `com.anonymous.stitymobile`.
