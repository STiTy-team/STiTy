# stity-audio

Local Expo module that owns every native audio concern of the app, so no third-party audio
package or `patch-package` patch is needed.

| Area    | Android                                                                                                     | iOS                                                                    |
| ------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Capture | `AudioRecord`, `VOICE_COMMUNICATION` source, platform echo canceller                                        | `AVAudioEngine` with voice processing, resampled by `AVAudioConverter` |
| Routing | `MODE_IN_COMMUNICATION` + communication device (speaker / headset / earpiece)                               | `.playAndRecord` + `.voiceChat`, `overrideOutputAudioPort`             |
| Speech  | `TextToSpeech` (Google engine preferred) with voice-communication audio attributes, so it follows the route | `AVSpeechSynthesizer` on the shared app session                        |

Audio is delivered as `onAudioChunk` events carrying base64 PCM (s16le, mono) at the sample rate
passed to `startCaptureAsync`. Speech is fire-and-forget: `speak` returns at once and
`onSpeechFinished` / `onSpeechFailed` report the outcome per `utteranceId`.

The session stays in communication mode across pause and resume; call `releaseAudioSession` only
when the conversation ends. Native code changes need a rebuild (`npx expo run:android|ios`).
