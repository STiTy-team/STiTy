import type { LiveMonitor } from '@/api';
import type { LanguageCode } from '@/domain/languages';
import type { OutputMode } from '@/domain/output-modes';
import type { OutputRoute } from '@/services/audio/audio-output';
import type { Utterance } from '@/services/speech/speech-player';

/** What the controller needs from the outside world. Tests pass fakes for each. */

export type ServerMonitorPort = Pick<LiveMonitor, 'start' | 'stop' | 'waitUntilReady'>;

export type AudioOutputPort = {
  route(route: OutputRoute): void;
  /** Returns the phone to its normal audio mode. */
  release(): void;
};

export type MicrophonePort = {
  /** Asks for permission if needed, then streams until stop(). */
  start(): Promise<void>;
  stop(): void;
};

export type SpeechPort = {
  enqueue(utterance: Utterance): void;
  stop(): void;
};

/** What a running session reads, at each start and resume and per chunk and translation. */
export type SessionPreferences = {
  myLanguage: LanguageCode;
  peerLanguage: LanguageCode;
  outputMode: OutputMode;
  muted: boolean;
  holdToTalk: boolean;
  vibrate: boolean;
  saveRequested: boolean;
};
