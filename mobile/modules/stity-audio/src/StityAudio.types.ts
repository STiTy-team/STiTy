export type OutputRoute = 'speaker' | 'earpiece';

export type PermissionResponse = {
  status: 'granted' | 'denied' | 'undetermined';
  granted: boolean;
  canAskAgain: boolean;
};

export type AudioChunkEvent = {
  /** Base64-encoded PCM, signed 16-bit little-endian, mono. */
  data: string;
};

export type CaptureErrorEvent = {
  message: string;
};

export type SpeechFinishedEvent = {
  utteranceId: string;
};

export type SpeechFailedEvent = {
  utteranceId: string;
  reason: 'cancelled' | 'error' | 'unsupported-language';
};

export type StityAudioEvents = {
  onAudioChunk(event: AudioChunkEvent): void;
  onCaptureError(event: CaptureErrorEvent): void;
  onSpeechFinished(event: SpeechFinishedEvent): void;
  onSpeechFailed(event: SpeechFailedEvent): void;
};
