import { NativeModule, requireNativeModule } from 'expo';

import type { OutputRoute, PermissionResponse, StityAudioEvents } from './StityAudio.types';

declare class StityAudioModule extends NativeModule<StityAudioEvents> {
  getMicrophonePermissionAsync(): Promise<PermissionResponse>;
  requestMicrophonePermissionAsync(): Promise<PermissionResponse>;
  startCaptureAsync(sampleRate: number, chunkSamples: number): Promise<void>;
  stopCapture(): void;
  setOutputRoute(route: OutputRoute): void;
  isHeadsetConnected(): boolean;
  releaseAudioSession(): void;
  speak(utteranceId: string, text: string, languageTag: string, rate: number): void;
  stopSpeaking(): void;
}

export default requireNativeModule<StityAudioModule>('StityAudio');
