import { NativeModule, registerWebModule } from 'expo';

import type { OutputRoute, PermissionResponse, StityAudioEvents } from './StityAudio.types';

const UNAVAILABLE: PermissionResponse = { status: 'denied', granted: false, canAskAgain: false };

class StityAudioModule extends NativeModule<StityAudioEvents> {
  async getMicrophonePermissionAsync(): Promise<PermissionResponse> {
    return UNAVAILABLE;
  }

  async requestMicrophonePermissionAsync(): Promise<PermissionResponse> {
    return UNAVAILABLE;
  }

  async startCaptureAsync(_sampleRate: number, _chunkSamples: number): Promise<void> {
    throw new Error('Audio capture is not supported on web');
  }

  stopCapture(): void {}

  setOutputRoute(_route: OutputRoute): void {}

  isHeadsetConnected(): boolean {
    return false;
  }

  releaseAudioSession(): void {}

  speak(utteranceId: string, _text: string, _languageTag: string, _rate: number): void {
    this.emit('onSpeechFailed', { utteranceId, reason: 'error' });
  }

  stopSpeaking(): void {}
}

export default registerWebModule(StityAudioModule, 'StityAudio');
