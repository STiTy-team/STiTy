import { NativeModule, registerWebModule } from 'expo';

import type { OutputRoute, PermissionResponse, StityAudioEvents } from './StityAudio.types';

/**
 * Runs on the audio thread: averages the context's samples down to the target rate (a box filter,
 * enough to keep speech from aliasing) and posts s16le chunks of `chunkSamples`.
 */
const CAPTURE_WORKLET = `
class StityCapture extends AudioWorkletProcessor {
  constructor({ processorOptions: { targetRate, chunkSamples } }) {
    super();
    this.step = sampleRate / targetRate;
    this.position = 0;
    this.sum = 0;
    this.count = 0;
    this.chunkSamples = chunkSamples;
    this.chunk = new Int16Array(chunkSamples);
    this.filled = 0;
  }
  process([input]) {
    const samples = input && input[0];
    if (!samples) return true;
    for (let i = 0; i < samples.length; i += 1) {
      this.sum += samples[i];
      this.count += 1;
      this.position += 1;
      if (this.position < this.step) continue;
      this.position -= this.step;
      const value = Math.max(-1, Math.min(1, this.sum / this.count));
      this.chunk[this.filled++] = value < 0 ? value * 0x8000 : value * 0x7fff;
      this.sum = 0;
      this.count = 0;
      if (this.filled === this.chunkSamples) {
        this.port.postMessage(this.chunk.buffer, [this.chunk.buffer]);
        this.chunk = new Int16Array(this.chunkSamples);
        this.filled = 0;
      }
    }
    return true;
  }
}
registerProcessor('stity-capture', StityCapture);
`;

const SPEECH_MS_PER_CHAR = 120;
const SPEECH_GRACE_MS = 5000;

const DENIED: PermissionResponse = { status: 'denied', granted: false, canAskAgain: false };

const toBase64 = (buffer: ArrayBuffer): string => {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
};

const permissionFrom = (state: PermissionState): PermissionResponse => ({
  status: state === 'prompt' ? 'undetermined' : state,
  granted: state === 'granted',
  canAskAgain: state !== 'denied',
});

const voiceFor = (languageTag: string): SpeechSynthesisVoice | undefined => {
  const voices = speechSynthesis.getVoices();
  const tag = languageTag.toLowerCase();
  const language = tag.split('-')[0];
  return (
    voices.find((voice) => voice.lang.toLowerCase().replace('_', '-') === tag) ??
    voices.find((voice) => voice.lang.toLowerCase().split(/[-_]/)[0] === language)
  );
};

type Capture = { stream: MediaStream; context: AudioContext; node: AudioWorkletNode };

type Speech = {
  utterance: SpeechSynthesisUtterance;
  started: boolean;
  timer: ReturnType<typeof setTimeout>;
};

const unlockSpeechOnFirstTap = () => {
  if (typeof document === 'undefined' || typeof speechSynthesis === 'undefined') return;
  const unlock = () => {
    speechSynthesis.speak(new SpeechSynthesisUtterance(''));
    document.removeEventListener('touchend', unlock, true);
    document.removeEventListener('click', unlock, true);
  };
  document.addEventListener('touchend', unlock, true);
  document.addEventListener('click', unlock, true);
};

/**
 * The browser version: microphone through getUserMedia and an AudioWorklet, speech through
 * speechSynthesis. Browsers choose the output device themselves, so routing is a no-op, and the
 * microphone needs a secure origin (https or localhost). Browser echo cancellation doesn't cover
 * speechSynthesis, so capture sends silence while a translation is audibly playing (from `onstart`
 * on) instead of hearing it. An utterance that never settles is failed after a length-based
 * deadline, so a browser that loses speech events can't silence the microphone or stall the queue.
 */
class StityAudioModule extends NativeModule<StityAudioEvents> {
  private capture: Capture | null = null;
  private utterances = new Map<string, Speech>();

  constructor() {
    super();
    unlockSpeechOnFirstTap();
  }

  private get speaking(): boolean {
    for (const speech of this.utterances.values()) if (speech.started) return true;
    return false;
  }

  async getMicrophonePermissionAsync(): Promise<PermissionResponse> {
    if (!navigator.mediaDevices?.getUserMedia) return DENIED;
    try {
      const { state } = await navigator.permissions.query({
        name: 'microphone' as PermissionName,
      });
      return permissionFrom(state);
    } catch {
      return { status: 'undetermined', granted: false, canAskAgain: true };
    }
  }

  async requestMicrophonePermissionAsync(): Promise<PermissionResponse> {
    if (!navigator.mediaDevices?.getUserMedia) return DENIED;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
      return permissionFrom('granted');
    } catch {
      return DENIED;
    }
  }

  async startCaptureAsync(sampleRate: number, chunkSamples: number): Promise<void> {
    this.stopCapture();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    const context = new AudioContext();
    try {
      const url = URL.createObjectURL(
        new Blob([CAPTURE_WORKLET], { type: 'application/javascript' }),
      );
      try {
        await context.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      const node = new AudioWorkletNode(context, 'stity-capture', {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        processorOptions: { targetRate: sampleRate, chunkSamples },
      });
      node.port.onmessage = ({ data }: MessageEvent<ArrayBuffer>) => {
        if (this.speaking) new Int16Array(data).fill(0);
        this.emit('onAudioChunk', { data: toBase64(data) });
      };
      context.createMediaStreamSource(stream).connect(node);
      stream.getAudioTracks()[0]?.addEventListener('ended', () => {
        if (this.capture?.stream !== stream) return;
        this.stopCapture();
        this.emit('onCaptureError', { message: 'The microphone was disconnected' });
      });
      context.onstatechange = () => {
        if (this.capture?.context !== context) return;
        if (context.state !== 'running' && context.state !== 'closed')
          void context.resume().catch(() => {});
      };
      await context.resume();
      this.capture = { stream, context, node };
    } catch (error) {
      stream.getTracks().forEach((track) => track.stop());
      void context.close();
      throw error;
    }
  }

  stopCapture(): void {
    if (!this.capture) return;
    const { stream, context, node } = this.capture;
    this.capture = null;
    node.port.onmessage = null;
    context.onstatechange = null;
    node.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    void context.close();
  }

  setOutputRoute(_route: OutputRoute): void {}

  isHeadsetConnected(): boolean {
    return false;
  }

  releaseAudioSession(): void {}

  speak(utteranceId: string, text: string, languageTag: string, rate: number): void {
    if (typeof speechSynthesis === 'undefined') {
      this.emit('onSpeechFailed', { utteranceId, reason: 'error' });
      return;
    }
    const voice = voiceFor(languageTag);
    if (!voice && speechSynthesis.getVoices().length > 0) {
      this.emit('onSpeechFailed', { utteranceId, reason: 'unsupported-language' });
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = languageTag;
    if (voice) utterance.voice = voice;
    utterance.rate = rate;
    const settle = (event: 'finished' | SpeechSynthesisErrorCode) => {
      const speech = this.utterances.get(utteranceId);
      if (speech?.utterance !== utterance) return;
      clearTimeout(speech.timer);
      this.utterances.delete(utteranceId);
      if (event === 'finished') this.emit('onSpeechFinished', { utteranceId });
      else {
        const reason = event === 'canceled' || event === 'interrupted' ? 'cancelled' : 'error';
        this.emit('onSpeechFailed', { utteranceId, reason });
      }
    };
    utterance.onstart = () => {
      const speech = this.utterances.get(utteranceId);
      if (speech?.utterance === utterance) speech.started = true;
    };
    utterance.onend = () => settle('finished');
    utterance.onerror = ({ error }) => settle(error);
    const deadline = (text.length * SPEECH_MS_PER_CHAR) / rate + SPEECH_GRACE_MS;
    const timer = setTimeout(() => {
      speechSynthesis.cancel();
      settle('synthesis-failed');
    }, deadline);
    // Kept referenced until it settles: some browsers drop events of a collected utterance.
    this.utterances.set(utteranceId, { utterance, started: false, timer });
    speechSynthesis.speak(utterance);
  }

  stopSpeaking(): void {
    const pending = [...this.utterances.keys()];
    this.utterances.forEach(({ timer }) => clearTimeout(timer));
    this.utterances.clear();
    if (typeof speechSynthesis !== 'undefined') speechSynthesis.cancel();
    pending.forEach((utteranceId) =>
      this.emit('onSpeechFailed', { utteranceId, reason: 'cancelled' }),
    );
  }
}

export default registerWebModule(StityAudioModule, 'StityAudio');
