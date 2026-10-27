import StityAudio from '@modules/stity-audio';
import type { EventSubscription } from 'expo';

import { base64ToArrayBuffer } from '@/lib/base64';
import { createLogger } from '@/lib/logger';

const WATCHDOG_MS = 4_000;
const RESTART_DELAY_MS = 800;

const log = createLogger('microphone');

export class MicrophonePermissionError extends Error {}
export class MicrophoneUnavailableError extends Error {}

type Options = {
  sampleRate: number;
  chunkSamples: number;
  onChunk: (pcm: ArrayBuffer) => void;
  /** Capture stopped on its own: no data, a native error or an OS interruption. */
  onFailure: (error: MicrophoneUnavailableError) => void;
};

/**
 * Streams PCM from the native module. A watchdog restarts capture once when no audio arrives —
 * call apps such as Zoom can hold the microphone for a moment — and reports failure after that.
 */
export class Microphone {
  private subscriptions: EventSubscription[] = [];
  private watchdog: ReturnType<typeof setTimeout> | null = null;
  private receivedAudio = false;
  private active = false;
  /** Bumped by stop(), so a start or restart still in flight knows it was overtaken. */
  private run = 0;

  constructor(private readonly options: Options) {}

  /** Resolves once capture runs. A stop() while it waits, even for permission, wins. */
  async start(): Promise<void> {
    if (this.active) return;
    this.active = true;
    const run = ++this.run;
    try {
      await this.ensurePermission();
    } catch (error) {
      if (run === this.run) this.stop();
      throw error;
    }
    if (run !== this.run) return;

    this.subscriptions = [
      StityAudio.addListener('onAudioChunk', ({ data }) => {
        this.receivedAudio = true;
        this.options.onChunk(base64ToArrayBuffer(data));
      }),
      StityAudio.addListener('onCaptureError', ({ message }) => {
        log.warn('capture error', message);
        this.fail(message);
      }),
    ];

    try {
      await this.startCapture();
    } catch (error) {
      if (run === this.run) this.stop();
      throw error;
    }
    if (this.stoppedSince(run)) return;
    this.armWatchdog(() => void this.restart());
  }

  stop(): void {
    this.run += 1;
    this.active = false;
    this.clearWatchdog();
    this.subscriptions.forEach((subscription) => subscription.remove());
    this.subscriptions = [];
    StityAudio.stopCapture();
  }

  private async ensurePermission(): Promise<void> {
    const current = await StityAudio.getMicrophonePermissionAsync();
    if (current.granted) return;
    const requested = current.canAskAgain
      ? await StityAudio.requestMicrophonePermissionAsync()
      : current;
    if (!requested.granted) throw new MicrophonePermissionError('Microphone permission denied');
  }

  /** Leaves `active` alone, so a failed restart still reaches `fail()` and `onFailure`. */
  private async startCapture(): Promise<void> {
    this.receivedAudio = false;
    try {
      await StityAudio.startCaptureAsync(this.options.sampleRate, this.options.chunkSamples);
    } catch (error) {
      StityAudio.stopCapture();
      throw new MicrophoneUnavailableError(String(error));
    }
  }

  private async restart(): Promise<void> {
    log.warn('no audio from the microphone, restarting capture');
    const run = this.run;
    StityAudio.stopCapture();
    await new Promise((resolve) => setTimeout(resolve, RESTART_DELAY_MS));
    if (run !== this.run) return;
    try {
      await this.startCapture();
      if (this.stoppedSince(run)) return;
      this.armWatchdog(() => this.fail('no audio after restart'));
    } catch (error) {
      if (run === this.run) this.fail(String(error));
    }
  }

  /** Native capture may have started after a stop(); it is stopped unless a newer run owns it. */
  private stoppedSince(run: number): boolean {
    if (run === this.run) return false;
    if (!this.active) StityAudio.stopCapture();
    return true;
  }

  private armWatchdog(onSilence: () => void): void {
    this.clearWatchdog();
    this.watchdog = setTimeout(() => {
      if (this.active && !this.receivedAudio) onSilence();
    }, WATCHDOG_MS);
  }

  private clearWatchdog(): void {
    if (this.watchdog) clearTimeout(this.watchdog);
    this.watchdog = null;
  }

  private fail(reason: string): void {
    if (!this.active) return;
    this.stop();
    this.options.onFailure(new MicrophoneUnavailableError(reason));
  }
}
