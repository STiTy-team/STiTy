import StityAudio, { type OutputRoute } from '@modules/stity-audio';
import type { EventSubscription } from 'expo';

import { toLanguageTag } from '@/domain/languages';

const SPEECH_RATE = 1.3;

export type Utterance = { text: string; language: string; route: OutputRoute };

export type SpokenUtterance = Utterance & { startedAt: string; endedAt: string };

/**
 * Speaks translations one at a time. Each utterance can go to a different output (the "both"
 * mode alternates speaker and earpiece), so the queue lives here rather than in the TTS engine.
 */
export class SpeechPlayer {
  private queue: Utterance[] = [];
  private current: { id: string; utterance: Utterance; startedAt: string } | null = null;
  private nextId = 0;
  private readonly subscriptions: EventSubscription[];

  constructor(private readonly onSpoken: (utterance: SpokenUtterance) => void) {
    this.subscriptions = [
      StityAudio.addListener('onSpeechFinished', ({ utteranceId }) => {
        if (this.current?.id !== utteranceId) return;
        const { utterance, startedAt } = this.current;
        this.onSpoken({ ...utterance, startedAt, endedAt: new Date().toISOString() });
        this.playNext();
      }),
      StityAudio.addListener('onSpeechFailed', ({ utteranceId }) => {
        if (this.current?.id === utteranceId) this.playNext();
      }),
    ];
  }

  enqueue(utterance: Utterance): void {
    if (!utterance.text) return;
    this.queue.push(utterance);
    if (!this.current) this.playNext();
  }

  stop(): void {
    this.queue = [];
    this.current = null;
    StityAudio.stopSpeaking();
  }

  dispose(): void {
    this.stop();
    this.subscriptions.forEach((subscription) => subscription.remove());
  }

  private playNext(): void {
    const next = this.queue.shift();
    if (!next) {
      this.current = null;
      return;
    }
    this.nextId += 1;
    const id = String(this.nextId);
    this.current = { id, utterance: next, startedAt: new Date().toISOString() };
    StityAudio.setOutputRoute(next.route);
    StityAudio.speak(id, next.text, toLanguageTag(next.language), SPEECH_RATE);
  }
}
