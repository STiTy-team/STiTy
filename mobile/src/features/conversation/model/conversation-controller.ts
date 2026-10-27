import {
  ApiError,
  AUDIO_FORMAT,
  SessionOpenError,
  type CapacityMessage,
  type FinalMessage,
  type LiveOpenOptions,
  type LiveSession,
  type PartialMessage,
} from '@/api';
import { normalizeLanguageCode } from '@/domain/languages';
import type { OutputMode } from '@/domain/output-modes';
import { createLogger } from '@/lib/logger';
import { MicrophonePermissionError } from '@/services/audio/microphone';
import { hasSpeechEnergy } from '@/services/audio/speech-energy';
import type { SpokenUtterance } from '@/services/speech/speech-player';
import type { TelemetryEvent } from '@/services/telemetry';

import { ConversationError, type ConversationErrorCode } from './conversation-error';
import {
  canStart,
  isConnecting,
  isInConversation,
  nextPhase,
  type ConversationEvent,
  type ConversationPhase,
} from './conversation-machine';
import {
  conversationActions as actions,
  useConversationStore,
  usePresentingStore,
} from './conversation-store';
import { planPlayback, sessionRoute } from './playback-policy';
import type {
  AudioOutputPort,
  MicrophonePort,
  ServerMonitorPort,
  SessionPreferences,
  SpeechPort,
} from './ports';

const SERVER_READY_TIMEOUT_MS = 30_000;

const log = createLogger('conversation');

type OpenSession = (options: LiveOpenOptions) => Promise<LiveSession>;

export type ConversationDependencies = {
  monitor: ServerMonitorPort;
  openSession: OpenSession;
  /** The scripted introduction: no server and no microphone. */
  openIntro: OpenSession;
  output: AudioOutputPort;
  preferences: () => SessionPreferences;
  onCapacity: (capacity: { active: number; max: number }) => void;
  createMicrophone: (options: {
    sampleRate: number;
    chunkSamples: number;
    onChunk: (pcm: ArrayBuffer) => void;
    onFailure: (error: Error) => void;
  }) => MicrophonePort;
  createSpeechPlayer: (onSpoken: (utterance: SpokenUtterance) => void) => SpeechPort;
  track: (event: TelemetryEvent) => void;
  feedback: { translationArrived: () => void };
};

/** What the End screen needs once the transcript is cleared. */
export type EndedConversation = { conversationId: string | null; sentences: number };

/** End and the background cancel an attempt and clean up themselves. */
class AttemptCancelled extends Error {}

/**
 * One start or resume. Each step checks it after its `await`, so a step that finishes after
 * End, the background or a lost connection goes no further.
 */
class Attempt {
  private reason: Error | null = null;

  get cancelled(): boolean {
    return this.reason instanceof AttemptCancelled;
  }

  get stopped(): boolean {
    return this.reason !== null;
  }

  cancel(): void {
    this.reason ??= new AttemptCancelled();
  }

  /** The attempt's own catch handles this one, as a failed start or resume. */
  fail(error: ConversationError): void {
    this.reason ??= error;
  }

  check(): void {
    if (this.reason) throw this.reason;
  }
}

/**
 * Runs a conversation: microphone → live session → transcript store → speech. Translation comes
 * from the server. It lives outside React so the flow can be read top to bottom and tested with
 * fakes. Every phase change goes through `transition()`.
 */
export class ConversationController {
  private session: LiveSession | null = null;
  /** Bumped per opened session, so a late event from a closed one is ignored. */
  private generation = 0;
  /** A session closed by a background pause; its closing finals still belong in the transcript. */
  private draining: number | null = null;
  private attempt: Attempt | null = null;
  private readonly microphone: MicrophonePort;
  private readonly speech: SpeechPort;
  private pausedByBackground = false;
  private inBackground = false;
  private startedAt: number | null = null;
  private firstTranslationTracked = false;

  constructor(private readonly deps: ConversationDependencies) {
    this.microphone = deps.createMicrophone({
      ...AUDIO_FORMAT,
      onChunk: (pcm) => this.handleAudio(pcm),
      onFailure: () => this.interrupt('microphoneUnavailable'),
    });
    this.speech = deps.createSpeechPlayer((spoken) =>
      this.session?.logSpeech({
        text: spoken.text,
        lang: spoken.language,
        start: spoken.startedAt,
        end: spoken.endedAt,
      }),
    );
  }

  /** `demo` plays the scripted introduction through the same session flow. */
  async start({ demo = false }: { demo?: boolean } = {}): Promise<void> {
    if (!canStart(this.phase)) return;
    this.transition('start');
    actions.fail(null);
    actions.setDemo(demo);

    const { myLanguage, peerLanguage } = this.deps.preferences();
    if (!demo && myLanguage === peerLanguage) {
      this.transition('startFailed');
      actions.fail('sameLanguage');
      return;
    }

    const attempt = this.beginAttempt();
    try {
      if (!demo) {
        const ready = await this.deps.monitor.waitUntilReady(SERVER_READY_TIMEOUT_MS);
        attempt.check();
        if (!ready) throw new ConversationError('serverUnreachable');
        this.deps.monitor.stop();
      }
      await this.connectAndCapture(attempt);
      this.finishAttempt(attempt);
      this.transition('started');
      this.markStarted();
    } catch (error) {
      if (attempt.cancelled) return;
      this.finishAttempt(attempt);
      this.releaseTransport();
      this.deps.output.release();
      if (!this.inBackground) this.deps.monitor.start();
      this.transition('startFailed');
      this.fail(errorCode(error));
    }
  }

  pause(): void {
    if (!this.transition('pause')) return;
    this.pausedByBackground = false;
    this.stopCapture();
    // A script keeps playing while its session is open, so a paused demo closes it.
    if (this.isDemo) this.closeSession();
  }

  /** A failed resume stays paused and closes the session it opened. */
  async resume(): Promise<void> {
    if (!this.transition('resume')) return;
    actions.fail(null);
    const attempt = this.beginAttempt();
    try {
      await this.connectAndCapture(attempt);
      this.finishAttempt(attempt);
      this.transition('started');
      this.markStarted();
    } catch (error) {
      if (attempt.cancelled) return;
      this.finishAttempt(attempt);
      this.releaseTransport();
      this.transition('resumeFailed');
      this.fail(errorCode(error));
    }
  }

  end(): EndedConversation {
    const { conversationId, entries, demo, recordedMs, recordingSince } =
      useConversationStore.getState();
    if (this.startedAt !== null && !demo) {
      const running = recordingSince === null ? 0 : Date.now() - recordingSince;
      this.deps.track({ name: 'session_ended', durationMs: recordedMs + running });
    }
    this.transition('end');
    this.cancelAttempt();
    this.startedAt = null;
    this.pausedByBackground = false;
    this.releaseTransport();
    this.deps.output.release();
    actions.reset();
    this.transition('ended');
    if (!this.inBackground) this.deps.monitor.start();
    return { conversationId: demo ? null : conversationId, sentences: entries.length };
  }

  /** Push-to-talk: audio reaches the server only while the button is held. */
  setTalking(talking: boolean): void {
    if (this.phase !== 'recording') return;
    actions.setTalking(talking);
    if (talking) this.speech.stop();
  }

  applyOutputMode(mode: OutputMode): void {
    if (!isInConversation(this.phase)) return;
    this.speech.stop();
    this.deps.output.route(sessionRoute(mode));
  }

  applyLanguages(): void {
    if (!isInConversation(this.phase)) return;
    const { myLanguage, peerLanguage } = this.deps.preferences();
    if (myLanguage === peerLanguage) return;
    this.session?.configure({ lang: myLanguage, targetLang: peerLanguage });
  }

  applyMuted(muted: boolean): void {
    if (muted) this.speech.stop();
  }

  /**
   * The OS may suspend the app at any moment, so the socket and microphone are released. A start
   * or resume in flight is cancelled; returning to the foreground resumes.
   */
  enterBackground(): void {
    this.inBackground = true;
    const phase = this.phase;
    if (isConnecting(phase) || phase === 'recording') {
      this.cancelAttempt();
      this.transition('background');
      this.pausedByBackground = true;
      this.stopCapture();
      this.closeSession({ keepFinals: phase === 'recording' });
    } else if (phase === 'paused') {
      this.closeSession();
    }
    this.deps.monitor.stop();
  }

  enterForeground(): void {
    this.inBackground = false;
    const phase = this.phase;
    if (phase === 'paused' && this.pausedByBackground) {
      this.pausedByBackground = false;
      void this.resume();
    } else if (canStart(phase)) {
      this.deps.monitor.start();
    }
  }

  private get phase(): ConversationPhase {
    return useConversationStore.getState().phase;
  }

  private get isDemo(): boolean {
    return useConversationStore.getState().demo;
  }

  /** The only place the phase changes. False when the event does not apply now. */
  private transition(event: ConversationEvent): boolean {
    const next = nextPhase(this.phase, event);
    if (next === null) return false;
    actions.setPhase(next);
    return true;
  }

  private beginAttempt(): Attempt {
    this.attempt?.cancel();
    this.attempt = new Attempt();
    return this.attempt;
  }

  private finishAttempt(attempt: Attempt): void {
    if (this.attempt === attempt) this.attempt = null;
  }

  private cancelAttempt(): void {
    this.attempt?.cancel();
    this.attempt = null;
  }

  /** A resumed session names the same record, so the server appends instead of starting anew. */
  private async connectAndCapture(attempt: Attempt): Promise<void> {
    const demo = this.isDemo;
    if (!this.session?.isOpen) {
      const { myLanguage, peerLanguage, saveRequested } = this.deps.preferences();
      const { conversationId } = useConversationStore.getState();
      const generation = ++this.generation;
      const current = () => generation === this.generation;
      const open = demo ? this.deps.openIntro : this.deps.openSession;
      const session = await open({
        languages: { lang: myLanguage, targetLang: peerLanguage },
        save: !demo && saveRequested,
        conversationId,
        handlers: {
          onPartial: (message) => current() && this.handlePartial(message),
          onFinal: (message) =>
            (current() || this.draining === generation) && this.handleFinal(message),
          onCapacity: (message) => current() && this.handleCapacity(message),
          onLost: () => current() && this.handleConnectionLost(),
          onComplete: () => current() && this.handleComplete(),
        },
      });
      if (attempt.stopped) void session.close();
      attempt.check();
      this.session = session;
      actions.setConversationId(session.conversationId ?? conversationId);
    }
    if (demo) return;
    await this.microphone.start();
    attempt.check();
    this.deps.output.route(sessionRoute(this.deps.preferences().outputMode));
  }

  private markStarted(): void {
    if (this.startedAt !== null) return;
    this.startedAt = Date.now();
    this.firstTranslationTracked = false;
    if (!this.isDemo) this.deps.track({ name: 'session_started' });
  }

  /** `keepFinals`: the server's last sentence after `stop` still reaches the transcript. */
  private closeSession({ keepFinals = false } = {}): void {
    const closing = this.session;
    const closingGeneration = this.generation;
    this.generation += 1;
    this.session = null;
    this.draining = null;
    if (!closing) return;
    if (keepFinals) this.draining = closingGeneration;
    void closing.close().finally(() => {
      if (this.draining === closingGeneration) this.draining = null;
    });
  }

  private stopCapture(): void {
    this.speech.stop();
    this.microphone.stop();
    actions.clearLive();
    actions.setTalking(false);
  }

  private releaseTransport(): void {
    this.speech.stop();
    this.microphone.stop();
    this.closeSession();
  }

  private fail(code: ConversationErrorCode): void {
    actions.fail(code);
    this.deps.track({ name: 'session_error', code });
  }

  private interrupt(code: ConversationErrorCode): void {
    log.warn('conversation interrupted', code);
    if (this.transition('interrupt')) this.stopCapture();
    this.fail(code);
  }

  private handleAudio(pcm: ArrayBuffer): void {
    const { phase, partial, speechPending, talking } = useConversationStore.getState();
    if (phase !== 'recording') return;
    // A presenter talks freely: nobody can hold a button on a phone mirrored to a big screen.
    const { presenting } = usePresentingStore.getState();
    if (this.deps.preferences().holdToTalk && !presenting && !talking) return;
    if (!partial && !speechPending && hasSpeechEnergy(pcm)) actions.markSpeechPending();
    this.session?.sendAudio(pcm);
  }

  // An empty partial is the server's signal to clear the live line.
  private handlePartial({ text, language }: PartialMessage): void {
    const trimmed = text.trim();
    actions.setPartial(
      trimmed ? { text: trimmed, language: normalizeLanguageCode(language || 'auto') } : null,
    );
  }

  /** A final without a translation is shown as said and isn't spoken. */
  private handleFinal(message: FinalMessage): void {
    const text = message.original.trim();
    if (!text) return;

    const language = normalizeLanguageCode(message.language || 'auto');
    const translation = message.translation.trim();
    actions.commit(language, text, translation);
    if (!translation) return;

    if (!this.firstTranslationTracked && this.startedAt !== null && !this.isDemo) {
      this.firstTranslationTracked = true;
      this.deps.track({ name: 'first_translation_shown', ms: Date.now() - this.startedAt });
    }
    const { vibrate, myLanguage } = this.deps.preferences();
    if (vibrate && language !== myLanguage) {
      this.deps.feedback.translationArrived();
    }
    this.speak(language, translation);
  }

  private handleCapacity({ active, max }: CapacityMessage): void {
    this.deps.onCapacity({ active, max });
  }

  /** During a start or resume the attempt fails; afterwards the conversation pauses with an error. */
  private handleConnectionLost(): void {
    this.closeSession();
    if (isConnecting(this.phase)) {
      this.attempt?.fail(new ConversationError('connectionLost'));
      return;
    }
    this.pausedByBackground = false;
    this.interrupt('connectionLost');
  }

  /** A scripted session reached its end: pause; resuming plays it again. */
  private handleComplete(): void {
    this.closeSession();
    if (this.transition('interrupt')) this.stopCapture();
  }

  private speak(detected: string, translation: string): void {
    const { muted, outputMode, ...pair } = this.deps.preferences();
    if (muted || this.phase !== 'recording') return;
    const plan = planPlayback(detected, pair, outputMode);
    if (plan) this.speech.enqueue({ text: translation, ...plan });
  }
}

const errorCode = (error: unknown): ConversationErrorCode => {
  if (error instanceof ConversationError) return error.code;
  if (error instanceof MicrophonePermissionError) return 'microphoneDenied';
  if (error instanceof SessionOpenError) return 'serverUnreachable';
  if (error instanceof ApiError) return 'serverUnreachable';
  log.error('unexpected conversation error', error);
  return 'microphoneUnavailable';
};
