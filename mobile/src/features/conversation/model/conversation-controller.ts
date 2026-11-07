import {
  ApiError,
  AUDIO_FORMAT,
  SessionOpenError,
  SessionRefusedError,
  type CapacityMessage,
  type DeviceInfo,
  type RoomMessage,
  type FinalMessage,
  type FoldMessage,
  type LiveOpenOptions,
  type LiveSession,
  type PartialMessage,
} from '@/api';
import { normalizeLanguageCode } from '@/domain/languages';
import type { OutputMode } from '@/domain/output-modes';
import { newRoom, type Room } from '@/domain/rooms';
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
  selectInRoom,
  selectMicOff,
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
  device: () => DeviceInfo;
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
  /** The newest committed segment, so extra finals for other targets join its bubble. */
  private lastFinal: { segment: string; entryId: string } | null = null;
  /** Server segment id -> transcript entry, so a brig `fold` can edit the line it retracts. */
  private segmentEntries = new Map<string, string>();

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

  /**
   * `demo` plays the scripted introduction through the same session flow. `room` joins someone
   * else's conversation; otherwise a retry keeps its room and a new start gets one to invite to.
   */
  async start({ demo = false, room }: { demo?: boolean; room?: Room } = {}): Promise<void> {
    if (!canStart(this.phase)) return;
    this.transition('start');
    actions.fail(null);
    actions.setDemo(demo);
    actions.setRoom(demo ? null : (room ?? useConversationStore.getState().room ?? newRoom()));

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

  /** Inviting turns this conversation into a room; it stays one until it ends. */
  invite(): void {
    const { room } = useConversationStore.getState();
    if (!room || room.shared) return;
    actions.setRoom({ ...room, shared: true });
    this.speech.stop();
  }

  setMic(participantId: string, canSpeak: boolean): void {
    this.session?.setMic(participantId, canSpeak);
  }

  setAllMics(canSpeak: boolean): void {
    this.session?.setAllMics(canSpeak);
  }

  removeParticipant(participantId: string): void {
    this.session?.removeParticipant(participantId);
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

  applyServerRecording(): void {
    if (!isInConversation(this.phase) || this.isDemo) return;
    this.session?.setRecording(this.deps.preferences().serverRecording);
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
      const { myLanguage, peerLanguage, saveRequested, serverRecording } = this.deps.preferences();
      const { conversationId, room } = useConversationStore.getState();
      const generation = ++this.generation;
      const current = () => generation === this.generation;
      const open = demo ? this.deps.openIntro : this.deps.openSession;
      const session = await open({
        languages: { lang: myLanguage, targetLang: peerLanguage },
        save: !demo && saveRequested && !room?.shared,
        conversationId,
        room,
        recording: !demo && serverRecording,
        device: this.deps.device(),
        handlers: {
          onPartial: (message) => current() && this.handlePartial(message),
          onFinal: (message) =>
            (current() || this.draining === generation) && this.handleFinal(message),
          onFold: (message) =>
            (current() || this.draining === generation) && this.handleFold(message),
          onCapacity: (message) => current() && this.handleCapacity(message),
          onRoom: (message) => current() && this.handleRoom(message),
          onLost: (reason) => current() && this.handleConnectionLost(reason),
          onComplete: () => current() && this.handleComplete(),
        },
      });
      if (attempt.stopped) void session.close();
      attempt.check();
      this.session = session;
      actions.setConversationId(session.conversationId ?? conversationId);
      actions.setParticipantId(session.participantId);
      if (room) {
        actions.setRoom({ ...room, code: session.roomCode ?? room.code, joining: false });
      }
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
    this.lastFinal = null;
    this.segmentEntries.clear();
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
    const store = useConversationStore.getState();
    const { phase, partial, speechPending, talking } = store;
    if (phase !== 'recording' || selectMicOff(store)) return;
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
    const segment = `${message.start ?? ''}|${message.end ?? ''}|${language}|${text}`;
    if (this.lastFinal && this.lastFinal.segment === segment) {
      if (translation) actions.appendTranslation(this.lastFinal.entryId, translation);
      return;
    }
    this.lastFinal = { segment, entryId: actions.commit(language, text, translation) };
    if (message.segmentId) this.segmentEntries.set(message.segmentId, this.lastFinal.entryId);
    if (this.segmentEntries.size > 200) {
      const oldest = this.segmentEntries.keys().next().value;
      if (oldest !== undefined) this.segmentEntries.delete(oldest);
    }
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

  /** A brig fold edits exactly one committed line; empty replacement removes it. */
  private handleFold(message: FoldMessage): void {
    const entryId = this.segmentEntries.get(message.segmentId);
    if (!entryId) return;
    const replacement = message.replacement.trim();
    if (replacement) {
      actions.rewriteEntry(entryId, replacement, replacement);
    } else {
      actions.removeEntry(entryId);
      this.segmentEntries.delete(message.segmentId);
      if (this.lastFinal?.entryId === entryId) this.lastFinal = null;
    }
  }

  private handleCapacity({ active, max }: CapacityMessage): void {
    this.deps.onCapacity({ active, max });
  }

  /** Someone joining means it's shared, even if this phone never opened Invite. */
  private handleRoom({ hostId, participants }: RoomMessage): void {
    actions.setRoster({ hostId, participants });
    if (participants.length > 1) this.invite();
    if (selectMicOff(useConversationStore.getState())) {
      actions.clearLive();
      actions.setTalking(false);
    }
  }

  /** During a start or resume the attempt fails; afterwards the conversation pauses with an error. */
  private handleConnectionLost(reason?: string): void {
    this.closeSession();
    const closedRoom = reason ? CLOSED_ROOM_ERRORS[reason] : undefined;
    if (closedRoom) {
      actions.setRoom(null);
      actions.setRoster(null);
    }
    const code = closedRoom ?? 'connectionLost';
    if (isConnecting(this.phase)) {
      this.attempt?.fail(new ConversationError(code));
      return;
    }
    this.pausedByBackground = false;
    this.interrupt(code);
  }

  /** A scripted session reached its end: pause; resuming plays it again. */
  private handleComplete(): void {
    this.closeSession();
    if (this.transition('interrupt')) this.stopCapture();
  }

  private speak(detected: string, translation: string): void {
    const { muted, outputMode, ...pair } = this.deps.preferences();
    // Every phone in a room is listening, so one speaking aloud would be transcribed again.
    if (muted || this.phase !== 'recording' || selectInRoom(useConversationStore.getState())) {
      return;
    }
    const plan = planPlayback(detected, pair, outputMode);
    if (plan) this.speech.enqueue({ text: translation, ...plan });
  }
}

const ROOM_NOT_FOUND = 'conversation_not_found';

/** The server's reasons for ending a room for this phone. */
const CLOSED_ROOM_ERRORS: Record<string, ConversationErrorCode> = {
  removed_from_room: 'removedFromRoom',
  room_ended: 'roomEnded',
};

const errorCode = (error: unknown): ConversationErrorCode => {
  if (error instanceof ConversationError) return error.code;
  if (error instanceof MicrophonePermissionError) return 'microphoneDenied';
  if (error instanceof SessionRefusedError && error.code === ROOM_NOT_FOUND) return 'roomNotFound';
  if (error instanceof SessionOpenError) return 'serverUnreachable';
  if (error instanceof ApiError) return 'serverUnreachable';
  log.error('unexpected conversation error', error);
  return 'microphoneUnavailable';
};
