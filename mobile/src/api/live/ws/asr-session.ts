import type {
  CapacityMessage,
  FinalMessage,
  FoldMessage,
  PartialMessage,
  RoomMessage,
  StartMessage,
} from './protocol';
import { WebSocketConnection } from './websocket-connection';

const OPEN_TIMEOUT_MS = 15_000;
/** How long close() waits for the server's closing `final` after `stop`. */
const CLOSE_GRACE_MS = 2_000;
/** The server closes with 4000 + an HTTP status and puts its error code in the reason. */
const SERVER_REFUSAL_CLOSE_CODE = 4_000;

export type AsrSessionHandlers = {
  onPartial: (message: PartialMessage) => void;
  onFinal: (message: FinalMessage) => void;
  /** A brig server retracts a committed final: shrink it to `replacement`, or drop it when empty. */
  onFold?: (message: FoldMessage) => void;
  onCapacity: (message: CapacityMessage) => void;
  onRoom: (message: RoomMessage) => void;
  /** The server or network closed the session, with the server's reason if it gave one. */
  onLost: (reason?: string) => void;
  /** The stream reached its end with nothing lost. Only scripted sessions end on their own. */
  onComplete?: () => void;
};

export type SessionLanguages = { lang: string; targetLang: string };

export type StartOptions = Omit<StartMessage, 'type' | 'translate'>;

type SessionReady = { conversationId: string | null; participantId: string | null };

export class SessionOpenError extends Error {}

/** The server answered the handshake with an `error` message, or closed with its reason. */
export class SessionRefusedError extends SessionOpenError {
  constructor(
    message: string,
    readonly code?: string,
  ) {
    super(message || 'The server refused the session');
  }
}

/** One streaming conversation: hello → start → ready, then PCM up and results down. */
export class AsrSession {
  private connection: WebSocketConnection | null = null;

  constructor(
    private readonly url: string,
    private readonly handlers: AsrSessionHandlers,
  ) {}

  get isOpen(): boolean {
    return this.connection?.isOpen ?? false;
  }

  /** Resolves with the conversation's id and who this phone is in it. */
  open(start: StartOptions): Promise<SessionReady> {
    this.abandon('Reopened');

    return new Promise((resolve, reject) => {
      let settled = false;
      const settle = (result: SessionReady | Error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        if (result instanceof Error) reject(result);
        else resolve(result);
      };

      const connection = new WebSocketConnection(this.url, {
        onMessage: (message) => {
          switch (message.type) {
            case 'hello':
              connection.send({ type: 'start', ...start, translate: !!start.targetLang });
              return;
            case 'ready':
              settle({
                conversationId: message.conversationId,
                participantId: message.participantId,
              });
              return;
            case 'partial':
              this.handlers.onPartial(message);
              return;
            case 'final':
              this.handlers.onFinal(message);
              return;
            case 'fold':
              this.handlers.onFold?.(message);
              return;
            case 'capacity':
              this.handlers.onCapacity(message);
              return;
            case 'room':
              this.handlers.onRoom(message);
              return;
            case 'error':
              if (settled) return;
              if (this.connection === connection) this.abandon('Server refused the session');
              settle(new SessionRefusedError(message.message, message.code));
              return;
          }
        },
        onClose: ({ code, reason }) => {
          if (this.connection !== connection) {
            settle(new SessionOpenError('Session closed before it was ready'));
            return;
          }
          this.connection = null;
          if (settled) this.handlers.onLost(reason);
          else if (code >= SERVER_REFUSAL_CLOSE_CODE)
            settle(new SessionRefusedError(reason, reason));
          else settle(new SessionOpenError('Connection closed before the session was ready'));
        },
      });
      this.connection = connection;

      const timeout = setTimeout(() => {
        if (this.connection === connection) this.abandon('Open timed out');
        settle(new SessionOpenError('Timed out waiting for the server'));
      }, OPEN_TIMEOUT_MS);
    });
  }

  /** Changes the translation direction of the running stream without reopening it. */
  configure({ lang, targetLang }: SessionLanguages): void {
    this.connection?.send({ type: 'config', lang, targetLang });
  }

  /** Starts or stops the server's debug recording of the running stream. */
  setRecording(recording: boolean): void {
    this.connection?.send({ type: 'config', recording });
  }

  setMic(participantId: string, canSpeak: boolean): void {
    this.connection?.send({ type: 'set_mic', participantId, canSpeak });
  }

  setAllMics(canSpeak: boolean): void {
    this.connection?.send({ type: 'set_all_mics', canSpeak });
  }

  removeParticipant(participantId: string): void {
    this.connection?.send({ type: 'remove_participant', participantId });
  }

  sendAudio(pcm: ArrayBuffer): void {
    this.connection?.sendBinary(pcm);
  }

  logSpeech(entry: { text: string; lang: string; start: string; end: string }): void {
    this.connection?.send({ type: 'tts_log', ...entry });
  }

  /** Results keep arriving until the server closes after its last `final`; onLost isn't called. */
  close(): Promise<void> {
    const connection = this.connection;
    if (!connection) return Promise.resolve();
    this.connection = null;
    return connection.finish({ type: 'stop' }, 'User ended session', CLOSE_GRACE_MS);
  }

  /** Closes at once, without `stop` and without waiting for results. */
  private abandon(reason: string): void {
    this.connection?.close(reason);
    this.connection = null;
  }
}
