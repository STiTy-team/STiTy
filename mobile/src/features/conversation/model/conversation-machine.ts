/**
 * The conversation's one state. `failed` is idle with an error from the last start; `resuming`
 * is a start from `paused`, which falls back to `paused` instead of `failed`.
 *
 *   idle/failed → starting → recording ⇄ paused → resuming → recording
 *   any → ending → idle
 */
export type ConversationPhase =
  'idle' | 'starting' | 'recording' | 'paused' | 'resuming' | 'ending' | 'failed';

export type ConversationEvent =
  | 'start'
  | 'started'
  | 'startFailed'
  | 'pause'
  | 'resume'
  | 'resumeFailed'
  /** The connection or microphone stopped on its own, or a scripted session finished. */
  | 'interrupt'
  /** The app left the screen; a start or resume in flight is cancelled. */
  | 'background'
  | 'end'
  | 'ended';

const TRANSITIONS: Record<
  ConversationEvent,
  Partial<Record<ConversationPhase, ConversationPhase>>
> = {
  start: { idle: 'starting', failed: 'starting' },
  started: { starting: 'recording', resuming: 'recording' },
  startFailed: { starting: 'failed' },
  pause: { recording: 'paused' },
  resume: { paused: 'resuming' },
  resumeFailed: { resuming: 'paused' },
  interrupt: { recording: 'paused' },
  background: { starting: 'paused', resuming: 'paused', recording: 'paused' },
  end: {
    idle: 'ending',
    starting: 'ending',
    recording: 'ending',
    paused: 'ending',
    resuming: 'ending',
    failed: 'ending',
  },
  ended: { ending: 'idle' },
};

/** The phase after `event`, or null when the event does not apply in `phase`. */
export const nextPhase = (
  phase: ConversationPhase,
  event: ConversationEvent,
): ConversationPhase | null => TRANSITIONS[event][phase] ?? null;

/** A start or resume is in flight. */
export const isConnecting = (phase: ConversationPhase): boolean =>
  phase === 'starting' || phase === 'resuming';

/** A conversation is open: connecting, recording or paused. */
export const isInConversation = (phase: ConversationPhase): boolean =>
  isConnecting(phase) || phase === 'recording' || phase === 'paused';

/** Start (or Retry) is allowed. */
export const canStart = (phase: ConversationPhase): boolean =>
  phase === 'idle' || phase === 'failed';
