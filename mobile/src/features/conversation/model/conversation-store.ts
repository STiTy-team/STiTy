import { create } from 'zustand';

import type { ConversationErrorCode } from './conversation-error';
import type { ConversationPhase } from './conversation-machine';
import type { FeedEvent, LivePartial, SettingChange, TranscriptEntry } from './transcript';

type ConversationState = {
  /** Changed only by the controller's transition(); see conversation-machine.ts. */
  phase: ConversationPhase;
  entries: TranscriptEntry[];
  events: FeedEvent[];
  partial: LivePartial | null;
  /** Speech was heard but the server has not answered yet. */
  speechPending: boolean;
  error: ConversationErrorCode | null;
  /** A scripted demo is playing instead of a real conversation. */
  demo: boolean;
  /** Recording time banked by earlier stretches, and when the current stretch began. */
  recordedMs: number;
  recordingSince: number | null;
  /** The record the server writes for this conversation; kept across pauses and resumes. */
  conversationId: string | null;
  /** Hold-to-talk is held down. */
  talking: boolean;
};

const INITIAL_STATE: ConversationState = {
  phase: 'idle',
  entries: [],
  events: [],
  partial: null,
  speechPending: false,
  error: null,
  demo: false,
  recordedMs: 0,
  recordingSince: null,
  conversationId: null,
  talking: false,
};

export const useConversationStore = create<ConversationState>()(() => INITIAL_STATE);

let nextEntryId = 0;
let nextEventId = 0;

const set = useConversationStore.setState;
const get = useConversationStore.getState;

export const conversationActions = {
  /** Clears the conversation's data; the phase is the controller's to change. */
  reset: () => set(({ phase }) => ({ ...INITIAL_STATE, phase })),
  /** Recording time runs only while recording. */
  setPhase: (phase: ConversationPhase) =>
    set(({ recordedMs, recordingSince }) => {
      const now = Date.now();
      const banked = recordedMs + (recordingSince === null ? 0 : now - recordingSince);
      return {
        phase,
        recordedMs: banked,
        recordingSince: phase === 'recording' ? (recordingSince ?? now) : null,
      };
    }),
  setDemo: (demo: boolean) => set({ demo }),
  fail: (error: ConversationErrorCode | null) => set({ error }),
  setConversationId: (conversationId: string | null) => set({ conversationId }),
  setTalking: (talking: boolean) => set({ talking }),
  markSpeechPending: () => set({ speechPending: true }),
  clearLive: () => set({ partial: null, speechPending: false }),
  setPartial: (partial: LivePartial | null) => set({ partial, speechPending: false }),

  /** Adds a committed sentence, taking over the partial bubble when one was showing. */
  commit: (language: string, text: string, translation: string): string => {
    nextEntryId += 1;
    const id = String(nextEntryId);
    const fromPartial = get().partial !== null;
    set(({ entries }) => ({
      entries: [...entries, { id, language, text, translation, fromPartial, time: Date.now() }],
      partial: null,
      speechPending: false,
    }));
    return id;
  },

  recordChange: (change: SettingChange) => {
    nextEventId += 1;
    const id = `event-${nextEventId}`;
    set(({ entries, events }) => ({
      events: [...events, { id, after: entries.length, time: Date.now(), change }],
    }));
  },
};

/** Present mode is showing. Screen state, not session data, so ending a session keeps it. */
export const usePresentingStore = create<{ presenting: boolean }>()(() => ({ presenting: false }));

export const setPresenting = (presenting: boolean) => usePresentingStore.setState({ presenting });
