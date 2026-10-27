import { ApiError } from '@/api/errors';
import { mockLatency, useMockControls } from '@/api/mock/runtime';
import { playScript, type ScriptTiming } from '@/api/mock/script-player';
import { INTRO_SCRIPT, SCRIPTS, type ScriptLine } from '@/api/mock/scripts';
import type { LiveApi, LiveMonitor, LiveSession } from '@/api/types';

import type { ServerSnapshot } from './ws/server-monitor';

const MOCK_TIMING: ScriptTiming = {
  firstLineMs: 1_200,
  typing: 'words',
  typingMs: 380,
  lineGapMs: 1_600,
};

/** Each intro line stays a live, still-transcribing sentence for a moment before it commits. */
const INTRO_TIMING: ScriptTiming = {
  firstLineMs: 800,
  typing: 'line',
  typingMs: 1_600,
  lineGapMs: 1_100,
};

/** Always ready: there is no server to watch. */
const createMockMonitor = (): LiveMonitor => {
  const snapshot: ServerSnapshot = { status: 'ready', capacity: null };
  return {
    get current() {
      return snapshot;
    },
    subscribe: (listener) => {
      listener(snapshot);
      return () => true;
    },
    start: () => {},
    stop: () => {},
    retryNow: () => {},
    waitUntilReady: async () => true,
  };
};

type ScriptedOptions = { script: readonly ScriptLine[]; timing: ScriptTiming; mocked: boolean };

/** A session that replays a script instead of listening. Audio sent to it is dropped. */
const createScriptedLive = ({ script, timing, mocked }: ScriptedOptions): LiveApi => {
  let monitor: LiveMonitor | null = null;

  return {
    monitor: () => (monitor ??= createMockMonitor()),
    async open({ handlers }) {
      if (mocked) {
        await mockLatency();
        const { failure } = useMockControls.getState();
        if (failure) throw new ApiError(failure === 'unauthorized' ? 'server' : failure);
      }

      let open = true;
      const stop = playScript(script, handlers, timing);
      const session: LiveSession = {
        conversationId: null,
        get isOpen() {
          return open;
        },
        sendAudio: () => {},
        configure: () => {},
        logSpeech: () => {},
        close: async () => {
          open = false;
          stop();
        },
      };
      return session;
    },
  };
};

/** The mocked `live` area: a café conversation typed out word by word, then committed. */
export const createMockLive = (): LiveApi =>
  createScriptedLive({ script: SCRIPTS.cafe, timing: MOCK_TIMING, mocked: true });

/** The app's introduction, in every build: no server, no latency, no failure switch. */
export const createIntroLive = (): LiveApi =>
  createScriptedLive({ script: INTRO_SCRIPT, timing: INTRO_TIMING, mocked: false });
