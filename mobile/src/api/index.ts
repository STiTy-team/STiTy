import { createMockAuth } from '@/api/auth/auth.mock';
import { createLiveAuth } from '@/api/auth/auth.live';
import { tokenStore } from '@/api/auth/tokens';
import { createLiveCapabilities } from '@/api/capabilities/capabilities.live';
import { createMockCapabilities } from '@/api/capabilities/capabilities.mock';
import { createLiveConversations } from '@/api/conversations/conversations.live';
import { createMockConversations } from '@/api/conversations/conversations.mock';
import { createHttpClient } from '@/api/http';
import { createLiveSession } from '@/api/live/live.live';
import { createIntroLive, createMockLive } from '@/api/live/live.mock';
import { withMockRecording } from '@/api/mock/recorder';
import { mockStore } from '@/api/mock/store';
import { mockedAreas, type ApiModes } from '@/api/modes';
import { createLiveProfile } from '@/api/profile/profile.live';
import { createMockProfile } from '@/api/profile/profile.mock';
import type { Api, LiveApi } from '@/api/types';
import { env } from '@/config/env';

/** Builds the API from the mode map: each area answers from its live or its mock adapter. */
const createApi = (modes: ApiModes): Api => {
  const http = createHttpClient(env.apiUrl, tokenStore);
  const pick = <T>(area: keyof ApiModes, live: () => T, mock: () => T): T =>
    modes[area] === 'mock' ? mock() : live();

  const auth = pick('auth', () => createLiveAuth(http, tokenStore), createMockAuth);

  // Records are written while the session streams, so a mocked records store must hear every
  // final; with `live` real and `conversations` mocked, it listens to the real session.
  const live =
    modes.live === 'mock'
      ? withMockRecording(createMockLive())
      : modes.conversations === 'mock'
        ? withMockRecording(createLiveSession(env.serverUrl, auth))
        : createLiveSession(env.serverUrl, auth);

  return {
    capabilities: pick('capabilities', () => createLiveCapabilities(http), createMockCapabilities),
    auth,
    profile: pick('profile', () => createLiveProfile(http), createMockProfile),
    conversations: pick(
      'conversations',
      () => createLiveConversations(http),
      createMockConversations,
    ),
    live,
  };
};

let instance: Api | null = null;
let intro: LiveApi | null = null;

/** Created on first use, so tests can import modules that reference it without side effects. */
export const getApi = (): Api => (instance ??= createApi(env.apiModes));

/** The scripted introduction, played through the same session interface as a real conversation. */
export const getIntroLive = (): LiveApi => (intro ??= createIntroLive());

export const apiMocks = {
  areas: mockedAreas(env.apiModes),
  /** Erases mock records and the mock session; real data is never touched. */
  reset: () => mockStore.reset(),
};

export { ApiError, apiErrorKind, type ApiErrorKind } from '@/api/errors';
export { onUnauthorized } from '@/api/session-events';
export { setMockFailure } from '@/api/mock/runtime';
export { SessionOpenError } from '@/api/live/ws/asr-session';
export { AUDIO_FORMAT } from '@/api/live/ws/protocol';
export type { CapacityMessage, FinalMessage, PartialMessage } from '@/api/live/ws/protocol';
export type { AsrSessionHandlers, SessionLanguages } from '@/api/live/ws/asr-session';
export type { ServerSnapshot, ServerStatus } from '@/api/live/ws/server-monitor';
export type * from '@/api/types';
