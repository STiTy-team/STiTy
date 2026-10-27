import type { AuthApi, LiveApi, LiveSession } from '@/api/types';
import { appInfo } from '@/config/app-info';

import { AsrSession } from './ws/asr-session';
import { CONVERSATION_PATH } from './ws/protocol';
import { ServerMonitor } from './ws/server-monitor';

/** The streaming WebSocket. A signed-in session sends its token so the server records it. */
export const createLiveSession = (serverUrl: string, auth: AuthApi): LiveApi => {
  const url = `${serverUrl}${CONVERSATION_PATH}`;
  let monitor: ServerMonitor | null = null;

  return {
    monitor: () => (monitor ??= new ServerMonitor(url)),
    async open({ languages, save, conversationId, handlers }) {
      const token = save ? await auth.accessToken() : null;
      const session = new AsrSession(url, handlers);
      const ready = await session.open({
        ...languages,
        clientVersion: appInfo.version,
        ...(token
          ? { token, save: true, conversationId: conversationId ?? undefined }
          : { save: false }),
      });
      const live: LiveSession = {
        conversationId: token ? ready.conversationId : null,
        get isOpen() {
          return session.isOpen;
        },
        sendAudio: (pcm) => session.sendAudio(pcm),
        configure: (next) => session.configure(next),
        logSpeech: (entry) => session.logSpeech(entry),
        close: () => session.close(),
      };
      return live;
    },
  };
};
