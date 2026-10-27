import type { LiveApi, LiveOpenOptions } from '@/api/types';
import { createLogger } from '@/lib/logger';

import { mockStore } from './store';

const TITLE_LENGTH = 48;

const log = createLogger('mock-recorder');

const logFailure = (error: unknown) => log.warn('mock recording failed', error);

const newId = () => `rec-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;

const titleFrom = (text: string) =>
  text.length > TITLE_LENGTH ? `${text.slice(0, TITLE_LENGTH).trimEnd()}…` : text;

/** Creates the record for a session, or reuses the one a resumed session names. Null for guests. */
const startRecording = (options: LiveOpenOptions): Promise<string | null> =>
  mockStore.update((data) => {
    if (!options.save || !data.signedIn) return null;
    const existing = data.records.find((record) => record.id === options.conversationId);
    if (existing) {
      existing.endedAt = null;
      return existing.id;
    }
    const id = newId();
    data.records.unshift({
      id,
      title: '',
      startedAt: new Date().toISOString(),
      endedAt: null,
      languages: [options.languages.lang, options.languages.targetLang],
      myLanguage: options.languages.lang,
      segments: [],
      summary: { status: 'unavailable' },
    });
    return id;
  });

const appendSegment = (
  id: string,
  segment: { language: string; original: string; translation: string; speaker?: string },
) =>
  mockStore.update((data) => {
    const record = data.records.find((item) => item.id === id);
    if (!record) return;
    const endMs = Date.now() - Date.parse(record.startedAt);
    const startMs = Math.max(0, endMs - 1_500 - segment.original.length * 40);
    record.segments.push({ id: `${id}-s${record.segments.length}`, startMs, endMs, ...segment });
    if (!record.title) record.title = titleFrom(segment.original);
  });

/** The server marks a record finished when the socket closes; an empty record is dropped. */
const finishRecording = (id: string) =>
  mockStore.update((data) => {
    const record = data.records.find((item) => item.id === id);
    if (!record) return;
    if (record.segments.length === 0) {
      data.records = data.records.filter((item) => item !== record);
      return;
    }
    record.endedAt = new Date().toISOString();
  });

/**
 * Writes each `final` into the mock records as the session emits it, the way the server records
 * as it sends. Wraps the mock session, and the real one when only `conversations` is mocked.
 */
export const withMockRecording = (live: LiveApi): LiveApi => ({
  monitor: live.monitor,
  async open(options) {
    const conversationId = await startRecording(options);
    const session = await live
      .open({
        ...options,
        save: false,
        handlers: {
          ...options.handlers,
          onFinal: (message) => {
            if (conversationId && message.original.trim()) {
              appendSegment(conversationId, {
                language: message.language,
                original: message.original.trim(),
                translation: message.translation.trim(),
                speaker: message.speaker,
              }).catch(logFailure);
            }
            options.handlers.onFinal(message);
          },
        },
      })
      .catch(async (error: unknown) => {
        if (conversationId) await finishRecording(conversationId);
        throw error;
      });
    return {
      conversationId,
      get isOpen() {
        return session.isOpen;
      },
      sendAudio: (pcm) => session.sendAudio(pcm),
      configure: (languages) => session.configure(languages),
      logSpeech: (entry) => session.logSpeech(entry),
      close: async () => {
        await session.close();
        if (conversationId) await finishRecording(conversationId).catch(logFailure);
      },
    };
  },
});
