import AsyncStorage from '@react-native-async-storage/async-storage';
import { z } from 'zod';

import type { Profile } from '@/api/types';
import type { RecordedSegment, TranscriptDocument } from '@/domain/records';
import { createLogger } from '@/lib/logger';

import { seedMockData } from './fixtures';

/** Its own key prefix, so clearing mock data never touches real preferences. */
export const MOCK_STORAGE_KEY = 'stity.mock.v1';

export type MockRecord = {
  id: string;
  title: string;
  startedAt: string;
  endedAt: string | null;
  languages: string[];
  myLanguage: string;
  segments: RecordedSegment[];
  /** Meeting fixtures label speakers; two-person records infer them from the language. */
  speakers?: { id: string; label: string; language: string; isMe: boolean }[];
  summary: TranscriptDocument['summary'];
  chapters?: TranscriptDocument['chapters'];
  glossary?: TranscriptDocument['glossary'];
};

export type MockData = {
  signedIn: boolean;
  profile: Profile;
  records: MockRecord[];
};

/** Checks the shape the adapters index into; the fixtures' own fields pass through untouched. */
const mockDataSchema = z.looseObject({
  signedIn: z.boolean(),
  profile: z.looseObject({ id: z.string(), displayName: z.string() }),
  records: z.array(
    z.looseObject({ id: z.string(), startedAt: z.string(), segments: z.array(z.unknown()) }),
  ),
});

const log = createLogger('mock-store');

const parse = (raw: string): MockData | null => {
  try {
    const json: unknown = JSON.parse(raw);
    return mockDataSchema.safeParse(json).success ? (json as MockData) : null;
  } catch {
    return null;
  }
};

let cache: MockData | null = null;
let queue: Promise<unknown> = Promise.resolve();

/** Unreadable data is replaced with fresh fixtures instead of failing every mocked call. */
const load = async (): Promise<MockData> => {
  if (cache) return cache;
  const raw = await AsyncStorage.getItem(MOCK_STORAGE_KEY);
  const stored = raw ? parse(raw) : null;
  if (raw && !stored) log.warn('mock data was unreadable and has been reseeded');
  cache = stored ?? seedMockData(Date.now());
  if (!stored) await AsyncStorage.setItem(MOCK_STORAGE_KEY, JSON.stringify(cache));
  return cache;
};

/** Runs after everything queued before it; a failure reaches its caller but not the next step. */
const enqueue = <T>(run: () => Promise<T>): Promise<T> => {
  const next = queue.then(run);
  queue = next.catch(() => undefined);
  return next;
};

/** Mock data in AsyncStorage, so a record survives an app restart. Writes run one at a time. */
export const mockStore = {
  read: (): Promise<MockData> => enqueue(load),
  update: <T>(mutate: (data: MockData) => T): Promise<T> =>
    enqueue(async () => {
      const data = await load();
      const result = mutate(data);
      await AsyncStorage.setItem(MOCK_STORAGE_KEY, JSON.stringify(data));
      return result;
    }),
  reset: (): Promise<void> =>
    enqueue(async () => {
      cache = null;
      await AsyncStorage.removeItem(MOCK_STORAGE_KEY);
    }),
};
