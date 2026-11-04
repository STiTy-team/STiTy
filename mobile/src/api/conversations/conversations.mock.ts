import { ApiError } from '@/api/errors';
import { mockCall } from '@/api/mock/runtime';
import { mockStore, type MockRecord } from '@/api/mock/store';
import { pageSchema, recordSummarySchema, transcriptDocumentSchema } from '@/api/schemas';
import type { ConversationsApi } from '@/api/types';
import { buildParagraphs, type RecordSummary, type TranscriptDocument } from '@/domain/records';

const PAGE_SIZE = 20;

const durationOf = (record: MockRecord): number => {
  const last = record.segments[record.segments.length - 1];
  if (record.endedAt) return Date.parse(record.endedAt) - Date.parse(record.startedAt);
  return last?.endMs ?? 0;
};

const summarize = (record: MockRecord): RecordSummary => ({
  id: record.id,
  title: record.title,
  startedAt: record.startedAt,
  durationMs: Math.max(0, durationOf(record)),
  languages: record.languages,
  firstLine: record.segments[0]?.original ?? '',
});

const matches = (record: MockRecord, search: string): boolean => {
  const needle = search.toLowerCase();
  return (
    record.title.toLowerCase().includes(needle) ||
    record.segments.some(
      (segment) =>
        segment.original.toLowerCase().includes(needle) ||
        segment.translation.toLowerCase().includes(needle),
    )
  );
};

/** Two people sharing a phone are told apart by language, as playback does: mine is me. */
const speakersOf = (record: MockRecord): TranscriptDocument['speakers'] =>
  record.speakers ??
  record.languages.map((language) => ({
    id: language,
    label: language.toUpperCase(),
    language,
    isMe: language === record.myLanguage,
  }));

const toDocument = (record: MockRecord): TranscriptDocument => {
  const speakers = speakersOf(record);
  return {
    id: record.id,
    title: record.title,
    startedAt: record.startedAt,
    endedAt: record.endedAt ?? new Date().toISOString(),
    languages: record.languages,
    myLanguage: record.myLanguage,
    speakers,
    paragraphs: buildParagraphs(record.segments, (segment) =>
      record.speakers && segment.speaker ? segment.speaker : segment.language,
    ),
    summary: record.summary,
    chapters: record.chapters,
    glossary: record.glossary,
  };
};

const findRecord = (records: MockRecord[], id: string): MockRecord => {
  const record = records.find((item) => item.id === id);
  if (!record) throw new ApiError('notFound');
  return record;
};

export const createMockConversations = (): ConversationsApi => ({
  list: ({ cursor, search }) =>
    mockCall(pageSchema(recordSummarySchema), async () => {
      const { records } = await mockStore.read();
      const visible = records
        .filter((record) => record.segments.length > 0)
        .filter((record) => !search?.trim() || matches(record, search.trim()))
        .sort((a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt));
      const start = Number(cursor ?? 0);
      const end = start + PAGE_SIZE;
      return {
        items: visible.slice(start, end).map(summarize),
        nextCursor: end < visible.length ? String(end) : null,
      };
    }),
  get: (id) =>
    mockCall(transcriptDocumentSchema, async () =>
      toDocument(findRecord((await mockStore.read()).records, id)),
    ),
  rename: (id, title) =>
    mockCall(null, () =>
      mockStore.update((data) => {
        findRecord(data.records, id).title = title;
      }),
    ),
  remove: (id) =>
    mockCall(null, () =>
      mockStore.update((data) => {
        findRecord(data.records, id);
        data.records = data.records.filter((record) => record.id !== id);
      }),
    ),
});
