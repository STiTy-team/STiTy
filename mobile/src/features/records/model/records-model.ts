import type { RecordSummary, TranscriptParagraph } from '@/domain/records';

export type DaySection = { key: string; date: Date; data: RecordSummary[] };

const dayKey = (date: Date) => `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;

/** Newest first, one section per local calendar day. */
export const groupByDay = (records: readonly RecordSummary[]): DaySection[] => {
  const sections: DaySection[] = [];
  for (const record of records) {
    const date = new Date(record.startedAt);
    const key = dayKey(date);
    const last = sections[sections.length - 1];
    if (last?.key === key) last.data.push(record);
    else sections.push({ key, date, data: [record] });
  }
  return sections;
};

/** 0 for today, 1 for yesterday, counted in local calendar days. */
export const daysAgo = (date: Date, now: Date = new Date()): number => {
  const start = (value: Date) => new Date(value.getFullYear(), value.getMonth(), value.getDate());
  return Math.round((start(now).getTime() - start(date).getTime()) / 86_400_000);
};

export type TranscriptMatch = { paragraphId: string; field: 'original' | 'translation' };

/** Every paragraph field that contains the query, in reading order, original before translation. */
export const findMatches = (
  paragraphs: readonly TranscriptParagraph[],
  query: string,
): TranscriptMatch[] => {
  const needle = query.trim().toLowerCase();
  if (!needle) return [];
  return paragraphs.flatMap((paragraph) =>
    (['original', 'translation'] as const)
      .filter((field) => paragraph[field].toLowerCase().includes(needle))
      .map((field) => ({ paragraphId: paragraph.id, field })),
  );
};

export type HighlightPart = { text: string; hit: boolean };

/** Splits text around each case-insensitive occurrence of the query. */
export const highlight = (text: string, query: string): HighlightPart[] => {
  const needle = query.trim().toLowerCase();
  if (!needle) return [{ text, hit: false }];
  const parts: HighlightPart[] = [];
  const lower = text.toLowerCase();
  let from = 0;
  for (let at = lower.indexOf(needle); at !== -1; at = lower.indexOf(needle, from)) {
    if (at > from) parts.push({ text: text.slice(from, at), hit: false });
    parts.push({ text: text.slice(at, at + needle.length), hit: true });
    from = at + needle.length;
  }
  if (from < text.length) parts.push({ text: text.slice(from), hit: false });
  return parts;
};

/** The transcript as plain text for sharing: one paragraph per speaker turn. */
export const transcriptText = (
  title: string,
  paragraphs: readonly TranscriptParagraph[],
  label: (paragraph: TranscriptParagraph) => string,
  text: (paragraph: TranscriptParagraph) => string,
): string =>
  [title, ...paragraphs.map((paragraph) => `${label(paragraph)}\n${text(paragraph)}`)].join('\n\n');
