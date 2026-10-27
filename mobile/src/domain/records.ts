export const RETENTION_OPTIONS = ['30d', '1y', 'forever'] as const;

export type Retention = (typeof RETENTION_OPTIONS)[number];

export type RecordSummary = {
  id: string;
  title: string;
  startedAt: string;
  durationMs: number;
  languages: string[];
  firstLine: string;
};

export type TranscriptSpeaker = { id: string; label: string; language: string; isMe: boolean };

export type TranscriptParagraph = {
  id: string;
  speakerId: string;
  startMs: number;
  endMs: number;
  language: string;
  original: string;
  translation: string;
};

export type TranscriptSummary = {
  status: 'ready' | 'pending' | 'unavailable';
  overview?: string;
  keyPoints?: string[];
  decisions?: string[];
  actionItems?: { text: string; owner?: string }[];
};

/** A saved conversation as it is read afterwards: paragraphs per speaker turn, not bubbles. */
export type TranscriptDocument = {
  id: string;
  title: string;
  startedAt: string;
  endedAt: string;
  languages: string[];
  myLanguage: string;
  speakers: TranscriptSpeaker[];
  paragraphs: TranscriptParagraph[];
  summary: TranscriptSummary;
  chapters?: { title: string; startMs: number }[];
  glossary?: { term: string; translation: string }[];
};

/** One committed sentence, as the server records it while it streams results. */
export type RecordedSegment = {
  id: string;
  language: string;
  original: string;
  translation: string;
  startMs: number;
  endMs: number;
  speaker?: string;
};

const PARAGRAPH_GAP_MS = 2_000;

/**
 * Joins consecutive sentences by the same speaker, in the same language, less than 2 s apart into
 * one paragraph. The live server returns finished paragraphs; the mock builds them with this.
 */
export const buildParagraphs = (
  segments: readonly RecordedSegment[],
  speakerOf: (segment: RecordedSegment) => string,
): TranscriptParagraph[] => {
  const paragraphs: TranscriptParagraph[] = [];
  for (const segment of segments) {
    const speakerId = speakerOf(segment);
    const last = paragraphs[paragraphs.length - 1];
    if (
      last &&
      last.speakerId === speakerId &&
      last.language === segment.language &&
      segment.startMs - last.endMs < PARAGRAPH_GAP_MS
    ) {
      last.endMs = segment.endMs;
      last.original = joinSentences(last.original, segment.original);
      last.translation = joinSentences(last.translation, segment.translation);
      continue;
    }
    paragraphs.push({
      id: segment.id,
      speakerId,
      startMs: segment.startMs,
      endMs: segment.endMs,
      language: segment.language,
      original: segment.original,
      translation: segment.translation,
    });
  }
  return paragraphs;
};

const joinSentences = (a: string, b: string): string => (a && b ? `${a} ${b}` : a || b);

/** "04:12", or "1:04:12" past an hour. */
export const formatOffset = (ms: number): string => {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
  const seconds = String(total % 60).padStart(2, '0');
  return hours > 0 ? `${hours}:${minutes}:${seconds}` : `${minutes}:${seconds}`;
};
