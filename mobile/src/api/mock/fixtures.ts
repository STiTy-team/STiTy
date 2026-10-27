import type { RecordedSegment } from '@/domain/records';

import { SCRIPTS } from './scripts';
import type { MockData, MockRecord } from './store';

const DAY_MS = 86_400_000;
const MINUTE_MS = 60_000;

type FixtureSpec = Omit<
  MockRecord,
  'segments' | 'startedAt' | 'endedAt' | 'languages' | 'myLanguage'
> & {
  script: keyof typeof SCRIPTS;
  daysAgo: number;
  /** Hour of the day the record starts at. */
  hour: number;
  segmentCount: number;
  speakerCycle?: string[];
};

/** Walks the fixture's script line by line, with a pause pattern that also exercises paragraphs. */
const segmentsFor = (spec: FixtureSpec): { segments: RecordedSegment[]; durationMs: number } => {
  const script = SCRIPTS[spec.script];
  const segments: RecordedSegment[] = [];
  let cursor = 0;
  for (let index = 0; index < spec.segmentCount; index += 1) {
    const [language, original, translation] = script[index % script.length]!;
    const length = 1_800 + original.length * 55;
    segments.push({
      id: `${spec.id}-s${index}`,
      language,
      original,
      translation,
      startMs: cursor,
      endMs: cursor + length,
      speaker: spec.speakerCycle?.[index % spec.speakerCycle.length],
    });
    cursor += length + (index % 3 === 2 ? 3_500 : 900);
  }
  return { segments, durationMs: cursor };
};

const SPECS: FixtureSpec[] = [
  {
    id: 'rec-cafe',
    title: 'Coffee order at the counter',
    script: 'cafe',
    daysAgo: 0,
    hour: 9,
    segmentCount: 8,
    summary: { status: 'unavailable' },
  },
  {
    id: 'rec-sync',
    title: 'Weekly product sync',
    script: 'meeting',
    daysAgo: 1,
    hour: 14,
    segmentCount: 24,
    speakerCycle: ['a', 'b', 'c', 'a'],
    speakers: [
      { id: 'a', label: 'Speaker A', language: 'ko', isMe: true },
      { id: 'b', label: 'Speaker B', language: 'en', isMe: false },
      { id: 'c', label: 'Speaker C', language: 'en', isMe: false },
    ],
    summary: {
      status: 'ready',
      overview:
        'Reviewed feedback since the launch. Android sign-in is the main issue; the records release moves by a week.',
      keyPoints: [
        'Most reports concern the Android sign-in screen',
        'The keyboard covers the code field',
      ],
      decisions: ['Records release moves to next Monday'],
      actionItems: [
        { text: 'Fix the keyboard covering the code field', owner: 'Speaker A' },
        { text: 'Confirm the summary schedule with the server team', owner: 'Speaker B' },
      ],
    },
    chapters: [
      { title: 'User feedback', startMs: 0 },
      { title: 'Release date', startMs: 4 * MINUTE_MS },
      { title: 'Noisy place testing', startMs: 9 * MINUTE_MS },
    ],
  },
  {
    id: 'rec-lecture',
    title: 'Lecture: how speech recognition works',
    script: 'lecture',
    daysAgo: 2,
    hour: 10,
    segmentCount: 30,
    summary: {
      status: 'ready',
      overview: 'An introduction to streaming speech recognition and when translations appear.',
      keyPoints: [
        'Audio is cut into short frames and turned into numbers',
        'Streaming shows text before the sentence ends',
        'Translation is attached once a sentence is committed',
      ],
    },
    chapters: [
      { title: 'From sound to numbers', startMs: 0 },
      { title: 'Streaming results', startMs: 3 * MINUTE_MS },
    ],
    glossary: [
      { term: '음성 인식', translation: 'speech recognition' },
      { term: '스트리밍', translation: 'streaming' },
      { term: '확정', translation: 'commit' },
    ],
  },
  {
    id: 'rec-market',
    title: 'Market stall',
    script: 'market',
    daysAgo: 3,
    hour: 18,
    segmentCount: 5,
    summary: { status: 'unavailable' },
  },
  {
    id: 'rec-palace',
    title: 'Gyeongbokgung walk',
    script: 'palace',
    daysAgo: 4,
    hour: 11,
    segmentCount: 18,
    summary: { status: 'unavailable' },
    chapters: [
      { title: 'Gwanghwamun', startMs: 0 },
      { title: 'Geunjeongjeon', startMs: 2 * MINUTE_MS },
      { title: 'Gyeonghoeru', startMs: 5 * MINUTE_MS },
    ],
  },
  {
    id: 'rec-directions',
    title: 'Asking the way to the station',
    script: 'cafe',
    daysAgo: 6,
    hour: 16,
    segmentCount: 12,
    summary: { status: 'unavailable' },
  },
  {
    id: 'rec-retro',
    title: 'Sprint retrospective',
    script: 'meeting',
    daysAgo: 8,
    hour: 15,
    segmentCount: 16,
    speakerCycle: ['a', 'b'],
    speakers: [
      { id: 'a', label: 'Speaker A', language: 'ko', isMe: true },
      { id: 'b', label: 'Speaker B', language: 'en', isMe: false },
    ],
    summary: { status: 'pending' },
  },
  {
    id: 'rec-planning',
    title: 'Quarterly planning (60 min)',
    script: 'meeting',
    daysAgo: 12,
    hour: 13,
    segmentCount: 600,
    speakerCycle: ['a', 'b', 'c'],
    speakers: [
      { id: 'a', label: 'Speaker A', language: 'ko', isMe: true },
      { id: 'b', label: 'Speaker B', language: 'en', isMe: false },
      { id: 'c', label: 'Speaker C', language: 'en', isMe: false },
    ],
    summary: { status: 'unavailable' },
  },
];

/** One user and eight records of different lengths and speakers, ko↔en, dated relative to `now`. */
export const seedMockData = (now: number): MockData => {
  const today = new Date(now);
  today.setHours(0, 0, 0, 0);

  const records = SPECS.map((spec): MockRecord => {
    const {
      daysAgo,
      hour,
      script: _script,
      segmentCount: _count,
      speakerCycle: _cycle,
      ...rest
    } = spec;
    const { segments, durationMs } = segmentsFor(spec);
    const start = today.getTime() - daysAgo * DAY_MS + hour * 3_600_000;
    return {
      ...rest,
      startedAt: new Date(start).toISOString(),
      endedAt: new Date(start + durationMs).toISOString(),
      languages: ['ko', 'en'],
      myLanguage: 'ko',
      segments,
    };
  });

  return {
    signedIn: false,
    profile: {
      id: 'mock-user',
      displayName: 'Preview user',
      email: 'preview@example.com',
      myLanguage: 'ko',
      retention: '1y',
    },
    records,
  };
};
