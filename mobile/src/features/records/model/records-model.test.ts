import type { RecordSummary, TranscriptParagraph } from '@/domain/records';

import { daysAgo, findMatches, groupByDay, highlight } from './records-model';

const record = (id: string, startedAt: string): RecordSummary => ({
  id,
  title: id,
  startedAt,
  durationMs: 0,
  languages: ['ko', 'en'],
  firstLine: '',
});

const paragraph = (id: string, original: string, translation: string): TranscriptParagraph => ({
  id,
  speakerId: 'ko',
  startMs: 0,
  endMs: 0,
  language: 'ko',
  original,
  translation,
});

describe('groupByDay', () => {
  it('starts a section on each new day', () => {
    const sections = groupByDay([
      record('a', '2026-10-27T15:00:00'),
      record('b', '2026-10-27T09:00:00'),
      record('c', '2026-10-25T20:00:00'),
    ]);
    expect(sections.map((section) => section.data.map((item) => item.id))).toEqual([
      ['a', 'b'],
      ['c'],
    ]);
  });
});

describe('daysAgo', () => {
  it('counts calendar days, not 24-hour spans', () => {
    const now = new Date('2026-10-27T00:30:00');
    expect(daysAgo(new Date('2026-10-26T23:50:00'), now)).toBe(1);
    expect(daysAgo(new Date('2026-10-27T00:10:00'), now)).toBe(0);
  });
});

describe('findMatches', () => {
  it('finds text in both the original and the translation, in reading order', () => {
    const matches = findMatches(
      [paragraph('1', '회의 시작', 'Meeting starts'), paragraph('2', 'no', 'The meeting ends')],
      'meeting',
    );
    expect(matches).toEqual([
      { paragraphId: '1', field: 'translation' },
      { paragraphId: '2', field: 'translation' },
    ]);
    expect(findMatches([paragraph('1', '회의 시작', 'x')], '회의')).toEqual([
      { paragraphId: '1', field: 'original' },
    ]);
  });
});

describe('highlight', () => {
  it('marks every occurrence regardless of case', () => {
    expect(highlight('Go go GO', 'go')).toEqual([
      { text: 'Go', hit: true },
      { text: ' ', hit: false },
      { text: 'go', hit: true },
      { text: ' ', hit: false },
      { text: 'GO', hit: true },
    ]);
  });
});
