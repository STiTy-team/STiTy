import { buildParagraphs, formatOffset, type RecordedSegment } from './records';

const segment = (
  id: string,
  language: string,
  startMs: number,
  endMs: number,
): RecordedSegment => ({
  id,
  language,
  original: `o${id}`,
  translation: `t${id}`,
  startMs,
  endMs,
});

describe('buildParagraphs', () => {
  it('joins one speaker’s sentences less than 2 s apart', () => {
    const paragraphs = buildParagraphs(
      [
        segment('1', 'ko', 0, 1_000),
        segment('2', 'ko', 2_500, 3_000),
        segment('3', 'en', 3_500, 4_000),
        segment('4', 'en', 9_000, 10_000),
      ],
      (s) => s.language,
    );
    expect(paragraphs.map((p) => [p.speakerId, p.original, p.translation])).toEqual([
      ['ko', 'o1 o2', 't1 t2'],
      ['en', 'o3', 't3'],
      ['en', 'o4', 't4'],
    ]);
    expect(paragraphs[0]).toMatchObject({ startMs: 0, endMs: 3_000 });
  });

  it('starts a new paragraph when the same speaker switches language', () => {
    const paragraphs = buildParagraphs(
      [segment('1', 'ko', 0, 1_000), segment('2', 'en', 1_200, 2_000)],
      () => 'me',
    );
    expect(paragraphs.map((p) => p.language)).toEqual(['ko', 'en']);
  });
});

describe('formatOffset', () => {
  it('shows minutes and seconds, and hours past an hour', () => {
    expect(formatOffset(252_000)).toBe('04:12');
    expect(formatOffset(3_852_000)).toBe('1:04:12');
  });
});
