import { parseInlineMarkup } from './inline-markup';

describe('parseInlineMarkup', () => {
  it('splits bold segments', () => {
    expect(parseInlineMarkup('Audio is <bold>not stored</bold> after.')).toEqual([
      { text: 'Audio is ', bold: false },
      { text: 'not stored', bold: true },
      { text: ' after.', bold: false },
    ]);
  });

  it('returns plain text unchanged', () => {
    expect(parseInlineMarkup('plain')).toEqual([{ text: 'plain', bold: false }]);
  });
});
