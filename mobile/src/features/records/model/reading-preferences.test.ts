import { useReadingPreferences } from './reading-preferences';

const store = useReadingPreferences;

describe('reading preferences', () => {
  it('keeps the reading size within the scale', () => {
    store.getState().setReadingScale(99);
    expect(store.getState().readingScale).toBe(4);
    store.getState().setReadingScale(-3);
    expect(store.getState().readingScale).toBe(0);
  });

  it('drops an out-of-range saved size to the default', () => {
    const current = { ...store.getState(), readingScale: 1 };
    const restored = store.persist
      .getOptions()
      .merge?.({ readingScale: 42, transcriptView: 'bilingual' }, current);
    expect(restored).toMatchObject({ readingScale: 1, transcriptView: 'bilingual' });
  });
});
