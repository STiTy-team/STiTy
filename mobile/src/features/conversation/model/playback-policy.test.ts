import { planPlayback, sessionRoute, translationTarget } from './playback-policy';

const pair = { myLanguage: 'en', peerLanguage: 'ko' };

describe('translationTarget', () => {
  it('translates each side into the other side', () => {
    expect(translationTarget('en', pair)).toBe('ko');
    expect(translationTarget('ko', pair)).toBe('en');
  });

  it('translates a third language into mine', () => {
    expect(translationTarget('ja', pair)).toBe('en');
  });
});

describe('planPlayback', () => {
  it('speaks everything through the speaker in speaker mode', () => {
    expect(planPlayback('en', pair, 'speaker')).toEqual({ language: 'ko', route: 'speaker' });
    expect(planPlayback('ko', pair, 'speaker')).toEqual({ language: 'en', route: 'speaker' });
  });

  it('speaks only the peer, into my earphone, in earphone mode', () => {
    expect(planPlayback('en', pair, 'earphone')).toBeNull();
    expect(planPlayback('ko', pair, 'earphone')).toEqual({ language: 'en', route: 'earpiece' });
  });

  it('splits outputs by listener in both mode', () => {
    expect(planPlayback('ko', pair, 'both')).toEqual({ language: 'en', route: 'earpiece' });
    expect(planPlayback('en', pair, 'both')).toEqual({ language: 'ko', route: 'speaker' });
  });
});

describe('sessionRoute', () => {
  it('uses the earpiece only in earphone mode', () => {
    expect(sessionRoute('earphone')).toBe('earpiece');
    expect(sessionRoute('speaker')).toBe('speaker');
    expect(sessionRoute('both')).toBe('speaker');
  });
});
