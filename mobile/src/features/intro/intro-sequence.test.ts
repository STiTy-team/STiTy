import { greetingSequence } from './intro-sequence';

describe('greetingSequence', () => {
  it('ends on the greeting in the app language and never repeats it', () => {
    const words = greetingSequence('ko');
    expect(words.at(-1)).toBe('안녕하세요');
    expect(words.filter((word) => word === '안녕하세요')).toHaveLength(1);
  });

  it('adds the app language when it is not a conversation language', () => {
    const words = greetingSequence('fr');
    expect(words.at(-1)).toBe('Bonjour');
    expect(words).toContain('Hello');
  });
});
