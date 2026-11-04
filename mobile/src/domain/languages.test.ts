import { normalizeLanguageCode, toLanguageTag } from './languages';

describe('normalizeLanguageCode', () => {
  it('accepts codes and English names', () => {
    expect(normalizeLanguageCode('ko')).toBe('ko');
    expect(normalizeLanguageCode('Korean')).toBe('ko');
    expect(normalizeLanguageCode(' Japanese ')).toBe('ja');
    expect(normalizeLanguageCode('en-US')).toBe('en');
  });
});

describe('toLanguageTag', () => {
  it('maps to a BCP-47 tag and passes unknown codes through', () => {
    expect(toLanguageTag('zh')).toBe('zh-CN');
    expect(toLanguageTag('xx')).toBe('xx');
  });
});
