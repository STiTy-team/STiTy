import { getConfig } from 'expo/config';

import { parseWebUrl } from './app-info';

describe('parseWebUrl', () => {
  it('has no address in a local run', () => {
    expect(parseWebUrl(undefined)).toBe('');
  });

  it('refuses something that is not a web address', () => {
    expect(() => parseWebUrl('wss://example.com')).toThrow();
  });

  it('accepts what the real app config produces', () => {
    const { exp } = getConfig(process.cwd(), { skipSDKVersionRequirement: true });
    expect(parseWebUrl(exp.extra?.webUrl)).toMatch(/^https?:\/\//);
  });
});
