import { normalizeServerUrl } from './server-override';

describe('normalizeServerUrl', () => {
  it('turns a scanned or typed address into a WebSocket origin', () => {
    expect(normalizeServerUrl(' https://a-b.trycloudflare.com/ ')).toBe(
      'wss://a-b.trycloudflare.com',
    );
    expect(normalizeServerUrl('a-b.trycloudflare.com')).toBe('wss://a-b.trycloudflare.com');
    expect(normalizeServerUrl('http://192.168.0.2:8765')).toBe('ws://192.168.0.2:8765');
    expect(normalizeServerUrl('ws://192.168.0.2:8765')).toBe('ws://192.168.0.2:8765');
  });

  it('refuses anything that is not an origin', () => {
    for (const input of ['', 'ftp://host', 'wss://host/api/v1/ws', 'wss://host?x=1', 'wss://']) {
      expect(normalizeServerUrl(input)).toBeNull();
    }
  });
});
