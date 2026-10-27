import { encodeClientMessage, parseServerMessage } from './protocol';

describe('parseServerMessage', () => {
  it('parses a final and fills missing fields', () => {
    expect(
      parseServerMessage(JSON.stringify({ type: 'final', original: 'hi', language: 'en' })),
    ).toEqual({ type: 'final', original: 'hi', translation: '', language: 'en' });
  });

  it('keeps an empty partial, which clears the live line', () => {
    expect(parseServerMessage(JSON.stringify({ type: 'partial', text: '' }))).toMatchObject({
      type: 'partial',
      text: '',
    });
  });

  it('ignores binary frames, bad JSON, unknown and invalid messages', () => {
    expect(parseServerMessage(new ArrayBuffer(4))).toBeNull();
    expect(parseServerMessage('{not json')).toBeNull();
    expect(parseServerMessage(JSON.stringify({ type: 'config_ok' }))).toBeNull();
    expect(parseServerMessage(JSON.stringify({ type: 'capacity', active: 1, max: 0 }))).toBeNull();
  });
});

describe('error', () => {
  it('parses the server reason and code', () => {
    expect(
      parseServerMessage(JSON.stringify({ type: 'error', message: 'busy', code: 'capacity' })),
    ).toEqual({ type: 'error', message: 'busy', code: 'capacity' });
    expect(parseServerMessage(JSON.stringify({ type: 'error' }))).toEqual({
      type: 'error',
      message: '',
    });
  });
});

describe('session additions', () => {
  it('reads the conversation id from ready, and null from servers that send none', () => {
    expect(
      parseServerMessage(JSON.stringify({ type: 'ready', conversationId: 'c-1' })),
    ).toMatchObject({ conversationId: 'c-1' });
    expect(parseServerMessage(JSON.stringify({ type: 'ready' }))).toMatchObject({
      conversationId: null,
    });
  });

  it('keeps segment ids and speakers on a final', () => {
    expect(
      parseServerMessage(
        JSON.stringify({ type: 'final', original: 'hi', segmentId: 7, speaker: 'spk-2' }),
      ),
    ).toMatchObject({ segmentId: '7', speaker: 'spk-2' });
  });
});

describe('encodeClientMessage', () => {
  it('serializes the start handshake', () => {
    expect(
      JSON.parse(
        encodeClientMessage({ type: 'start', lang: 'en', targetLang: 'ko', translate: true }),
      ),
    ).toEqual({ type: 'start', lang: 'en', targetLang: 'ko', translate: true });
  });
});
