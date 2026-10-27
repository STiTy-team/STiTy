import { AsrSession, SessionOpenError, SessionRefusedError } from './asr-session';

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.CONNECTING;
  binaryType = 'blob';
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  send = jest.fn();
  close = jest.fn(() => {
    this.readyState = 3;
  });

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  receive(message: object) {
    this.onmessage?.({ data: JSON.stringify(message) });
  }

  accept() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
    this.receive({ type: 'hello' });
  }

  serverClose() {
    this.readyState = 3;
    this.onclose?.({ code: 1000, reason: '' });
  }

  sentTypes() {
    return this.send.mock.calls.map(([data]) => JSON.parse(data as string).type as string);
  }
}

const socket = () => FakeWebSocket.instances[FakeWebSocket.instances.length - 1]!;

const handlers = () => ({
  onPartial: jest.fn(),
  onFinal: jest.fn(),
  onCapacity: jest.fn(),
  onLost: jest.fn(),
});

const openSession = async () => {
  const on = handlers();
  const session = new AsrSession('wss://asr.test/api/v1/ws', on);
  const opened = session.open({ lang: 'ko', targetLang: 'en' });
  socket().accept();
  socket().receive({ type: 'ready' });
  await opened;
  return { session, on, ws: socket() };
};

beforeEach(() => {
  jest.useFakeTimers();
  FakeWebSocket.instances = [];
  (globalThis as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;
});
afterEach(() => jest.useRealTimers());

describe('closing a session', () => {
  it('delivers the final the server flushes after stop, then resolves when it closes', async () => {
    const { session, on, ws } = await openSession();

    let closed = false;
    const closing = session.close().then(() => (closed = true));
    expect(ws.sentTypes()).toContain('stop');
    expect(ws.close).not.toHaveBeenCalled();
    expect(session.isOpen).toBe(false);

    ws.receive({ type: 'final', original: '마지막', translation: 'Last', language: 'ko' });
    expect(on.onFinal).toHaveBeenCalledWith(expect.objectContaining({ original: '마지막' }));

    await Promise.resolve();
    expect(closed).toBe(false);
    ws.serverClose();
    await closing;
    expect(on.onLost).not.toHaveBeenCalled();
  });

  it('closes the socket itself when the server never does', async () => {
    const { session, on, ws } = await openSession();

    const closing = session.close();
    jest.advanceTimersByTime(2_000);
    await closing;

    expect(ws.close).toHaveBeenCalled();
    ws.receive({ type: 'final', original: 'too late', language: 'ko' });
    expect(on.onFinal).not.toHaveBeenCalled();
  });

  it('still reports a lost connection while streaming', async () => {
    const { on, ws } = await openSession();
    ws.serverClose();
    expect(on.onLost).toHaveBeenCalledTimes(1);
  });
});

describe('opening a session', () => {
  it('rejects at once with the server reason when it answers start with an error', async () => {
    const session = new AsrSession('wss://asr.test/api/v1/ws', handlers());
    const opened = session.open({ lang: 'ko', targetLang: 'en' });
    socket().accept();

    socket().receive({ type: 'error', message: 'No free slot', code: 'capacity' });

    const error = await opened.catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(SessionRefusedError);
    expect(error).toMatchObject({ message: 'No free slot', code: 'capacity' });
    expect(socket().close).toHaveBeenCalled();
  });

  it('rejects when closed before the server is ready', async () => {
    const session = new AsrSession('wss://asr.test/api/v1/ws', handlers());
    const opened = session.open({ lang: 'ko', targetLang: 'en' });
    socket().accept();

    void session.close();
    socket().serverClose();

    await expect(opened).rejects.toBeInstanceOf(SessionOpenError);
  });
});
