import { ServerMonitor, type ServerStatus } from './server-monitor';

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

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  close() {
    this.readyState = 3;
  }

  acceptWithHello() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
    this.onmessage?.({ data: JSON.stringify({ type: 'hello' }) });
  }

  drop() {
    this.readyState = 3;
    this.onclose?.({ code: 1006, reason: '' });
  }
}

const latest = () => FakeWebSocket.instances[FakeWebSocket.instances.length - 1]!;

beforeEach(() => {
  jest.useFakeTimers();
  FakeWebSocket.instances = [];
  (globalThis as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;
});

afterEach(() => jest.useRealTimers());

describe('ServerMonitor', () => {
  it('becomes ready on hello and reconnects with backoff after a drop', () => {
    const monitor = new ServerMonitor('wss://server/api/v1/ws');
    const statuses: ServerStatus[] = [];
    monitor.subscribe(({ status }) => statuses.push(status));

    monitor.start();
    latest().acceptWithHello();
    expect(monitor.current.status).toBe('ready');

    latest().drop();
    expect(monitor.current.status).toBe('connecting');
    expect(FakeWebSocket.instances).toHaveLength(1);

    jest.advanceTimersByTime(1_000);
    expect(FakeWebSocket.instances).toHaveLength(2);

    latest().acceptWithHello();
    expect(statuses).toEqual(['connecting', 'ready', 'connecting', 'ready']);
    monitor.stop();
  });

  it('reports unreachable after repeated failures but keeps trying', () => {
    const monitor = new ServerMonitor('wss://server/api/v1/ws');
    monitor.start();

    for (let attempt = 0; attempt < 4; attempt += 1) {
      latest().drop();
      jest.advanceTimersByTime(30_000);
    }

    expect(monitor.current.status).toBe('unreachable');
    expect(FakeWebSocket.instances.length).toBeGreaterThan(4);
    monitor.stop();
  });

  it('resolves waitUntilReady when the server answers, and false on timeout', async () => {
    const monitor = new ServerMonitor('wss://server/api/v1/ws');
    monitor.start();

    const ready = monitor.waitUntilReady(10_000);
    latest().acceptWithHello();
    await expect(ready).resolves.toBe(true);

    latest().drop();
    const timedOut = monitor.waitUntilReady(500);
    jest.advanceTimersByTime(500);
    await expect(timedOut).resolves.toBe(false);
    monitor.stop();
  });

  it('relays capacity notices', () => {
    const monitor = new ServerMonitor('wss://server/api/v1/ws');
    monitor.start();
    latest().acceptWithHello();
    latest().onmessage?.({ data: JSON.stringify({ type: 'capacity', active: 20, max: 20 }) });

    expect(monitor.current.capacity).toEqual({ active: 20, max: 20 });
    monitor.stop();
  });
});
