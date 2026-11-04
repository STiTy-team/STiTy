import { exponentialBackoff } from '@/lib/backoff';
import { createLogger } from '@/lib/logger';

import type { CapacityMessage } from './protocol';
import { WebSocketConnection } from './websocket-connection';

export type ServerStatus = 'idle' | 'connecting' | 'ready' | 'unreachable';

export type ServerSnapshot = {
  status: ServerStatus;
  capacity: { active: number; max: number } | null;
};

const HELLO_TIMEOUT_MS = 5_000;
const RETRY_BASE_MS = 1_000;
const RETRY_MAX_MS = 30_000;
const UNREACHABLE_AFTER_ATTEMPTS = 4;

const log = createLogger('server-monitor');

/**
 * Holds one standby connection while no conversation is running. It tells the UI whether the
 * server is up, relays capacity notices, and keeps a server started with idle shutdown awake.
 */
export class ServerMonitor {
  private connection: WebSocketConnection | null = null;
  private helloTimer: ReturnType<typeof setTimeout> | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private attempt = 0;
  private running = false;
  private snapshot: ServerSnapshot = { status: 'idle', capacity: null };
  private readonly listeners = new Set<(snapshot: ServerSnapshot) => void>();

  constructor(private readonly url: string) {}

  get current(): ServerSnapshot {
    return this.snapshot;
  }

  subscribe(listener: (snapshot: ServerSnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.attempt = 0;
    this.connect();
  }

  /** Stops the standby connection, e.g. while a conversation holds its own socket. */
  stop(): void {
    this.running = false;
    this.clearTimers();
    this.connection?.close('Standby stopped');
    this.connection = null;
  }

  /** Reconnects at once instead of waiting out the current backoff. */
  retryNow(): void {
    if (!this.running) return this.start();
    if (this.connection) return;
    this.clearTimers();
    this.connect();
  }

  waitUntilReady(timeoutMs: number): Promise<boolean> {
    if (this.snapshot.status === 'ready') return Promise.resolve(true);
    this.retryNow();
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        unsubscribe();
        resolve(false);
      }, timeoutMs);
      const unsubscribe = this.subscribe(({ status }) => {
        if (status !== 'ready') return;
        clearTimeout(timer);
        unsubscribe();
        resolve(true);
      });
    });
  }

  private connect(): void {
    this.update({ status: this.retryStatus() });

    const connection = new WebSocketConnection(this.url, {
      onMessage: (message) => {
        if (message.type === 'hello') {
          this.clearHelloTimer();
          this.attempt = 0;
          this.update({ status: 'ready' });
        } else if (message.type === 'capacity') {
          this.updateCapacity(message);
        }
      },
      onClose: ({ code }) => {
        log.debug('standby closed', code);
        this.connection = null;
        this.scheduleRetry();
      },
    });
    this.connection = connection;

    this.helloTimer = setTimeout(() => {
      connection.close('Hello timeout');
      this.connection = null;
      this.scheduleRetry();
    }, HELLO_TIMEOUT_MS);
  }

  private scheduleRetry(): void {
    this.clearTimers();
    if (!this.running) return;
    const delay = exponentialBackoff(this.attempt, RETRY_BASE_MS, RETRY_MAX_MS);
    this.attempt += 1;
    this.update({ status: this.retryStatus() });
    this.retryTimer = setTimeout(() => this.connect(), delay);
  }

  private retryStatus(): ServerStatus {
    return this.attempt >= UNREACHABLE_AFTER_ATTEMPTS ? 'unreachable' : 'connecting';
  }

  private updateCapacity({ active, max }: CapacityMessage): void {
    this.update({ capacity: { active, max } });
  }

  private update(patch: Partial<ServerSnapshot>): void {
    const next = { ...this.snapshot, ...patch };
    if (next.status === this.snapshot.status && next.capacity === this.snapshot.capacity) return;
    this.snapshot = next;
    this.listeners.forEach((listener) => listener(next));
  }

  private clearHelloTimer(): void {
    if (this.helloTimer) clearTimeout(this.helloTimer);
    this.helloTimer = null;
  }

  private clearTimers(): void {
    this.clearHelloTimer();
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
  }
}
