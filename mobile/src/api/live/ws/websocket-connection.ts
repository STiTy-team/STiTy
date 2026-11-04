import {
  encodeClientMessage,
  parseServerMessage,
  type ClientMessage,
  type ServerMessage,
} from './protocol';

const HEARTBEAT_INTERVAL_MS = 20_000;

type Handlers = {
  onMessage: (message: ServerMessage) => void;
  onClose: (event: { code: number; reason: string }) => void;
};

/**
 * A WebSocket with typed messages and an application-level ping, which keeps proxies such as
 * ngrok and the RunPod gateway from dropping an idle connection.
 */
export class WebSocketConnection {
  private readonly socket: WebSocket;
  private heartbeat: ReturnType<typeof setInterval> | null = null;
  private closed = false;
  private onSocketClosed: (() => void) | null = null;

  constructor(url: string, handlers: Handlers) {
    this.socket = new WebSocket(url);
    this.socket.binaryType = 'arraybuffer';

    this.socket.onopen = () => {
      this.heartbeat = setInterval(() => this.send({ type: 'ping' }), HEARTBEAT_INTERVAL_MS);
    };
    this.socket.onmessage = (event) => {
      if (this.closed) return;
      const message = parseServerMessage(event.data);
      if (message) handlers.onMessage(message);
    };
    this.socket.onerror = () => {
      this.socket.close();
    };
    this.socket.onclose = (event) => {
      this.stopHeartbeat();
      this.onSocketClosed?.();
      if (this.closed) return;
      this.closed = true;
      handlers.onClose({ code: event.code, reason: event.reason });
    };
  }

  get isOpen(): boolean {
    return this.socket.readyState === WebSocket.OPEN;
  }

  send(message: ClientMessage): void {
    if (this.isOpen) this.socket.send(encodeClientMessage(message));
  }

  sendBinary(data: ArrayBuffer): void {
    if (this.isOpen) this.socket.send(data);
  }

  /**
   * Sends a last message and keeps delivering messages until the server closes the socket, or
   * closes it after `graceMs`. onClose still fires when the server closes it first.
   */
  finish(last: ClientMessage, reason: string, graceMs: number): Promise<void> {
    if (!this.isOpen) {
      this.close(reason);
      return Promise.resolve();
    }
    this.send(last);
    this.stopHeartbeat();
    return new Promise((resolve) => {
      const grace = setTimeout(() => {
        this.close(reason);
        resolve();
      }, graceMs);
      this.onSocketClosed = () => {
        clearTimeout(grace);
        resolve();
      };
    });
  }

  /** Closes without notifying onClose — the caller already knows. */
  close(reason: string): void {
    this.closed = true;
    this.stopHeartbeat();
    if (this.socket.readyState === WebSocket.CONNECTING || this.isOpen) {
      this.socket.close(1000, reason);
    }
  }

  private stopHeartbeat(): void {
    if (this.heartbeat) clearInterval(this.heartbeat);
    this.heartbeat = null;
  }
}
