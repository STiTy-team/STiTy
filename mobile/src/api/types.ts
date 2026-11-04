import type { RecordSummary, Retention, TranscriptDocument } from '@/domain/records';

import type { AsrSessionHandlers, SessionLanguages, StartOptions } from '@/api/live/ws/asr-session';
import type { ServerMonitor } from '@/api/live/ws/server-monitor';

export type FeatureFlags = {
  accounts: boolean;
  records: boolean;
  summary: boolean;
  speakers: boolean;
};

export type Capabilities = {
  languages: string[];
  features: FeatureFlags;
};

export type CapabilitiesApi = { get(): Promise<Capabilities> };

export type User = { id: string; displayName: string; email?: string };

export type SignInMethod = 'apple' | 'google' | 'email';

export type AuthApi = {
  /** Methods this adapter can sign in with; the native Apple and Google SDKs come with live. */
  methods: SignInMethod[];
  /** The signed-in user from the last run, or null for a guest. */
  restore(): Promise<User | null>;
  signInWithApple(): Promise<User>;
  signInWithGoogle(): Promise<User>;
  requestEmailCode(email: string): Promise<void>;
  verifyEmailCode(email: string, code: string): Promise<User>;
  signOut(): Promise<void>;
  /** Sent in the WebSocket `start` so the server knows whose record it is writing. */
  accessToken(): Promise<string | null>;
};

export type Profile = User & {
  myLanguage: string;
  retention: Retention;
};

export type ProfileUpdate = Partial<Pick<Profile, 'displayName' | 'myLanguage' | 'retention'>>;

export type ProfileApi = {
  get(): Promise<Profile>;
  update(patch: ProfileUpdate): Promise<Profile>;
  /** Deletes the account and every saved record. */
  deleteAccount(): Promise<void>;
};

export type Page<T> = { items: T[]; nextCursor: string | null };

export type RecordsQuery = { cursor?: string; search?: string };

/** Records are written by the server while it streams results; the app only reads and manages them. */
export type ConversationsApi = {
  list(query: RecordsQuery): Promise<Page<RecordSummary>>;
  get(id: string): Promise<TranscriptDocument>;
  rename(id: string, title: string): Promise<void>;
  remove(id: string): Promise<void>;
};

export type LiveMonitor = Pick<
  ServerMonitor,
  'current' | 'subscribe' | 'start' | 'stop' | 'retryNow' | 'waitUntilReady'
>;

export type LiveOpenOptions = {
  languages: SessionLanguages;
  /** Ask the server to record this session; ignored for guests. */
  save: boolean;
  /** Set when resuming, so the server appends to the same record. */
  conversationId: string | null;
  handlers: AsrSessionHandlers;
};

export type LiveSession = {
  /** Set by the server's `ready` when the session is recorded; null for guests or "don't save". */
  conversationId: string | null;
  readonly isOpen: boolean;
  sendAudio(pcm: ArrayBuffer): void;
  configure(languages: SessionLanguages): void;
  logSpeech(entry: { text: string; lang: string; start: string; end: string }): void;
  /**
   * Sends `stop` and waits for the server to flush its last `final` and close, or for a short
   * grace period. Handlers stay live until then; `onLost` is not called.
   */
  close(): Promise<void>;
};

export type LiveApi = {
  monitor(): LiveMonitor;
  open(options: LiveOpenOptions): Promise<LiveSession>;
};

export type Api = {
  capabilities: CapabilitiesApi;
  auth: AuthApi;
  profile: ProfileApi;
  conversations: ConversationsApi;
  live: LiveApi;
};

export type { StartOptions };
