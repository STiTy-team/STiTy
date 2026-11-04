import type { OutputMode } from '@/domain/output-modes';

export type TranscriptEntry = {
  id: string;
  language: string;
  text: string;
  translation: string;
  /** The sentence was already on screen as a dimmed partial, so it must not animate in again. */
  fromPartial: boolean;
  /** When it was committed, in epoch ms. */
  time: number;
};

export type LivePartial = {
  text: string;
  language: string;
};

/** A setting changed while the conversation was running. */
export type SettingChange =
  | { kind: 'outputMode'; mode: OutputMode }
  | { kind: 'muted'; muted: boolean }
  | { kind: 'myLanguage'; language: string }
  | { kind: 'languageAdded'; language: string }
  | { kind: 'languageRemoved'; language: string };

export type FeedEvent = {
  id: string;
  /** Number of committed entries when the change happened; the event is shown after them. */
  after: number;
  time: number;
  change: SettingChange;
};
