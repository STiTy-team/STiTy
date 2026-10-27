export type ConversationErrorCode =
  | 'sameLanguage'
  | 'serverUnreachable'
  | 'connectionLost'
  | 'microphoneDenied'
  | 'microphoneUnavailable';

export class ConversationError extends Error {
  constructor(readonly code: ConversationErrorCode) {
    super(code);
  }
}
