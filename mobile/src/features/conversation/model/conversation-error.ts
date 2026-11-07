export type ConversationErrorCode =
  | 'sameLanguage'
  | 'serverUnreachable'
  | 'connectionLost'
  | 'roomNotFound'
  | 'microphoneDenied'
  | 'microphoneUnavailable';

export class ConversationError extends Error {
  constructor(readonly code: ConversationErrorCode) {
    super(code);
  }
}
