import { useState } from 'react';

import { joinConversation } from '@/features/conversation/conversation-runtime';
import type { ConversationErrorCode } from '@/features/conversation/model/conversation-error';

export const useJoinRoom = (options?: { replace?: boolean }) => {
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<ConversationErrorCode | null>(null);

  /** Resolves true once the room is joined; a refusal stays on this screen, in `error`. */
  const join = async (code: string) => {
    setChecking(true);
    setError(null);
    const failure = await joinConversation(code, options);
    setChecking(false);
    setError(failure);
    return failure === null;
  };

  return { checking, error, join, clearError: () => setError(null) };
};
