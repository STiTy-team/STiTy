import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { showDialog } from '@/components/dialog';
import { getConversationController } from '@/features/conversation/conversation-runtime';
import { endedActions } from '@/features/conversation/model/ended-store';
import { useConversationStore } from '@/features/conversation/model/conversation-store';

/** Asks before ending, says whether the transcript is kept, then ends it and runs `then`. */
export const useEndConversation = () => {
  const { t } = useTranslation();
  return useCallback(
    async (then: () => void) => {
      const saved = useConversationStore.getState().conversationId !== null;
      const confirmed = await showDialog({
        title: t('conversation.endConfirmTitle'),
        message: t(saved ? 'conversation.endConfirmSaved' : 'conversation.endConfirmMessage'),
        confirmLabel: t('conversation.endConfirmAction'),
        cancelLabel: t('common.cancel'),
        destructive: !saved,
      });
      if (!confirmed) return;
      endedActions.remember(getConversationController().end());
      then();
    },
    [t],
  );
};
