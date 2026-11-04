import { create } from 'zustand';

import type { EndedConversation } from './conversation-controller';

/** The conversation that just ended, for home's "Saved · Open" or sign-in prompt. */
export const useEndedStore = create<{ last: EndedConversation | null }>(() => ({ last: null }));

export const endedActions = {
  remember: (ended: EndedConversation) =>
    useEndedStore.setState({ last: ended.sentences > 0 ? ended : null }),
  dismiss: () => useEndedStore.setState({ last: null }),
};
