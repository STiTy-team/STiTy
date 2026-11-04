import { usePreferencesStore } from '@/stores/preferences-store';

import { sessionPreferences, useConversationPreferences } from './conversation-preferences';
import { conversationActions } from './conversation-store';

const store = useConversationPreferences;

beforeEach(() => store.getState().reset());

describe('conversation preferences', () => {
  it('switches each add-on on its own, all off by default', () => {
    expect(store.getState()).toMatchObject({ holdToTalk: false, largeText: false, vibrate: false });
    store.getState().setAddOn('holdToTalk', true);
    expect(store.getState()).toMatchObject({ holdToTalk: true, largeText: false, vibrate: false });
  });

  it('keeps Save as the user left it after a conversation ends and across launches', () => {
    const { partialize, merge } = store.persist.getOptions();
    store.getState().setSaveRequested(false);
    conversationActions.reset();
    expect(store.getState().saveRequested).toBe(false);

    const restored = merge?.(partialize?.(store.getState()), {
      ...store.getState(),
      saveRequested: true,
    });
    expect(restored?.saveRequested).toBe(false);
  });

  it('does not save the output mode or mute', () => {
    const saved = store.persist.getOptions().partialize!(store.getState());
    expect(saved).not.toHaveProperty('outputMode');
    expect(saved).not.toHaveProperty('muted');
  });

  it('drops an invalid saved field to its default', () => {
    const restored = store.persist
      .getOptions()
      .merge?.({ vibrate: 'yes', holdToTalk: true }, store.getState());
    expect(restored).toMatchObject({ vibrate: false, holdToTalk: true });
  });

  it('combines languages and conversation settings for the session', () => {
    usePreferencesStore.setState({ languages: ['ja', 'ko', 'en'], myLanguage: 'ko' });
    store.getState().setSaveRequested(false);
    expect(sessionPreferences(usePreferencesStore.getState(), store.getState())).toMatchObject({
      myLanguage: 'ko',
      peerLanguage: 'ja',
      saveRequested: false,
    });
  });
});
