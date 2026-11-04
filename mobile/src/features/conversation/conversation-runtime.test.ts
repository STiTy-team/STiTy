import { router } from 'expo-router';

import { getConversationController, openConversation } from './conversation-runtime';
import { conversationActions } from './model/conversation-store';

jest.mock('expo-router', () => ({ router: { push: jest.fn() } }));

describe('openConversation', () => {
  beforeEach(() => {
    conversationActions.setPhase('idle');
    jest.mocked(router.push).mockClear();
  });

  it('opens one screen and starts once when Start is tapped twice', () => {
    const start = jest
      .spyOn(getConversationController(), 'start')
      .mockImplementation(async () => conversationActions.setPhase('starting'));

    openConversation();
    openConversation();

    expect(start).toHaveBeenCalledTimes(1);
    expect(router.push).toHaveBeenCalledTimes(1);
    expect(router.push).toHaveBeenCalledWith('/conversation');
  });

  it('opens nothing while a conversation is running', () => {
    conversationActions.setPhase('recording');
    openConversation({ demo: true });
    expect(router.push).not.toHaveBeenCalled();
  });
});
