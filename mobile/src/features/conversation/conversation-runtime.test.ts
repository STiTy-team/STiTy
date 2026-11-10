import { router } from 'expo-router';

import { getApi } from '@/api';

import {
  getConversationController,
  joinConversation,
  openConversation,
} from './conversation-runtime';
import { conversationActions } from './model/conversation-store';

jest.mock('expo-router', () => ({ router: { push: jest.fn() } }));

describe('joinConversation', () => {
  beforeEach(() => {
    conversationActions.setPhase('idle');
    jest.mocked(router.push).mockClear();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('opens nothing when the server has no such room', async () => {
    jest.spyOn(getApi().live, 'isRoomOpen').mockResolvedValue(false);
    const start = jest.spyOn(getConversationController(), 'start');

    await expect(joinConversation('ABC234')).resolves.toBe('roomNotFound');

    expect(start).not.toHaveBeenCalled();
    expect(router.push).not.toHaveBeenCalled();
  });

  it('opens nothing when the server cannot be reached', async () => {
    jest.spyOn(getApi().live, 'isRoomOpen').mockRejectedValue(new Error('offline'));

    await expect(joinConversation('ABC234')).resolves.toBe('serverUnreachable');

    expect(router.push).not.toHaveBeenCalled();
  });

  it('joins an open room', async () => {
    jest.spyOn(getApi().live, 'isRoomOpen').mockResolvedValue(true);
    const start = jest
      .spyOn(getConversationController(), 'start')
      .mockImplementation(async () => conversationActions.setPhase('starting'));

    await expect(joinConversation('ABC234')).resolves.toBeNull();

    expect(start).toHaveBeenCalledWith({
      demo: false,
      room: { code: 'ABC234', joining: true, shared: true },
    });
    expect(router.push).toHaveBeenCalledWith('/conversation');
  });
});

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
