import { conversationActions, useConversationStore } from './conversation-store';

beforeEach(() => {
  jest.useFakeTimers({ now: 0 });
  conversationActions.reset();
});

afterEach(() => jest.useRealTimers());

describe('recording clock', () => {
  it('counts only time spent recording', () => {
    conversationActions.setPhase('recording');
    jest.advanceTimersByTime(5_000);
    conversationActions.setPhase('paused');
    jest.advanceTimersByTime(60_000);
    conversationActions.setPhase('recording');
    jest.advanceTimersByTime(2_000);
    conversationActions.setPhase('paused');

    expect(useConversationStore.getState().recordedMs).toBe(7_000);
    expect(useConversationStore.getState().recordingSince).toBeNull();
  });
});

describe('setting changes', () => {
  it('remembers how many sentences came before each change', () => {
    conversationActions.commit('en', 'Hello', '안녕하세요');
    conversationActions.recordChange({ kind: 'outputMode', mode: 'earphone' });

    const [event] = useConversationStore.getState().events;
    expect(event).toMatchObject({ after: 1, change: { kind: 'outputMode', mode: 'earphone' } });
  });

  it('forgets changes when the conversation ends', () => {
    conversationActions.recordChange({ kind: 'muted', muted: true });
    conversationActions.reset();
    expect(useConversationStore.getState().events).toEqual([]);
  });
});
