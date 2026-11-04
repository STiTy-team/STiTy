import { fireEvent, render, screen } from '@testing-library/react-native';

import { getConversationController } from '@/features/conversation/conversation-runtime';
import { conversationActions } from '@/features/conversation/model/conversation-store';
import type { ConversationPhase } from '@/features/conversation/model/conversation-machine';

import { SessionControls } from './session-controls';

jest.mock('@expo/vector-icons', () => ({ Ionicons: () => null }));
jest.mock('react-native-safe-area-context', () => ({
  useSafeAreaInsets: () => ({ top: 0, bottom: 0, left: 0, right: 0 }),
}));

const controller = getConversationController();

const renderPresent = async (phase: ConversationPhase) => {
  conversationActions.setPhase(phase);
  const onExit = jest.fn();
  await render(<SessionControls present={{ scheme: 'dark', onExit }} />);
  return { onExit };
};

afterEach(() => jest.restoreAllMocks());

describe('SessionControls in Present', () => {
  it('retries a failed start', async () => {
    const start = jest.spyOn(controller, 'start').mockResolvedValue();
    await renderPresent('failed');

    await fireEvent.press(screen.getByLabelText('conversation.retry'));

    expect(start).toHaveBeenCalled();
  });

  it('resumes when paused and shows connecting while starting', async () => {
    const resume = jest.spyOn(controller, 'resume').mockResolvedValue();
    await renderPresent('paused');
    await fireEvent.press(screen.getByLabelText('conversation.resume'));
    expect(resume).toHaveBeenCalled();

    await renderPresent('starting');
    expect(screen.getByLabelText('conversation.connecting')).toBeDisabled();
  });

  it('shows Exit in place of the output mode', async () => {
    const { onExit } = await renderPresent('recording');
    await fireEvent.press(screen.getByLabelText('present.exit'));
    expect(onExit).toHaveBeenCalled();
    expect(screen.queryByLabelText(/home\.outputTitle/)).toBeNull();
  });
});
