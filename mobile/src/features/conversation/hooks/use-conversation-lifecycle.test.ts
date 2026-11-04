import { followAppState } from './use-conversation-lifecycle';

const setup = () => {
  const controller = { enterBackground: jest.fn(), enterForeground: jest.fn() };
  return { controller, follow: followAppState(controller) };
};

describe('followAppState', () => {
  it('ignores inactive, such as Control Center on iOS', () => {
    const { controller, follow } = setup();
    follow('inactive');
    follow('active');
    expect(controller.enterBackground).not.toHaveBeenCalled();
    expect(controller.enterForeground).not.toHaveBeenCalled();
  });

  it('pauses in the background and resumes once active again, through inactive', () => {
    const { controller, follow } = setup();
    follow('inactive');
    follow('background');
    follow('inactive');
    follow('active');
    expect(controller.enterBackground).toHaveBeenCalledTimes(1);
    expect(controller.enterForeground).toHaveBeenCalledTimes(1);
  });
});
