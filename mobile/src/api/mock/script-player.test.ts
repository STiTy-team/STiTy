import { playScript } from './script-player';
import type { ScriptLine } from './scripts';

const SCRIPT: ScriptLine[] = [
  ['en', 'Hello there', '안녕하세요'],
  ['ko', '네', 'Yes'],
];

const handlers = () => ({ onPartial: jest.fn(), onFinal: jest.fn(), onComplete: jest.fn() });

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

describe('playScript', () => {
  it('types word by word, commits each line, then completes', () => {
    const on = handlers();
    playScript(SCRIPT, on, { firstLineMs: 10, typing: 'words', typingMs: 10, lineGapMs: 10 });
    jest.runAllTimers();

    expect(on.onPartial.mock.calls.map(([message]) => message.text)).toEqual([
      'Hello',
      'Hello there',
      '네',
    ]);
    expect(on.onFinal.mock.calls.map(([message]) => message.original)).toEqual([
      'Hello there',
      '네',
    ]);
    expect(on.onComplete).toHaveBeenCalledTimes(1);
    expect(on.onComplete.mock.invocationCallOrder[0]).toBeGreaterThan(
      on.onFinal.mock.invocationCallOrder[1]!,
    );
  });

  it('shows a whole line at once, and stops when told', () => {
    const on = handlers();
    const stop = playScript(SCRIPT, on, {
      firstLineMs: 10,
      typing: 'line',
      typingMs: 100,
      lineGapMs: 100,
    });
    jest.advanceTimersByTime(50);
    expect(on.onPartial).toHaveBeenCalledWith(expect.objectContaining({ text: 'Hello there' }));

    stop();
    jest.runAllTimers();
    expect(on.onFinal).not.toHaveBeenCalled();
    expect(on.onComplete).not.toHaveBeenCalled();
  });
});
