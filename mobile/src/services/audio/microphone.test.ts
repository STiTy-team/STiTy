import StityAudio from '@modules/stity-audio';

import { Microphone, MicrophoneUnavailableError } from './microphone';

const native = jest.mocked(StityAudio);

const setup = () => {
  const onFailure = jest.fn();
  const microphone = new Microphone({
    sampleRate: 16_000,
    chunkSamples: 1_600,
    onChunk: jest.fn(),
    onFailure,
  });
  return { microphone, onFailure };
};

/** The permission prompt stays open until the returned function answers it. */
const holdPermission = () => {
  let answer!: () => void;
  native.getMicrophonePermissionAsync.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        answer = () => resolve({ granted: true, canAskAgain: true } as never);
      }),
  );
  return () => answer();
};

beforeEach(() => {
  jest.useFakeTimers();
  jest.clearAllMocks();
});
afterEach(() => jest.useRealTimers());

describe('Microphone', () => {
  it('reports failure once when the watchdog restart is rejected', async () => {
    const { microphone, onFailure } = setup();
    await microphone.start();
    native.startCaptureAsync.mockRejectedValueOnce(new Error('busy'));

    await jest.advanceTimersByTimeAsync(10_000);

    expect(onFailure).toHaveBeenCalledTimes(1);
    expect(onFailure.mock.calls[0]![0]).toBeInstanceOf(MicrophoneUnavailableError);
  });

  it('throws from start without reporting a failure when the first capture is rejected', async () => {
    const { microphone, onFailure } = setup();
    native.startCaptureAsync.mockRejectedValueOnce(new Error('busy'));

    await expect(microphone.start()).rejects.toBeInstanceOf(MicrophoneUnavailableError);
    await jest.advanceTimersByTimeAsync(10_000);

    expect(onFailure).not.toHaveBeenCalled();
    expect(native.startCaptureAsync).toHaveBeenCalledTimes(1);
  });
});

describe('stopping while starting', () => {
  it('never starts capture when stopped during the permission prompt', async () => {
    const { microphone } = setup();
    const grant = holdPermission();

    const starting = microphone.start();
    microphone.stop();
    grant();
    await starting;

    expect(native.startCaptureAsync).not.toHaveBeenCalled();
    expect(native.addListener).not.toHaveBeenCalled();
  });

  it('lets a new start win over an older one still waiting for permission', async () => {
    const { microphone } = setup();
    const grant = holdPermission();

    const first = microphone.start();
    microphone.stop();
    await microphone.start();
    grant();
    await first;

    expect(native.startCaptureAsync).toHaveBeenCalledTimes(1);
  });
});
