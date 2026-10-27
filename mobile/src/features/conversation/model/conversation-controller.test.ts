import type { AsrSessionHandlers, LiveOpenOptions, LiveSession } from '@/api';
import { MicrophonePermissionError } from '@/services/audio/microphone';

import { ConversationController } from './conversation-controller';
import { conversationActions, setPresenting, useConversationStore } from './conversation-store';
import type {
  AudioOutputPort,
  MicrophonePort,
  ServerMonitorPort,
  SessionPreferences,
  SpeechPort,
} from './ports';

const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

const deferred = <T = void>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

type FakeSession = LiveSession & {
  isOpen: boolean;
  close: jest.Mock<Promise<void>, []>;
  configure: jest.Mock;
  sendAudio: jest.Mock;
  logSpeech: jest.Mock;
};

const fakeSession = (conversationId: string | null): FakeSession => {
  const session: FakeSession = {
    conversationId,
    isOpen: true,
    close: jest.fn(async () => {
      session.isOpen = false;
    }),
    configure: jest.fn(),
    sendAudio: jest.fn(),
    logSpeech: jest.fn(),
  };
  return session;
};

type Overrides = {
  preferences?: Partial<SessionPreferences>;
  ready?: boolean;
  /** What the server answers in `ready`. */
  conversationId?: string | null;
};

const setup = (overrides: Overrides = {}) => {
  let onChunk!: (pcm: ArrayBuffer) => void;
  let onMicrophoneFailure!: (error: Error) => void;
  const sessions: FakeSession[] = [];
  const handlers: AsrSessionHandlers[] = [];

  const open = (options: LiveOpenOptions) => {
    handlers.push(options.handlers);
    const session = fakeSession(
      options.save ? (options.conversationId ?? overrides.conversationId ?? null) : null,
    );
    sessions.push(session);
    return session;
  };
  const openSession = jest.fn(async (options: LiveOpenOptions): Promise<LiveSession> =>
    open(options),
  );
  const openIntro = jest.fn(async (options: LiveOpenOptions): Promise<LiveSession> =>
    open(options),
  );

  const microphone = {
    start: jest.fn(async () => {}),
    stop: jest.fn(),
  } satisfies MicrophonePort;
  const speech = { enqueue: jest.fn(), stop: jest.fn() } satisfies SpeechPort;
  const monitor = {
    waitUntilReady: jest.fn(async () => overrides.ready ?? true),
    start: jest.fn(),
    stop: jest.fn(),
  } satisfies ServerMonitorPort;
  const output = { route: jest.fn(), release: jest.fn() } satisfies AudioOutputPort;
  const track = jest.fn();
  const feedback = { translationArrived: jest.fn() };
  const preferences: SessionPreferences = {
    myLanguage: 'en',
    peerLanguage: 'ko',
    outputMode: 'speaker',
    muted: false,
    holdToTalk: false,
    vibrate: false,
    saveRequested: true,
    ...overrides.preferences,
  };

  const controller = new ConversationController({
    monitor,
    openSession,
    openIntro,
    output,
    preferences: () => preferences,
    onCapacity: jest.fn(),
    track,
    feedback,
    createMicrophone: (options) => {
      onChunk = options.onChunk;
      onMicrophoneFailure = options.onFailure;
      return microphone;
    },
    createSpeechPlayer: () => speech,
  });

  return {
    controller,
    sessions,
    session: () => sessions[sessions.length - 1]!,
    microphone,
    speech,
    monitor,
    output,
    openSession,
    openIntro,
    track,
    feedback,
    preferences,
    handlers: () => handlers[handlers.length - 1]!,
    sendChunk: (pcm: ArrayBuffer) => onChunk(pcm),
    failMicrophone: () => onMicrophoneFailure(new Error('gone')),
  };
};

const phase = () => useConversationStore.getState().phase;

beforeEach(() => {
  conversationActions.reset();
  conversationActions.setPhase('idle');
  setPresenting(false);
});

describe('ConversationController', () => {
  it('starts a session: hands the socket over from the monitor, opens, captures, routes', async () => {
    const { controller, openSession, microphone, monitor, output } = setup();

    const starting = controller.start();
    expect(phase()).toBe('starting');
    await starting;

    expect(monitor.stop).toHaveBeenCalled();
    expect(openSession).toHaveBeenCalledWith(
      expect.objectContaining({
        languages: { lang: 'en', targetLang: 'ko' },
        save: true,
        conversationId: null,
      }),
    );
    expect(microphone.start).toHaveBeenCalled();
    expect(output.route).toHaveBeenCalledWith('speaker');
    expect(phase()).toBe('recording');
  });

  it('starts once when Start is pressed twice', async () => {
    const { controller, openSession } = setup();
    await Promise.all([controller.start(), controller.start()]);
    expect(openSession).toHaveBeenCalledTimes(1);
  });

  it('refuses identical languages without touching the network', async () => {
    const { controller, openSession } = setup({ preferences: { peerLanguage: 'en' } });

    await controller.start();

    expect(openSession).not.toHaveBeenCalled();
    expect(useConversationStore.getState()).toMatchObject({
      phase: 'failed',
      error: 'sameLanguage',
    });
  });

  it('reports an unreachable server and keeps monitoring', async () => {
    const { controller, openSession, monitor } = setup({ ready: false });

    await controller.start();

    expect(openSession).not.toHaveBeenCalled();
    expect(monitor.start).toHaveBeenCalled();
    expect(useConversationStore.getState()).toMatchObject({
      phase: 'failed',
      error: 'serverUnreachable',
    });
  });

  it('maps a denied microphone, releases the socket, and Retry starts again', async () => {
    const { controller, session, microphone, openSession, output } = setup();
    microphone.start.mockRejectedValueOnce(new MicrophonePermissionError('denied'));

    await controller.start();

    expect(session().close).toHaveBeenCalled();
    expect(output.release).toHaveBeenCalled();
    expect(useConversationStore.getState()).toMatchObject({
      phase: 'failed',
      error: 'microphoneDenied',
    });

    await controller.start();
    expect(openSession).toHaveBeenCalledTimes(2);
    expect(phase()).toBe('recording');
  });

  it('streams audio only while recording', async () => {
    const { controller, session, sendChunk } = setup();
    const chunk = new ArrayBuffer(3200);

    sendChunk(chunk);
    await controller.start();
    sendChunk(chunk);
    expect(session().sendAudio).toHaveBeenCalledTimes(1);

    controller.pause();
    sendChunk(chunk);
    expect(session().sendAudio).toHaveBeenCalledTimes(1);
  });

  it('turns a partial into a final in place and speaks the server translation', async () => {
    const { controller, speech, handlers } = setup();
    await controller.start();

    handlers().onPartial({ type: 'partial', text: '안녕', language: 'Korean' });
    expect(useConversationStore.getState().partial).toEqual({ text: '안녕', language: 'ko' });

    handlers().onFinal({
      type: 'final',
      original: '안녕하세요',
      translation: 'Hello',
      language: 'ko',
    });

    const { entries, partial } = useConversationStore.getState();
    expect(partial).toBeNull();
    expect(entries).toEqual([
      expect.objectContaining({
        language: 'ko',
        text: '안녕하세요',
        translation: 'Hello',
        fromPartial: true,
      }),
    ]);
    expect(speech.enqueue).toHaveBeenCalledWith({
      text: 'Hello',
      language: 'en',
      route: 'speaker',
    });
  });

  it('shows a final without a translation as said and does not speak it', async () => {
    const { controller, speech, handlers } = setup();
    await controller.start();

    handlers().onFinal({ type: 'final', original: 'Hello', translation: '', language: 'en' });

    expect(useConversationStore.getState().entries).toEqual([
      expect.objectContaining({ text: 'Hello', translation: '' }),
    ]);
    expect(speech.enqueue).not.toHaveBeenCalled();
  });

  it('stays silent when muted', async () => {
    const { controller, speech, handlers } = setup({ preferences: { muted: true } });
    await controller.start();

    handlers().onFinal({ type: 'final', original: 'Hi', translation: '안녕', language: 'en' });

    expect(speech.enqueue).not.toHaveBeenCalled();
  });

  it('sends a language change to the running session', async () => {
    const { controller, session, preferences } = setup();
    await controller.start();

    preferences.myLanguage = 'ja';
    controller.applyLanguages();

    expect(session().configure).toHaveBeenCalledWith({ lang: 'ja', targetLang: 'ko' });
  });

  it('ends: closes the socket, restores audio, clears the transcript, resumes monitoring', async () => {
    const { controller, session, output, monitor, handlers, track } = setup();
    await controller.start();
    handlers().onFinal({ type: 'final', original: 'Hi', translation: '안녕', language: 'en' });

    const ended = controller.end();

    expect(ended).toEqual({ conversationId: null, sentences: 1 });
    expect(session().close).toHaveBeenCalled();
    expect(output.release).toHaveBeenCalled();
    expect(monitor.start).toHaveBeenCalled();
    expect(useConversationStore.getState()).toMatchObject({ phase: 'idle', entries: [] });
    expect(track).toHaveBeenCalledWith(expect.objectContaining({ name: 'session_ended' }));
  });

  describe('cancelling a start or resume', () => {
    it('End while waiting for the server opens nothing', async () => {
      const { controller, monitor, openSession, microphone } = setup();
      const ready = deferred<boolean>();
      monitor.waitUntilReady.mockReturnValueOnce(ready.promise);

      const starting = controller.start();
      controller.end();
      ready.resolve(true);
      await starting;

      expect(openSession).not.toHaveBeenCalled();
      expect(microphone.start).not.toHaveBeenCalled();
      expect(phase()).toBe('idle');
    });

    it('End while the socket opens closes it when it arrives and never starts the mic', async () => {
      const { controller, openSession, microphone, sessions } = setup();
      const opened = deferred<LiveSession>();
      openSession.mockImplementationOnce(() => opened.promise);

      const starting = controller.start();
      await flush();
      controller.end();
      const late = fakeSession(null);
      opened.resolve(late);
      await starting;

      expect(late.close).toHaveBeenCalled();
      expect(microphone.start).not.toHaveBeenCalled();
      expect(sessions).toHaveLength(0);
      expect(useConversationStore.getState()).toMatchObject({ phase: 'idle', error: null });
    });

    it('End while the mic starts stops it and leaves no session', async () => {
      const { controller, microphone, session } = setup();
      const micStarted = deferred();
      microphone.start.mockReturnValueOnce(micStarted.promise);

      const starting = controller.start();
      await flush();
      controller.end();
      micStarted.resolve();
      await starting;

      expect(session().close).toHaveBeenCalled();
      expect(microphone.stop).toHaveBeenCalled();
      expect(useConversationStore.getState()).toMatchObject({ phase: 'idle', error: null });
    });

    it('End during Resume leaves nothing open', async () => {
      const { controller, microphone, sessions } = setup();
      await controller.start();
      controller.enterBackground();
      await flush();

      const micStarted = deferred();
      microphone.start.mockReturnValueOnce(micStarted.promise);
      const resuming = controller.resume();
      expect(phase()).toBe('resuming');
      await flush();
      controller.end();
      micStarted.resolve();
      await resuming;

      expect(sessions.every((session) => session.close.mock.calls.length > 0)).toBe(true);
      expect(phase()).toBe('idle');
    });

    it('the background during Start cancels it, and the foreground starts over', async () => {
      const { controller, microphone, session, monitor, openSession } = setup();
      const micStarted = deferred();
      microphone.start.mockReturnValueOnce(micStarted.promise);

      const starting = controller.start();
      await flush();
      controller.enterBackground();
      micStarted.resolve();
      await starting;

      expect(session().close).toHaveBeenCalled();
      expect(microphone.stop).toHaveBeenCalled();
      expect(monitor.stop).toHaveBeenCalled();
      expect(phase()).toBe('paused');

      monitor.start.mockClear();
      controller.enterForeground();
      await flush();
      expect(monitor.start).not.toHaveBeenCalled();
      expect(openSession).toHaveBeenCalledTimes(2);
      expect(phase()).toBe('recording');
    });

    it('a connection lost during Start fails it instead of showing Listening', async () => {
      const { controller, microphone, session, handlers } = setup();
      const micStarted = deferred();
      microphone.start.mockReturnValueOnce(micStarted.promise);

      const starting = controller.start();
      await flush();
      handlers().onLost();
      micStarted.resolve();
      await starting;

      expect(session().close).toHaveBeenCalled();
      expect(microphone.stop).toHaveBeenCalled();
      expect(useConversationStore.getState()).toMatchObject({
        phase: 'failed',
        error: 'connectionLost',
      });
    });

    it('the foreground while starting does not restart the monitor', async () => {
      const { controller, monitor } = setup();
      const ready = deferred<boolean>();
      monitor.waitUntilReady.mockReturnValueOnce(ready.promise);

      const starting = controller.start();
      controller.enterForeground();
      expect(monitor.start).not.toHaveBeenCalled();
      ready.resolve(true);
      await starting;
      expect(phase()).toBe('recording');
    });
  });

  describe('failure paths', () => {
    it('pauses with an error when the connection drops, closes it, and resume reopens', async () => {
      const { controller, session, microphone, handlers, openSession } = setup();
      await controller.start();
      const dropped = session();

      handlers().onLost();
      expect(useConversationStore.getState()).toMatchObject({
        phase: 'paused',
        error: 'connectionLost',
      });
      expect(microphone.stop).toHaveBeenCalled();
      expect(dropped.close).toHaveBeenCalled();

      await controller.resume();
      expect(openSession).toHaveBeenCalledTimes(2);
      expect(useConversationStore.getState()).toMatchObject({ phase: 'recording', error: null });
    });

    it('a failed resume stays paused and closes the socket it opened', async () => {
      const { controller, microphone, session, handlers } = setup();
      await controller.start();
      handlers().onLost();

      microphone.start.mockRejectedValueOnce(new MicrophonePermissionError('denied'));
      await controller.resume();

      expect(session().close).toHaveBeenCalled();
      expect(useConversationStore.getState()).toMatchObject({
        phase: 'paused',
        error: 'microphoneDenied',
      });
    });

    it('a microphone failure pauses with an error', async () => {
      const { controller, failMicrophone } = setup();
      await controller.start();
      failMicrophone();
      expect(useConversationStore.getState()).toMatchObject({
        phase: 'paused',
        error: 'microphoneUnavailable',
      });
    });

    it('ending in the background leaves the monitor off', async () => {
      const { controller, monitor } = setup();
      await controller.start();
      controller.enterBackground();
      monitor.start.mockClear();

      controller.end();

      expect(monitor.start).not.toHaveBeenCalled();
    });
  });

  describe('background', () => {
    it('releases everything in the background and resumes in the foreground', async () => {
      const { controller, session } = setup();
      await controller.start();

      controller.enterBackground();
      expect(session().close).toHaveBeenCalled();
      expect(phase()).toBe('paused');

      controller.enterForeground();
      await flush();
      expect(phase()).toBe('recording');
    });

    it('closes the socket of a paused conversation but does not resume it', async () => {
      const { controller, session } = setup();
      await controller.start();
      controller.pause();

      controller.enterBackground();
      controller.enterForeground();
      await flush();

      expect(session().close).toHaveBeenCalled();
      expect(phase()).toBe('paused');
    });

    it('keeps the last sentence the server flushes after a background stop', async () => {
      const { controller, handlers, speech } = setup();
      await controller.start();
      const closing = handlers();

      controller.enterBackground();
      closing.onPartial({ type: 'partial', text: 'late', language: 'en' });
      closing.onFinal({ type: 'final', original: 'late', translation: '늦음', language: 'en' });

      const { entries, partial } = useConversationStore.getState();
      expect(entries).toEqual([expect.objectContaining({ text: 'late', translation: '늦음' })]);
      expect(partial).toBeNull();
      expect(speech.enqueue).not.toHaveBeenCalled();
    });

    it('ignores results from a session that was ended', async () => {
      const { controller, handlers } = setup();
      await controller.start();
      const stale = handlers();

      controller.end();
      stale.onFinal({ type: 'final', original: 'late', translation: '늦음', language: 'en' });

      expect(useConversationStore.getState().entries).toEqual([]);
    });

    it('ignores a background session once it has finished closing', async () => {
      const { controller, handlers } = setup();
      await controller.start();
      const old = handlers();

      controller.enterBackground();
      await flush();
      controller.enterForeground();
      await flush();
      old.onFinal({ type: 'final', original: 'late', translation: '늦음', language: 'en' });

      expect(useConversationStore.getState().entries).toEqual([]);
    });
  });

  describe('records', () => {
    it('keeps the record id from ready and resends it when the background resumes', async () => {
      const { controller, openSession } = setup({ conversationId: 'rec-1' });
      await controller.start();
      expect(useConversationStore.getState().conversationId).toBe('rec-1');

      controller.enterBackground();
      controller.enterForeground();
      await flush();

      expect(openSession).toHaveBeenCalledTimes(2);
      expect(openSession.mock.calls[1]![0]).toMatchObject({ conversationId: 'rec-1', save: true });
      expect(controller.end().conversationId).toBe('rec-1');
    });

    it('asks not to save when saving is off, and gets no record id back', async () => {
      const { controller, openSession } = setup({
        conversationId: 'rec-1',
        preferences: { saveRequested: false },
      });

      await controller.start();

      expect(openSession.mock.calls[0]![0]).toMatchObject({ save: false });
      expect(useConversationStore.getState().conversationId).toBeNull();
    });
  });

  describe('add-ons', () => {
    it('sends no audio while hold-to-talk is released', async () => {
      const { controller, session, sendChunk } = setup({ preferences: { holdToTalk: true } });
      const chunk = new ArrayBuffer(3200);
      await controller.start();

      sendChunk(chunk);
      expect(session().sendAudio).not.toHaveBeenCalled();

      controller.setTalking(true);
      sendChunk(chunk);
      expect(session().sendAudio).toHaveBeenCalledTimes(1);

      controller.setTalking(false);
      sendChunk(chunk);
      expect(session().sendAudio).toHaveBeenCalledTimes(1);
    });

    it('ignores hold-to-talk while presenting', async () => {
      const { controller, session, sendChunk } = setup({ preferences: { holdToTalk: true } });
      await controller.start();
      setPresenting(true);
      sendChunk(new ArrayBuffer(3200));
      expect(session().sendAudio).toHaveBeenCalledTimes(1);
    });

    it('vibrates only for the other side’s translation, and only when switched on', async () => {
      const { controller, handlers, feedback } = setup({ preferences: { vibrate: true } });
      await controller.start();

      handlers().onFinal({ type: 'final', original: '안녕', translation: 'Hi', language: 'ko' });
      handlers().onFinal({ type: 'final', original: 'Hi', translation: '안녕', language: 'en' });

      expect(feedback.translationArrived).toHaveBeenCalledTimes(1);
    });

    it('reports the time to the first translation once', async () => {
      const { controller, handlers, track } = setup();
      await controller.start();
      handlers().onFinal({ type: 'final', original: 'a', translation: 'b', language: 'en' });
      handlers().onFinal({ type: 'final', original: 'c', translation: 'd', language: 'en' });
      expect(
        track.mock.calls.filter(([event]) => event.name === 'first_translation_shown'),
      ).toHaveLength(1);
    });
  });

  describe('demo', () => {
    it('opens the scripted session without the server or the microphone', async () => {
      const { controller, openIntro, openSession, monitor, microphone, track } = setup();

      await controller.start({ demo: true });

      expect(openIntro).toHaveBeenCalledWith(expect.objectContaining({ save: false }));
      expect(openSession).not.toHaveBeenCalled();
      expect(monitor.waitUntilReady).not.toHaveBeenCalled();
      expect(microphone.start).not.toHaveBeenCalled();
      expect(track).not.toHaveBeenCalled();
      expect(useConversationStore.getState()).toMatchObject({ phase: 'recording', demo: true });
    });

    it('pauses when the script completes, and resume plays it again', async () => {
      const { controller, openIntro, handlers, session } = setup();
      await controller.start({ demo: true });
      handlers().onFinal({ type: 'final', original: 'Hi', translation: '안녕', language: 'en' });

      handlers().onComplete?.();
      expect(phase()).toBe('paused');
      expect(session().close).toHaveBeenCalled();
      expect(useConversationStore.getState().error).toBeNull();

      await controller.resume();
      expect(openIntro).toHaveBeenCalledTimes(2);
      expect(phase()).toBe('recording');
    });

    it('stops the script on pause, and End leaves no record', async () => {
      const { controller, session } = setup();
      await controller.start({ demo: true });

      controller.pause();
      expect(session().close).toHaveBeenCalled();

      expect(controller.end()).toMatchObject({ conversationId: null });
      expect(useConversationStore.getState()).toMatchObject({ phase: 'idle', demo: false });
    });
  });
});
