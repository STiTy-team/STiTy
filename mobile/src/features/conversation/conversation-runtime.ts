import { router } from 'expo-router';

import { getApi, getIntroLive } from '@/api';
import { features } from '@/config/features';
import { joinRoom, type Room } from '@/domain/rooms';
import { audioOutput } from '@/services/audio/audio-output';
import { Microphone } from '@/services/audio/microphone';
import { haptics } from '@/services/haptics';
import { SpeechPlayer } from '@/services/speech/speech-player';
import { track } from '@/services/telemetry';
import { usePreferencesStore } from '@/stores/preferences-store';
import { useServerStore } from '@/stores/server-store';

import { ConversationController } from './model/conversation-controller';
import type { ConversationErrorCode } from './model/conversation-error';
import { canStart } from './model/conversation-machine';
import { sessionPreferences, useConversationPreferences } from './model/conversation-preferences';
import { useConversationStore } from './model/conversation-store';
import type { SpeechPort } from './model/ports';

export { useConversationPreferences };

/** Stands in for the speech player where `features.speech` is off: translations stay on screen. */
const noSpeech: SpeechPort = { enqueue: () => {}, stop: () => {} };

let controller: ConversationController | null = null;

export const getConversationController = (): ConversationController =>
  (controller ??= new ConversationController({
    monitor: getApi().live.monitor(),
    openSession: (options) => getApi().live.open(options),
    openIntro: (options) => getIntroLive().open(options),
    output: audioOutput,
    preferences: () =>
      sessionPreferences(usePreferencesStore.getState(), useConversationPreferences.getState()),
    onCapacity: (capacity) => useServerStore.getState().setCapacity(capacity),
    createMicrophone: (options) => new Microphone(options),
    createSpeechPlayer: (onSpoken) => (features.speech ? new SpeechPlayer(onSpoken) : noSpeech),
    track,
    feedback: haptics,
  }));

export const getServerMonitor = () => getApi().live.monitor();

type OpenOptions = { demo?: boolean; room?: Room; replace?: boolean };

/**
 * Start, from home or the intro: the session starts here, before the screen opens, so a second
 * tap finds it already starting and opens nothing. `replace` stands in for a link's own screen.
 */
export const openConversation = ({
  demo = false,
  room,
  replace = false,
}: OpenOptions = {}): void => {
  if (!canStart(useConversationStore.getState().phase)) return;
  void getConversationController().start({ demo, room });
  if (replace) router.replace('/conversation');
  else router.push('/conversation');
};

/** Join: nothing opens unless the server has that room. Resolves with why it didn't. */
export const joinConversation = async (
  code: string,
  { replace = false }: { replace?: boolean } = {},
): Promise<ConversationErrorCode | null> => {
  try {
    if (!(await getApi().live.isRoomOpen(code))) return 'roomNotFound';
  } catch {
    return 'serverUnreachable';
  }
  openConversation({ room: joinRoom(code), replace });
  return null;
};
