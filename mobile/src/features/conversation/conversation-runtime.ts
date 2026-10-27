import { router } from 'expo-router';

import { getApi, getIntroLive } from '@/api';
import { audioOutput } from '@/services/audio/audio-output';
import { Microphone } from '@/services/audio/microphone';
import { haptics } from '@/services/haptics';
import { SpeechPlayer } from '@/services/speech/speech-player';
import { track } from '@/services/telemetry';
import { usePreferencesStore } from '@/stores/preferences-store';
import { useServerStore } from '@/stores/server-store';

import { ConversationController } from './model/conversation-controller';
import { canStart } from './model/conversation-machine';
import { sessionPreferences, useConversationPreferences } from './model/conversation-preferences';
import { useConversationStore } from './model/conversation-store';

export { useConversationPreferences };

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
    createSpeechPlayer: (onSpoken) => new SpeechPlayer(onSpoken),
    track,
    feedback: haptics,
  }));

export const getServerMonitor = () => getApi().live.monitor();

/**
 * Start, from home or the intro: the session starts here, before the screen opens, so a second
 * tap finds it already starting and opens nothing.
 */
export const openConversation = ({ demo = false }: { demo?: boolean } = {}): void => {
  if (!canStart(useConversationStore.getState().phase)) return;
  void getConversationController().start({ demo });
  router.push('/conversation');
};
