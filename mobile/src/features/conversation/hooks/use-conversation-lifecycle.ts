import { useEffect } from 'react';
import { AppState, type AppStateStatus } from 'react-native';

import {
  getConversationController,
  getServerMonitor,
} from '@/features/conversation/conversation-runtime';
import type { ConversationController } from '@/features/conversation/model/conversation-controller';
import { isInConversation } from '@/features/conversation/model/conversation-machine';
import { useConversationPreferences } from '@/features/conversation/model/conversation-preferences';
import {
  conversationActions,
  useConversationStore,
} from '@/features/conversation/model/conversation-store';
import type { SettingChange } from '@/features/conversation/model/transcript';
import { audioOutput } from '@/services/audio/audio-output';
import {
  peerLanguageOf,
  usePreferencesStore,
  type LanguageSettings,
} from '@/stores/preferences-store';
import { useServerStore } from '@/stores/server-store';

/** Changes are noted in the transcript only while a conversation is open. */
const noteChange = (...changes: SettingChange[]) => {
  if (!isInConversation(useConversationStore.getState().phase)) return;
  changes.forEach(conversationActions.recordChange);
};

const languageChanges = (next: LanguageSettings, previous: LanguageSettings): SettingChange[] => [
  ...(next.myLanguage !== previous.myLanguage
    ? [{ kind: 'myLanguage', language: next.myLanguage } as const]
    : []),
  ...next.languages
    .filter((code) => !previous.languages.includes(code))
    .map((language): SettingChange => ({ kind: 'languageAdded', language })),
  ...previous.languages
    .filter((code) => !next.languages.includes(code))
    .map((language): SettingChange => ({ kind: 'languageRemoved', language })),
];

/**
 * Only `background` releases the session. iOS reports `inactive` for Control Center, a call
 * banner or the app switcher, while the app keeps running and the microphone keeps working.
 */
export const followAppState = (
  controller: Pick<ConversationController, 'enterBackground' | 'enterForeground'>,
) => {
  let inBackground = false;
  return (next: AppStateStatus) => {
    if (next === 'background' && !inBackground) {
      inBackground = true;
      controller.enterBackground();
    } else if (next === 'active' && inBackground) {
      inBackground = false;
      controller.enterForeground();
    }
  };
};

/** Connects the conversation runtime to the app: server status, app state and preferences. */
export const useConversationLifecycle = () => {
  useEffect(() => {
    const controller = getConversationController();
    const monitor = getServerMonitor();

    const stopMirroring = monitor.subscribe(useServerStore.getState().setSnapshot);
    monitor.start();

    useConversationPreferences
      .getState()
      .setOutputMode(audioOutput.isHeadsetConnected() ? 'earphone' : 'speaker');

    const stopWatchingConversation = useConversationPreferences.subscribe((next, previous) => {
      if (next.outputMode !== previous.outputMode) {
        controller.applyOutputMode(next.outputMode);
        noteChange({ kind: 'outputMode', mode: next.outputMode });
      }
      if (next.muted !== previous.muted) {
        controller.applyMuted(next.muted);
        noteChange({ kind: 'muted', muted: next.muted });
      }
    });

    const stopWatchingLanguages = usePreferencesStore.subscribe((next, previous) => {
      if (
        next.myLanguage !== previous.myLanguage ||
        peerLanguageOf(next) !== peerLanguageOf(previous)
      ) {
        controller.applyLanguages();
      }
      noteChange(...languageChanges(next, previous));
    });

    const appState = AppState.addEventListener('change', followAppState(controller));

    return () => {
      appState.remove();
      stopWatchingConversation();
      stopWatchingLanguages();
      stopMirroring();
      controller.end();
      monitor.stop();
    };
  }, []);
};
