import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';
import { router, usePreventRemove } from 'expo-router';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { goBack } from '@/components/go-back';
import { IconButton } from '@/components/icon-button';
import { ScreenHeader } from '@/components/screen-header';
import { ErrorBanner } from '@/features/conversation/components/error-banner';
import { LargeFeed } from '@/features/conversation/components/large-feed';
import { SessionControls } from '@/features/conversation/components/session-controls';
import { SessionStatusLine } from '@/features/conversation/components/session-status';
import { TranscriptFeed } from '@/features/conversation/components/transcript-feed';
import { getConversationController } from '@/features/conversation/conversation-runtime';
import { useEndConversation } from '@/features/conversation/hooks/use-end-conversation';
import { languageName } from '@/features/conversation/language-name';
import { isInConversation } from '@/features/conversation/model/conversation-machine';
import { useConversationPreferences } from '@/features/conversation/model/conversation-preferences';
import { useConversationStore } from '@/features/conversation/model/conversation-store';
import { usePreferencesStore } from '@/stores/preferences-store';
import { makeStyles } from '@/theme';

const KEEP_AWAKE_TAG = 'conversation';

/** A demo has nothing to lose, so only a real conversation asks before closing. */
const hasConversation = () => {
  const { phase, entries, demo } = useConversationStore.getState();
  return !demo && (isInConversation(phase) || entries.length > 0);
};

/** The live screen: one layout, shaped by the add-ons switched on in conversation settings. */
export const ConversationScreen = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const endConversation = useEndConversation();
  const controller = getConversationController();

  const error = useConversationStore((state) => state.error);
  const demo = useConversationStore((state) => state.demo);
  const languages = usePreferencesStore((state) => state.languages);
  const largeText = useConversationPreferences((state) => state.largeText);
  const holdToTalk = useConversationPreferences((state) => state.holdToTalk);

  // Start normally opens this screen with the session already starting; a direct visit starts it.
  useEffect(() => {
    if (useConversationStore.getState().phase === 'idle') void controller.start();
    return () => {
      controller.end();
    };
  }, [controller]);

  useEffect(() => {
    void activateKeepAwakeAsync(KEEP_AWAKE_TAG);
    return () => void deactivateKeepAwake(KEEP_AWAKE_TAG);
  }, []);

  // Leaving by the back button or gesture would silently drop the conversation, so ask first.
  // A demo, or a conversation already ended in conversation settings, leaves without asking.
  const [leaving, setLeaving] = useState(false);
  usePreventRemove(!leaving, ({ repeat }) => {
    const leave = () => {
      setLeaving(true);
      repeat();
    };
    if (hasConversation()) endConversation(leave);
    else leave();
  });

  // The demo is the app's introduction, so it always plays in the plain layout.
  const large = !demo && largeText;
  const hold = !demo && holdToTalk;
  const title = languages.map((code) => languageName(t, code)).join(', ');

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
      <ScreenHeader
        title={title}
        subtitle={<SessionStatusLine />}
        left={<IconButton icon="chevron-back" label={t('common.back')} onPress={goBack} />}
        right={
          <View style={styles.actions}>
            <IconButton
              icon="tv-outline"
              label={t('present.start')}
              onPress={() => router.push('/present')}
            />
            <IconButton
              icon="options-outline"
              label={t('conversation.settings')}
              onPress={() => router.push('/conversation-settings')}
            />
          </View>
        }
      />
      {large ? <LargeFeed /> : <TranscriptFeed />}
      {error && <ErrorBanner code={error} />}
      <SessionControls holdToTalk={hold} />
    </SafeAreaView>
  );
};

const useStyles = makeStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  actions: { flexDirection: 'row' },
}));
