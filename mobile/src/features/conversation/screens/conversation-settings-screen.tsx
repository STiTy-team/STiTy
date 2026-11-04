import { router } from 'expo-router';
import { useTranslation } from 'react-i18next';
import { ScrollView, Switch, Text, View } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { useShallow } from 'zustand/react/shallow';

import { goBack, goHome } from '@/components/go-back';
import { IconButton } from '@/components/icon-button';
import { ListRow, ListSection } from '@/components/list';
import { ScreenHeader } from '@/components/screen-header';
import { LanguageList } from '@/features/conversation/components/language-list';
import { OutputModeControl } from '@/features/conversation/components/output-mode-control';
import { SessionStatusLine } from '@/features/conversation/components/session-status';
import { useEndConversation } from '@/features/conversation/hooks/use-end-conversation';
import { isInConversation } from '@/features/conversation/model/conversation-machine';
import {
  ADD_ONS,
  useConversationPreferences,
  type AddOns,
} from '@/features/conversation/model/conversation-preferences';
import { useConversationStore } from '@/features/conversation/model/conversation-store';
import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

/** Settings of the running conversation. Every change applies at once; nothing pauses. */
export const ConversationSettingsScreen = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const colors = useColors();
  const { bottom } = useSafeAreaInsets();
  const endConversation = useEndConversation();
  const phase = useConversationStore((state) => state.phase);
  const muted = useConversationPreferences((state) => state.muted);
  const addOns = useConversationPreferences(
    useShallow((state): AddOns => ({
      holdToTalk: state.holdToTalk,
      largeText: state.largeText,
      vibrate: state.vibrate,
    })),
  );
  const toggle = (value: boolean, onChange: (on: boolean) => void, label: string) => (
    <Switch
      value={value}
      onValueChange={onChange}
      trackColor={{ true: colors.accent, false: colors.surfaceStrong }}
      thumbColor="#FFFFFF"
      accessibilityLabel={label}
    />
  );

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
      <ScreenHeader
        title={t('conversation.settings')}
        left={<IconButton icon="chevron-back" label={t('common.back')} onPress={goBack} />}
      />
      {isInConversation(phase) && (
        <View style={styles.liveBar}>
          <Text style={styles.liveText}>
            {t(
              phase === 'recording'
                ? 'conversationSettings.running'
                : 'conversationSettings.paused',
            )}
          </Text>
          <SessionStatusLine prefix="" />
        </View>
      )}
      <ScrollView
        contentContainerStyle={[styles.content, { paddingBottom: bottom + spacing.xxl }]}
        showsVerticalScrollIndicator={false}
      >
        <LanguageList />
        <OutputModeControl />
        <ListSection footer={t('conversationSettings.speakHint')}>
          <ListRow title={t('conversationSettings.speak')}>
            {toggle(
              !muted,
              (on) => useConversationPreferences.getState().setMuted(!on),
              t('conversationSettings.speak'),
            )}
          </ListRow>
        </ListSection>
        <ListSection title={t('addOns.title')}>
          {ADD_ONS.map((addOn) => (
            <ListRow key={addOn} title={t(`addOns.${addOn}`)} subtitle={t(`addOns.${addOn}Hint`)}>
              {toggle(
                addOns[addOn],
                (on) => useConversationPreferences.getState().setAddOn(addOn, on),
                t(`addOns.${addOn}`),
              )}
            </ListRow>
          ))}
        </ListSection>
        <ListSection footer={t('present.hint')}>
          <ListRow title={t('present.start')} chevron onPress={() => router.replace('/present')} />
        </ListSection>
        <ListSection>
          <ListRow
            title={t('conversation.end')}
            tone="destructive"
            onPress={() => endConversation(goHome)}
          />
        </ListSection>
      </ScrollView>
    </SafeAreaView>
  );
};

const useStyles = makeStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  liveBar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginHorizontal: spacing.lg,
    marginBottom: spacing.md,
    paddingVertical: 10,
    paddingHorizontal: spacing.md,
    borderRadius: radius.control,
    backgroundColor: colors.surface,
  },
  liveText: { ...typography.caption, flex: 1, color: colors.textSecondary },
  content: { paddingHorizontal: spacing.lg, paddingTop: spacing.xs, gap: spacing.xxl },
}));
