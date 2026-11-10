import { router, useLocalSearchParams } from 'expo-router';
import { useTranslation } from 'react-i18next';
import { ScrollView, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '@/components/button';
import { ScreenHeader } from '@/components/screen-header';
import { LoadingView, StateView } from '@/components/state-view';
import { isRoomCode, normalizeRoomCode } from '@/domain/rooms';
import { LanguageList } from '@/features/conversation/components/language-list';
import { useJoinRoom } from '@/features/conversation/hooks/use-join-room';
import { useServerLink } from '@/features/settings/hooks/use-server-link';
import { makeStyles, spacing, typography } from '@/theme';

/**
 * Opened by an invite link: /join?room=…, or stity://join?room=… in the app. A link can come from
 * anyone and the mic starts on joining, so it joins only on a tap, after the languages are shown.
 */
export const JoinLinkScreen = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const { room, server } = useLocalSearchParams<{ room?: string; server?: string }>();
  const { pending, target } = useServerLink(server);
  const { checking, error, join } = useJoinRoom({ replace: true });
  const code = normalizeRoomCode(room ?? '');
  const goHome = () => router.replace('/');

  if (pending) return <LoadingView label={target ?? ''} />;
  if (!isRoomCode(code)) {
    return (
      <StateView
        icon="alert-circle-outline"
        title={t('errors.roomNotFound')}
        action={{ label: t('common.ok'), onPress: goHome }}
      />
    );
  }

  return (
    <SafeAreaView style={styles.screen}>
      <ScreenHeader title={t('conversation.room', { code })} />
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <Text style={styles.hint}>{t('home.joinLinkHint')}</Text>
        <LanguageList />
      </ScrollView>
      <View style={styles.footer}>
        {error && (
          <Text style={styles.error} accessibilityLiveRegion="polite">
            {t(`errors.${error}`)}
          </Text>
        )}
        <PrimaryButton label={t('home.join')} loading={checking} onPress={() => void join(code)} />
        <PrimaryButton label={t('common.cancel')} variant="secondary" onPress={goHome} />
      </View>
    </SafeAreaView>
  );
};

const useStyles = makeStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  content: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm, gap: spacing.lg },
  hint: { ...typography.body, color: colors.textSecondary },
  footer: { gap: spacing.sm, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  error: { ...typography.small, color: colors.danger },
}));
