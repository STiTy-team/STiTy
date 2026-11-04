import { useTranslation } from 'react-i18next';
import { ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '@/components/button';
import { Logo } from '@/components/logo';
import { EndedBanner } from '@/features/conversation/components/ended-banner';
import { LanguageList } from '@/features/conversation/components/language-list';
import { OutputModeControl } from '@/features/conversation/components/output-mode-control';
import { SaveToggle } from '@/features/conversation/components/save-toggle';
import { ServerNotice } from '@/features/conversation/components/server-notice';
import { ServerStatusButton } from '@/features/conversation/components/server-status-button';
import { openConversation } from '@/features/conversation/conversation-runtime';
import { selectServerFull, useServerStore } from '@/stores/server-store';
import { makeStyles, spacing } from '@/theme';

export const HomeScreen = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const full = useServerStore(selectServerFull);

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
      <View style={styles.header}>
        <Logo />
        <View style={styles.actions}>
          <ServerStatusButton />
        </View>
      </View>

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <EndedBanner />
        <ServerNotice />
        <LanguageList />
        <OutputModeControl />
        <SaveToggle />
      </ScrollView>

      <View style={styles.footer}>
        <PrimaryButton
          label={t('home.start')}
          icon="mic"
          disabled={full}
          onPress={() => openConversation()}
        />
      </View>
    </SafeAreaView>
  );
};

const useStyles = makeStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: 56,
    paddingLeft: spacing.xl,
    paddingRight: spacing.sm,
  },
  actions: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs },
  content: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.xxl,
    gap: spacing.xxl,
  },
  footer: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.md },
}));
