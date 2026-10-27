import { router } from 'expo-router';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ScrollView, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '@/components/button';
import { showDialog, showPrompt } from '@/components/dialog';
import { ListRow, ListSection, ListValue } from '@/components/list';
import { OptionSheet } from '@/components/option-sheet';
import { SegmentedControl } from '@/components/segmented-control';
import { TabHeader } from '@/components/tab-header';
import { appInfo } from '@/config/app-info';
import { SUPPORTED_LANGUAGES } from '@/domain/languages';
import { RETENTION_OPTIONS, type Retention } from '@/domain/records';
import { deleteAccount, signOut } from '@/features/account/account-runtime';
import { useLoadProfileOnMount, useProfileStore } from '@/features/account/model/profile-store';
import {
  getConversationController,
  openConversation,
} from '@/features/conversation/conversation-runtime';
import { languageName } from '@/features/conversation/language-name';
import { UI_LANGUAGE_NAMES } from '@/i18n/device-language';
import { useFeatures } from '@/stores/capabilities-store';
import { THEME_PREFERENCES, usePreferencesStore } from '@/stores/preferences-store';
import { selectSignedIn, useSessionStore } from '@/stores/session-store';
import { makeStyles, radius, spacing, typography } from '@/theme';

type Sheet = 'language' | 'retention' | null;

/** Who I am and how the app behaves: the profile on top, then the app settings. */
export const ProfileScreen = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const signedIn = useSessionStore(selectSignedIn);
  const { accounts } = useFeatures();

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
      <TabHeader title={t('tabs.profile')} />
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        {signedIn ? <SignedInProfile /> : accounts ? <GuestCard /> : null}
        <AppSettings />
        {signedIn && <AccountActions />}
        <AboutFooter />
      </ScrollView>
    </SafeAreaView>
  );
};

const GuestCard = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle} accessibilityRole="header">
        {t('profile.guestTitle')}
      </Text>
      <Text style={styles.cardHint}>{t('profile.guestHint')}</Text>
      <PrimaryButton label={t('profile.signIn')} onPress={() => router.push('/sign-in')} />
    </View>
  );
};

const SignedInProfile = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const user = useSessionStore((state) => state.user);
  const { profile, loadError, saveError, load, update, dismissSaveError } = useProfileStore();
  const [sheet, setSheet] = useState<Sheet>(null);
  useLoadProfileOnMount();

  useEffect(() => {
    if (!saveError) return;
    void showDialog({
      title: t('profile.saveFailed'),
      message: t(`apiErrors.${saveError}`),
      confirmLabel: t('common.ok'),
    }).then(dismissSaveError);
  }, [saveError, dismissSaveError, t]);

  const name = profile?.displayName ?? user?.displayName ?? '';

  const editName = async () => {
    const next = await showPrompt({
      title: t('profile.displayName'),
      value: name,
      confirmLabel: t('common.save'),
      cancelLabel: t('common.cancel'),
      maxLength: 40,
    });
    if (next && next !== name) void update({ displayName: next });
  };

  const changeMyLanguage = (language: string) => {
    void update({ myLanguage: language });
    const preferences = usePreferencesStore.getState();
    if (SUPPORTED_LANGUAGES.some((code) => code === language)) {
      const code = language as (typeof SUPPORTED_LANGUAGES)[number];
      preferences.addLanguage(code);
      preferences.setMyLanguage(code);
    }
  };

  return (
    <>
      <View style={styles.identity}>
        <View style={styles.avatar} accessibilityElementsHidden>
          <Text style={styles.avatarText}>{name.slice(0, 1).toUpperCase()}</Text>
        </View>
        <View style={styles.identityText}>
          <Text style={styles.name} numberOfLines={1} accessibilityRole="header">
            {name}
          </Text>
          {!!(profile?.email ?? user?.email) && (
            <Text style={styles.email} numberOfLines={1}>
              {profile?.email ?? user?.email}
            </Text>
          )}
        </View>
      </View>

      {loadError && !profile ? (
        <ListSection footer={t(`apiErrors.${loadError}`)}>
          <ListRow title={t('common.retry')} tone="action" onPress={() => void load()} />
        </ListSection>
      ) : (
        <ListSection title={t('profile.section')} footer={t('profile.retentionHint')}>
          <ListRow title={t('profile.displayName')} chevron onPress={() => void editName()}>
            <ListValue>{name}</ListValue>
          </ListRow>
          <ListRow title={t('profile.myLanguage')} chevron onPress={() => setSheet('language')}>
            <ListValue>{profile ? languageName(t, profile.myLanguage) : ''}</ListValue>
          </ListRow>
          <ListRow title={t('profile.keepRecords')} chevron onPress={() => setSheet('retention')}>
            <ListValue>{profile ? t(`profile.retention.${profile.retention}`) : ''}</ListValue>
          </ListRow>
        </ListSection>
      )}

      {profile && (
        <>
          <OptionSheet
            visible={sheet === 'language'}
            title={t('profile.myLanguage')}
            options={SUPPORTED_LANGUAGES.map((code) => ({
              value: code,
              label: languageName(t, code),
            }))}
            selected={profile.myLanguage}
            onSelect={changeMyLanguage}
            onClose={() => setSheet(null)}
          />
          <OptionSheet<Retention>
            visible={sheet === 'retention'}
            title={t('profile.keepRecords')}
            options={RETENTION_OPTIONS.map((value) => ({
              value,
              label: t(`profile.retention.${value}`),
            }))}
            selected={profile.retention}
            footer={t('profile.retentionHint')}
            onSelect={(retention) => void update({ retention })}
            onClose={() => setSheet(null)}
          />
        </>
      )}
    </>
  );
};

const AppSettings = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const uiLanguage = usePreferencesStore((state) => state.uiLanguage);
  const theme = usePreferencesStore((state) => state.theme);

  return (
    <>
      <ListSection title={t('settings.display')}>
        <ListRow
          title={t('settings.language')}
          chevron
          onPress={() => router.push('/settings/language')}
        >
          <ListValue>
            {uiLanguage === 'system' ? t('settings.systemLanguage') : UI_LANGUAGE_NAMES[uiLanguage]}
          </ListValue>
        </ListRow>
      </ListSection>

      <View>
        <Text style={styles.sectionTitle} accessibilityRole="header">
          {t('settings.theme')}
        </Text>
        <SegmentedControl
          options={THEME_PREFERENCES}
          value={theme}
          label={(option) => t(`settings.themes.${option}`)}
          onChange={usePreferencesStore.getState().setTheme}
        />
      </View>

      <ListSection title={t('settings.general')}>
        <ListRow
          title={t('settings.about')}
          chevron
          onPress={() => openConversation({ demo: true })}
        />
      </ListSection>

      <ListSection title={t('settings.legal')}>
        <ListRow
          title={t('settings.privacy')}
          chevron
          onPress={() => router.push('/settings/privacy')}
        />
        <ListRow
          title={t('settings.terms')}
          chevron
          onPress={() => router.push('/settings/terms')}
        />
      </ListSection>
    </>
  );
};

const AccountActions = () => {
  const { t } = useTranslation();

  const confirmDelete = async () => {
    const confirmed = await showDialog({
      title: t('profile.deleteConfirmTitle'),
      message: t('profile.deleteConfirmMessage'),
      confirmLabel: t('profile.deleteConfirmAction'),
      cancelLabel: t('common.cancel'),
      destructive: true,
    });
    if (!confirmed) return;
    try {
      getConversationController().end();
      await deleteAccount();
      await showDialog({
        title: t('profile.deletedTitle'),
        message: t('profile.deletedMessage'),
        confirmLabel: t('common.ok'),
      });
    } catch {
      await showDialog({
        title: t('profile.deleteFailed'),
        message: t('profile.deleteFailedMessage'),
        confirmLabel: t('common.ok'),
      });
    }
  };

  return (
    <ListSection title={t('profile.account')} footer={t('profile.deleteHint')}>
      <ListRow title={t('profile.signOut')} tone="action" onPress={() => void signOut()} />
      <ListRow
        title={t('profile.deleteAccount')}
        tone="destructive"
        onPress={() => void confirmDelete()}
      />
    </ListSection>
  );
};

const AboutFooter = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  return (
    <View style={styles.footer}>
      <Text style={styles.disclaimer}>{t('settings.disclaimer')}</Text>
      <Text style={styles.version}>
        STiTy {t('settings.version', { version: appInfo.version })}
      </Text>
      <Text style={styles.version}>{t('about.copyright')}</Text>
    </View>
  );
};

const AVATAR = 52;

const useStyles = makeStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  content: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.xxxl,
    gap: spacing.xxl,
  },
  card: {
    gap: spacing.sm,
    padding: spacing.lg,
    borderRadius: radius.card,
    backgroundColor: colors.surface,
  },
  cardTitle: { ...typography.title, color: colors.text },
  cardHint: { ...typography.caption, color: colors.textSecondary, marginBottom: spacing.sm },
  identity: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.xs,
  },
  avatar: {
    width: AVATAR,
    height: AVATAR,
    borderRadius: AVATAR / 2,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.accent,
  },
  avatarText: { ...typography.title, fontSize: 22, color: colors.onAccent },
  identityText: { flex: 1, gap: 2 },
  name: { ...typography.title, color: colors.text },
  email: { ...typography.caption, color: colors.textTertiary },
  sectionTitle: {
    ...typography.small,
    color: colors.textTertiary,
    marginBottom: spacing.sm,
    marginLeft: spacing.xs,
  },
  footer: { gap: spacing.sm, paddingHorizontal: spacing.xs },
  disclaimer: { ...typography.caption, color: colors.textTertiary },
  version: { ...typography.small, color: colors.textTertiary },
}));
