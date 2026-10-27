import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useTranslation } from 'react-i18next';
import { Pressable, Text, View } from 'react-native';

import { endedActions, useEndedStore } from '@/features/conversation/model/ended-store';
import { useFeatures } from '@/stores/capabilities-store';
import { selectSignedIn, useSessionStore } from '@/stores/session-store';
import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

/**
 * After a conversation: "Saved · Open" when the server recorded it, or a sign-in prompt for a
 * guest. Saving a guest's conversation afterwards is left out on purpose.
 */
export const EndedBanner = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const colors = useColors();
  const last = useEndedStore((state) => state.last);
  const signedIn = useSessionStore(selectSignedIn);
  const { accounts } = useFeatures();

  if (!last) return null;
  const conversationId = last.conversationId;
  if (!conversationId && (signedIn || !accounts)) return null;

  const action = conversationId
    ? {
        label: t('ended.open'),
        onPress: () => {
          endedActions.dismiss();
          router.push({ pathname: '/records/[id]', params: { id: conversationId } });
        },
      }
    : { label: t('ended.signIn'), onPress: () => router.push('/sign-in') };

  return (
    <View style={styles.banner} accessibilityLiveRegion="polite">
      <Ionicons
        name={conversationId ? 'checkmark-circle' : 'bookmark-outline'}
        size={20}
        color={conversationId ? colors.accent : colors.textSecondary}
      />
      <Text style={styles.text}>{t(conversationId ? 'ended.saved' : 'ended.guestPrompt')}</Text>
      <Pressable onPress={action.onPress} accessibilityRole="button" hitSlop={8}>
        <Text style={styles.action}>{action.label}</Text>
      </Pressable>
      <Pressable
        onPress={endedActions.dismiss}
        accessibilityRole="button"
        accessibilityLabel={t('ended.dismiss')}
        hitSlop={8}
      >
        <Ionicons name="close" size={18} color={colors.textTertiary} />
      </Pressable>
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm + 2,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    borderRadius: radius.control,
    backgroundColor: colors.surface,
  },
  text: { ...typography.caption, flex: 1, color: colors.text },
  action: { ...typography.label, color: colors.accent },
}));
