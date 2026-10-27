import { memo, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Text, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';

import { languageName } from '@/features/conversation/language-name';
import { makeStyles, radius, spacing, typography } from '@/theme';
import { APPEAR_MS, APPEAR_OFFSET } from './transcript-item.constants';

/**
 * - appear: a new final slides up and fades in.
 * - firm: a final that replaces the partial in place; it is already on screen.
 * - partial: a live sentence that is still being transcribed.
 */
export type ItemEntrance = 'appear' | 'firm' | 'partial';

type Props = {
  language: string;
  text: string;
  translation: string;
  mine: boolean;
  entrance: ItemEntrance;
};

/**
 * Someone else's sentence leads with the translation, since that is what this phone's owner
 * reads. My own sentence leads with what I said, so I can check it was heard right.
 */
export const TranscriptItem = memo(function TranscriptItem({
  language,
  text,
  translation,
  mine,
  entrance,
}: Props) {
  const { t } = useTranslation();
  const styles = useStyles();
  const live = entrance === 'partial';

  const opacity = useSharedValue(entrance === 'firm' ? 1 : 0);
  const offset = useSharedValue(entrance === 'firm' ? 0 : APPEAR_OFFSET);
  useEffect(() => {
    const config = { duration: APPEAR_MS, easing: Easing.out(Easing.cubic) };
    opacity.value = withTiming(1, config);
    offset.value = withTiming(0, config);
  }, [opacity, offset]);
  const animated = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateY: offset.value }],
  }));

  const headline = mine ? text : translation || text;
  const secondary = mine ? translation : translation ? text : '';
  const code = language.toUpperCase();

  return (
    <Animated.View style={[styles.item, mine && styles.mine, animated]}>
      <View style={styles.who}>
        <View style={[styles.avatar, mine && styles.avatarMine]}>
          <Text style={[styles.avatarText, mine && styles.avatarTextMine]}>
            {mine ? t('conversation.me') : code}
          </Text>
        </View>
        <Text style={styles.whoText} numberOfLines={1}>
          {live ? t('conversation.transcribing') : languageName(t, language)}
        </Text>
      </View>
      <Text
        style={[mine ? styles.spoken : styles.translation, live && styles.live]}
        accessibilityLiveRegion={live ? 'polite' : 'none'}
      >
        {headline}
        {live && <Text style={styles.ellipsis}> …</Text>}
      </Text>
      {!!secondary && <Text style={styles.secondary}>{secondary}</Text>}
    </Animated.View>
  );
});

const useStyles = makeStyles((colors) => ({
  item: { marginHorizontal: spacing.lg, marginBottom: spacing.xl },
  mine: {
    marginLeft: spacing.xxxl + spacing.xs,
    backgroundColor: colors.surface,
    borderRadius: radius.card,
    paddingVertical: spacing.md,
    paddingHorizontal: 14,
  },
  who: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 6 },
  avatar: {
    minWidth: 22,
    height: 22,
    borderRadius: 11,
    paddingHorizontal: 4,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.surfaceStrong,
  },
  avatarMine: { backgroundColor: colors.accent },
  avatarText: { ...typography.badge, fontSize: 9, color: colors.textSecondary },
  avatarTextMine: { color: colors.onAccent },
  whoText: { ...typography.small, color: colors.textTertiary, flexShrink: 1 },
  translation: { ...typography.translation, color: colors.text },
  spoken: { ...typography.bodyStrong, color: colors.text },
  live: { color: colors.textTertiary },
  ellipsis: { color: colors.textTertiary },
  secondary: { ...typography.caption, color: colors.textTertiary, marginTop: spacing.xs },
}));
