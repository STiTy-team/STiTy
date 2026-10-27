import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Pressable, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { goBack } from '@/components/go-back';
import { SessionControls } from '@/features/conversation/components/session-controls';
import {
  setPresenting,
  useConversationStore,
} from '@/features/conversation/model/conversation-store';
import type { TranscriptEntry } from '@/features/conversation/model/transcript';
import { screenOrientation } from '@/services/screen-orientation';
import { usePreferencesStore } from '@/stores/preferences-store';
import { fonts, spacing } from '@/theme';
import { themes } from '@/theme/colors';

const KEEP_AWAKE_TAG = 'present';
const CONTROLS_VISIBLE_MS = 4_000;
const VISIBLE_SENTENCES = 3;

/** Projectors wash out light backgrounds, so this screen is dark in either theme. */
const colors = themes.dark;

/**
 * Sized from the largest 16:9 frame that fits, so a phone and a mirrored TV both read from the
 * back row, and a phone that couldn't turn sideways still fits a sentence.
 */
const headlineSize = (width: number, height: number) =>
  Math.round(Math.min(72, Math.max(28, Math.min(height, (width * 9) / 16) * 0.09)));

/**
 * The conversation for a room: landscape, made to be mirrored to a big screen. Only the last few
 * sentences show, translation large and original under it; controls appear on a tap.
 */
export const PresentScreen = () => {
  const { t } = useTranslation();
  const { width, height } = useWindowDimensions();
  const insets = useSafeAreaInsets();
  const entries = useConversationStore((state) => state.entries);
  const partial = useConversationStore((state) => state.partial);
  const phase = useConversationStore((state) => state.phase);
  const error = useConversationStore((state) => state.error);
  const languages = usePreferencesStore((state) => state.languages);
  const [controlsShown, setControlsShown] = useState(false);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setPresenting(true);
    void screenOrientation.landscape();
    void activateKeepAwakeAsync(KEEP_AWAKE_TAG);
    return () => {
      setPresenting(false);
      void screenOrientation.portrait();
      void deactivateKeepAwake(KEEP_AWAKE_TAG);
    };
  }, []);

  useEffect(
    () => () => {
      if (hideTimer.current) clearTimeout(hideTimer.current);
    },
    [],
  );

  const showControls = () => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    setControlsShown(true);
    hideTimer.current = setTimeout(() => setControlsShown(false), CONTROLS_VISIBLE_MS);
  };

  const size = headlineSize(width, height);
  const recording = phase === 'recording';
  const shown = entries.slice(-VISIBLE_SENTENCES);

  return (
    <Pressable
      style={[styles.screen, { paddingLeft: insets.left, paddingRight: insets.right }]}
      onPress={showControls}
      accessibilityLabel={t('present.showControls')}
    >
      <StatusBar hidden style="light" />
      <View style={styles.badge}>
        <View
          style={[
            styles.dot,
            { backgroundColor: recording ? colors.recording : colors.textTertiary },
          ]}
        />
        <Text style={styles.badgeText}>
          {languages.map((code) => code.toUpperCase()).join(' ↔ ')}
        </Text>
      </View>

      <View style={styles.stage} accessibilityLiveRegion="polite">
        {shown.length === 0 && !partial && (
          <Text style={[styles.prompt, { fontSize: size * 0.7, lineHeight: size * 0.9 }]}>
            {t('conversation.emptyTitle')}
          </Text>
        )}
        {shown.map((entry, index) => (
          <Sentence
            key={entry.id}
            entry={entry}
            size={size}
            faded={index < shown.length - 1 || partial !== null}
          />
        ))}
        {partial && (
          <Text
            style={[styles.original, { fontSize: size * 0.55, lineHeight: size * 0.75 }]}
            numberOfLines={2}
          >
            {partial.text} …
          </Text>
        )}
      </View>

      {error && <Text style={styles.error}>{t(`errors.${error}`)}</Text>}

      {/* Stopped, connecting or failed, the controls stay up so the room sees why. */}
      {(controlsShown || !recording) && (
        <View style={[styles.controls, { bottom: Math.max(insets.bottom, spacing.lg) }]}>
          <SessionControls present={{ scheme: 'dark', onExit: goBack }} onInteract={showControls} />
        </View>
      )}
    </Pressable>
  );
};

const Sentence = ({
  entry,
  size,
  faded,
}: {
  entry: TranscriptEntry;
  size: number;
  faded: boolean;
}) => {
  const headline = entry.translation || entry.text;
  return (
    <View style={[styles.sentence, faded && styles.faded]}>
      <Text style={[styles.headline, { fontSize: size, lineHeight: size * 1.22 }]}>{headline}</Text>
      {!!entry.translation && (
        <Text style={[styles.original, { fontSize: size * 0.5, lineHeight: size * 0.7 }]}>
          {entry.text}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  badge: {
    position: 'absolute',
    top: spacing.lg,
    left: spacing.xl,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    opacity: 0.7,
  },
  dot: { width: 8, height: 8, borderRadius: 4 },
  badgeText: { fontFamily: fonts.medium, fontSize: 13, color: colors.textTertiary },
  stage: {
    flex: 1,
    justifyContent: 'flex-end',
    overflow: 'hidden',
    paddingHorizontal: spacing.xxl * 2,
    paddingTop: spacing.xxl * 2,
    paddingBottom: spacing.xxl,
    gap: spacing.xl,
  },
  sentence: { gap: spacing.xs },
  faded: { opacity: 0.4 },
  headline: { fontFamily: fonts.semibold, color: colors.text, letterSpacing: -0.4 },
  original: { fontFamily: fonts.regular, color: colors.textSecondary },
  prompt: { fontFamily: fonts.semibold, color: colors.textTertiary, textAlign: 'center' },
  error: {
    fontFamily: fonts.regular,
    fontSize: 15,
    color: colors.danger,
    textAlign: 'center',
    paddingBottom: spacing.lg,
  },
  controls: { position: 'absolute', alignSelf: 'center' },
});
