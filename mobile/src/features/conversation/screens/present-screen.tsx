import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Pressable, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, { Defs, LinearGradient, Rect, Stop } from 'react-native-svg';

import { goBack } from '@/components/go-back';
import { Logo } from '@/components/logo';
import { SessionControls } from '@/features/conversation/components/session-controls';
import {
  setPresenting,
  useConversationStore,
} from '@/features/conversation/model/conversation-store';
import type { TranscriptEntry } from '@/features/conversation/model/transcript';
import { screenOrientation } from '@/services/screen-orientation';
import { fonts, spacing } from '@/theme';
import { themes } from '@/theme/colors';

const KEEP_AWAKE_TAG = 'present';
const CONTROLS_VISIBLE_MS = 4_000;
const RECENT_SENTENCES = 12;
const FADE_HEIGHT = 40;
const STAGE_BOTTOM_PADDING = spacing.xxl;

/** Projectors wash out light backgrounds, so this screen is dark in either theme. */
const colors = themes.dark;

/**
 * Sized from the largest 16:9 frame that fits, so a phone and a mirrored TV both read from the
 * back row, and a phone that couldn't turn sideways still fits a sentence.
 */
const headlineSize = (width: number, height: number) =>
  Math.round(Math.min(72, Math.max(28, Math.min(height, (width * 9) / 16) * 0.09)));

/**
 * The conversation for a room: landscape, made to be mirrored to a big screen. Sentences fill from
 * the top, then the newest holds the bottom while older ones fade out under the logo, translation
 * large and original under it; a tap shows or hides the controls.
 */
export const PresentScreen = () => {
  const { t } = useTranslation();
  const { width, height } = useWindowDimensions();
  const insets = useSafeAreaInsets();
  const entries = useConversationStore((state) => state.entries);
  const partial = useConversationStore((state) => state.partial);
  const phase = useConversationStore((state) => state.phase);
  const error = useConversationStore((state) => state.error);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [stageHeight, setStageHeight] = useState(0);
  const [sentencesHeight, setSentencesHeight] = useState(0);
  const recording = phase === 'recording';
  const [wasRecording, setWasRecording] = useState(recording);
  const [controlsShown, setControlsShown] = useState(!recording);

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

  // Stopped, connecting or failed, the controls come up so the room sees why.
  if (recording !== wasRecording) {
    setWasRecording(recording);
    if (!recording) setControlsShown(true);
  }
  useEffect(() => {
    if (!recording && hideTimer.current) clearTimeout(hideTimer.current);
  }, [recording]);

  const showControls = () => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    setControlsShown(true);
    if (recording) {
      hideTimer.current = setTimeout(() => setControlsShown(false), CONTROLS_VISIBLE_MS);
    }
  };

  const hideControls = () => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    setControlsShown(false);
  };

  const size = headlineSize(width, height);
  const shown = entries.slice(-RECENT_SENTENCES);
  const overflowing = sentencesHeight > stageHeight - FADE_HEIGHT - STAGE_BOTTOM_PADDING;

  return (
    <Pressable
      style={[styles.screen, { paddingLeft: insets.left, paddingRight: insets.right }]}
      onPress={controlsShown ? hideControls : showControls}
      accessibilityLabel={t('present.showControls')}
    >
      <StatusBar hidden style="light" />
      <View style={[styles.header, { paddingTop: Math.max(insets.top, spacing.lg) }]}>
        <Logo scheme="dark" />
        <View
          style={[
            styles.dot,
            { backgroundColor: recording ? colors.recording : colors.textTertiary },
          ]}
        />
      </View>

      <View
        style={[styles.stage, overflowing && styles.stageFull]}
        onLayout={(event) => setStageHeight(event.nativeEvent.layout.height)}
        accessibilityLiveRegion="polite"
      >
        <View
          style={styles.sentences}
          onLayout={(event) => setSentencesHeight(event.nativeEvent.layout.height)}
        >
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
        <TopFade />
      </View>

      {error && <Text style={styles.error}>{t(`errors.${error}`)}</Text>}

      {controlsShown && (
        <Pressable
          style={[styles.controls, { bottom: Math.max(insets.bottom, spacing.lg) }]}
          onPress={showControls}
          accessible={false}
        >
          <SessionControls present={{ scheme: 'dark', onExit: goBack }} onInteract={showControls} />
        </Pressable>
      )}
    </Pressable>
  );
};

const TopFade = () => (
  <Svg width="100%" height={FADE_HEIGHT} style={styles.fade} pointerEvents="none">
    <Defs>
      <LinearGradient id="present-top-fade" x1="0" y1="0" x2="0" y2="1">
        <Stop offset="0" stopColor={colors.background} stopOpacity={1} />
        <Stop offset="1" stopColor={colors.background} stopOpacity={0} />
      </LinearGradient>
    </Defs>
    <Rect width="100%" height="100%" fill="url(#present-top-fade)" />
  </Svg>
);

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
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.xxl * 2,
    paddingBottom: spacing.md,
  },
  dot: { width: 12, height: 12, borderRadius: 6 },
  stage: {
    flex: 1,
    overflow: 'hidden',
    paddingHorizontal: spacing.xxl * 2,
    paddingTop: FADE_HEIGHT,
    paddingBottom: STAGE_BOTTOM_PADDING,
  },
  stageFull: { justifyContent: 'flex-end' },
  sentences: { gap: spacing.xl },
  fade: { position: 'absolute', top: 0, left: 0 },
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
