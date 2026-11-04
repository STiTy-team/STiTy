import { useEffect, useState } from 'react';
import { Image, StyleSheet, View } from 'react-native';
import Animated, {
  Easing,
  FadeIn,
  FadeOut,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';

import { deviceLanguage } from '@/i18n/device-language';
import { usePreferencesStore } from '@/stores/preferences-store';
import { makeStyles, typography, useColorSchemeName } from '@/theme';

import { greetingSequence, INTRO_TIMING } from './intro-sequence';

const LOGO = {
  light: require('../../../assets/images/splash-icon.png'),
  dark: require('../../../assets/images/splash-icon-dark.png'),
};

/** Same width as `imageWidth` in the splash-screen plugin, so the hand-off is seamless. */
const LOGO_SIZE = 200;
const LIFT = -28;

type Props = {
  /** The first frame is on screen and matches the native splash, which can now hide. */
  onShown: () => void;
  onDone: () => void;
};

/**
 * Starts as an exact copy of the native splash, then lifts the wordmark and says hello in each
 * conversation language, ending in the user's own, before fading into the app.
 */
export const IntroOverlay = ({ onShown, onDone }: Props) => {
  const styles = useStyles();
  const scheme = useColorSchemeName();
  const reduceMotion = useReducedMotion();
  const uiLanguage = usePreferencesStore((state) => state.uiLanguage);
  const [words] = useState(() =>
    greetingSequence(uiLanguage === 'system' ? deviceLanguage() : uiLanguage),
  );
  const [index, setIndex] = useState(-1);

  const lift = useSharedValue(0);
  const opacity = useSharedValue(1);

  useEffect(() => {
    const { liftMs, firstWordAtMs, wordMs, holdLastMs, fadeOutMs, reducedMotionHoldMs } =
      INTRO_TIMING;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const at = (ms: number, run: () => void) => timers.push(setTimeout(run, ms));

    let fadeAt = reducedMotionHoldMs;
    if (!reduceMotion) {
      lift.value = withTiming(LIFT, { duration: liftMs, easing: Easing.out(Easing.cubic) });
      words.forEach((_, word) => at(firstWordAtMs + word * wordMs, () => setIndex(word)));
      fadeAt = firstWordAtMs + (words.length - 1) * wordMs + holdLastMs;
    }
    at(fadeAt, () => {
      opacity.value = withTiming(0, { duration: fadeOutMs, easing: Easing.in(Easing.quad) });
    });
    at(fadeAt + fadeOutMs, onDone);
    return () => timers.forEach(clearTimeout);
  }, [reduceMotion, words, lift, opacity, onDone]);

  const overlayStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));
  const logoStyle = useAnimatedStyle(() => ({ transform: [{ translateY: lift.value }] }));

  return (
    <Animated.View
      style={[StyleSheet.absoluteFill, styles.overlay, overlayStyle]}
      onLayout={onShown}
      accessible
      accessibilityLabel="STiTy"
    >
      <Animated.View style={logoStyle}>
        <Image source={LOGO[scheme]} style={styles.logo} resizeMode="contain" />
      </Animated.View>
      <View style={styles.greeting} importantForAccessibility="no-hide-descendants">
        {index >= 0 && (
          <Animated.Text
            key={index}
            entering={FadeIn.duration(120).delay(70)}
            exiting={FadeOut.duration(70)}
            style={styles.word}
          >
            {words[index]}
          </Animated.Text>
        )}
      </View>
    </Animated.View>
  );
};

const useStyles = makeStyles((colors) => ({
  overlay: {
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10,
  },
  logo: { width: LOGO_SIZE, height: LOGO_SIZE },
  greeting: {
    position: 'absolute',
    top: '50%',
    left: 0,
    right: 0,
    height: 32,
    marginTop: 20,
    alignItems: 'center',
  },
  word: { ...typography.title, color: colors.textTertiary, position: 'absolute' },
}));
