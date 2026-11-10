import { useEffect, useState, type ReactNode } from 'react';
import {
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { makeStyles, radius, spacing, typography } from '@/theme';

const SLIDE_MS = 260;

type Props = {
  visible: boolean;
  title?: string;
  onClose: () => void;
  children: ReactNode;
};

/** The dimmed backdrop fades in place while only the sheet slides up from the bottom edge. */
export const BottomSheet = ({ visible, title, onClose, children }: Props) => {
  const styles = useStyles();
  const { bottom } = useSafeAreaInsets();
  const { height: windowHeight } = useWindowDimensions();
  const [mounted, setMounted] = useState(visible);
  const shown = useSharedValue(0);
  const sheetHeight = useSharedValue(windowHeight);
  if (visible && !mounted) setMounted(true);

  useEffect(() => {
    const config = { duration: SLIDE_MS, easing: Easing.out(Easing.cubic) };
    if (visible) {
      shown.value = withTiming(1, config);
      return;
    }
    shown.value = withTiming(0, config);
    const unmount = setTimeout(() => setMounted(false), SLIDE_MS);
    return () => clearTimeout(unmount);
  }, [visible, shown]);

  const scrim = useAnimatedStyle(() => ({ opacity: shown.value }));
  const sheet = useAnimatedStyle(() => ({
    transform: [{ translateY: (1 - shown.value) * sheetHeight.value }],
  }));

  return (
    <Modal
      visible={mounted}
      animationType="none"
      transparent
      statusBarTranslucent
      navigationBarTranslucent
      onRequestClose={onClose}
    >
      <KeyboardAvoidingView
        style={styles.overlay}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <Animated.View style={[StyleSheet.absoluteFill, styles.scrim, scrim]}>
          <Pressable style={styles.backdrop} onPress={onClose} accessibilityRole="button" />
        </Animated.View>
        <Animated.View
          style={[
            styles.sheet,
            { paddingBottom: Math.max(bottom, spacing.lg) + spacing.sm },
            sheet,
          ]}
          onLayout={({ nativeEvent }) => {
            sheetHeight.value = nativeEvent.layout.height;
          }}
        >
          <View style={styles.handle} />
          {title && (
            <Text style={styles.title} accessibilityRole="header">
              {title}
            </Text>
          )}
          {children}
        </Animated.View>
      </KeyboardAvoidingView>
    </Modal>
  );
};

const useStyles = makeStyles((colors) => ({
  overlay: { flex: 1, justifyContent: 'flex-end' },
  scrim: { backgroundColor: colors.scrim },
  backdrop: { flex: 1 },
  sheet: {
    backgroundColor: colors.background,
    borderTopLeftRadius: radius.sheet,
    borderTopRightRadius: radius.sheet,
    paddingTop: spacing.sm,
    paddingHorizontal: spacing.xl,
    maxHeight: '75%',
  },
  handle: {
    alignSelf: 'center',
    width: 36,
    height: 5,
    borderRadius: 3,
    backgroundColor: colors.surfaceStrong,
    marginBottom: spacing.lg,
  },
  title: { ...typography.title, color: colors.text, marginBottom: spacing.md },
}));
