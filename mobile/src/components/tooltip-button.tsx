import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Modal, Pressable, Text, View, useWindowDimensions, type HostInstance } from 'react-native';

import { makeStyles, radius, spacing, typography } from '@/theme';

const AUTO_HIDE_MS = 4_000;
const GAP = 6;

type Anchor = { bottom: number; right: number };

type Props = {
  message: string;
  children: ReactNode;
};

/**
 * A 44 pt icon target that shows `message` in a bubble under it when tapped. Screen readers get
 * the message as the button's label, so they never need the bubble.
 */
export const TooltipButton = ({ message, children }: Props) => {
  const styles = useStyles();
  const { width } = useWindowDimensions();
  const target = useRef<HostInstance>(null);
  const [anchor, setAnchor] = useState<Anchor | null>(null);

  useEffect(() => {
    if (!anchor) return;
    const timer = setTimeout(() => setAnchor(null), AUTO_HIDE_MS);
    return () => clearTimeout(timer);
  }, [anchor]);

  const show = () =>
    target.current?.measureInWindow((x, y, w, h) =>
      setAnchor({ bottom: y + h + GAP, right: Math.max(spacing.sm, width - (x + w)) }),
    );

  return (
    <>
      <Pressable
        ref={target}
        onPress={() => (anchor ? setAnchor(null) : show())}
        accessibilityRole="button"
        accessibilityLabel={message}
        hitSlop={4}
        style={({ pressed }) => [styles.button, pressed && styles.pressed]}
      >
        {children}
      </Pressable>
      <Modal
        visible={anchor !== null}
        transparent
        animationType="fade"
        statusBarTranslucent
        navigationBarTranslucent
        onRequestClose={() => setAnchor(null)}
      >
        <Pressable style={styles.backdrop} onPress={() => setAnchor(null)}>
          {anchor && (
            <View style={[styles.bubble, { top: anchor.bottom, right: anchor.right }]}>
              <View style={styles.arrow} />
              <Text style={styles.text}>{message}</Text>
            </View>
          )}
        </Pressable>
      </Modal>
    </>
  );
};

const useStyles = makeStyles((colors) => ({
  button: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pressed: { backgroundColor: colors.surface },
  backdrop: { flex: 1 },
  bubble: {
    position: 'absolute',
    maxWidth: 260,
    paddingVertical: spacing.sm + 2,
    paddingHorizontal: spacing.md,
    borderRadius: radius.control,
    backgroundColor: colors.text,
    shadowColor: '#000000',
    shadowOpacity: 0.16,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 6,
  },
  arrow: {
    position: 'absolute',
    top: -5,
    right: 17,
    width: 10,
    height: 10,
    backgroundColor: colors.text,
    transform: [{ rotate: '45deg' }],
  },
  text: { ...typography.caption, color: colors.background },
}));
