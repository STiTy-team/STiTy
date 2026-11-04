import { Ionicons } from '@expo/vector-icons';
import type { ComponentProps } from 'react';
import { ActivityIndicator, Pressable, Text } from 'react-native';

import { makeStyles, radius, typography, useColors } from '@/theme';

type Props = {
  label: string;
  onPress: () => void;
  icon?: ComponentProps<typeof Ionicons>['name'];
  disabled?: boolean;
  loading?: boolean;
  /**
   * Secondary sits on the surface colour, for a choice next to the main action. Contrast is
   * black on light and white on dark, as Sign in with Apple requires.
   */
  variant?: 'primary' | 'secondary' | 'contrast';
};

export const PrimaryButton = ({
  label,
  onPress,
  icon,
  disabled = false,
  loading = false,
  variant = 'primary',
}: Props) => {
  const styles = useStyles();
  const colors = useColors();
  const inactive = disabled || loading;
  const foreground = {
    primary: colors.onAccent,
    secondary: colors.text,
    contrast: colors.background,
  }[variant];
  return (
    <Pressable
      onPress={onPress}
      disabled={inactive}
      accessibilityRole="button"
      accessibilityState={{ disabled: inactive, busy: loading }}
      style={({ pressed }) => [
        styles.button,
        variant !== 'primary' && styles[variant],
        inactive && styles.disabled,
        pressed && styles.pressed,
      ]}
    >
      {loading ? (
        <ActivityIndicator color={foreground} />
      ) : (
        icon && <Ionicons name={icon} size={20} color={foreground} />
      )}
      <Text style={[styles.label, { color: foreground }]}>{label}</Text>
    </Pressable>
  );
};

const useStyles = makeStyles((colors) => ({
  button: {
    minHeight: 56,
    borderRadius: radius.card,
    backgroundColor: colors.accent,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: 20,
  },
  secondary: { backgroundColor: colors.surface },
  contrast: { backgroundColor: colors.text },
  label: { ...typography.button },
  disabled: { opacity: 0.4 },
  pressed: { opacity: 0.85 },
}));
