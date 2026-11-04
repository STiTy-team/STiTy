import { Pressable, Text } from 'react-native';

import { makeStyles, radius, spacing, typography } from '@/theme';

type Props = {
  label: string;
  selected: boolean;
  onPress: () => void;
};

/** A filter choice in a horizontal row. */
export const Chip = ({ label, selected, onPress }: Props) => {
  const styles = useStyles();
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="radio"
      accessibilityState={{ selected }}
      style={({ pressed }) => [styles.chip, selected && styles.selected, pressed && styles.pressed]}
    >
      <Text style={[styles.label, selected && styles.labelSelected]} numberOfLines={1}>
        {label}
      </Text>
    </Pressable>
  );
};

const useStyles = makeStyles((colors) => ({
  chip: {
    minHeight: 34,
    paddingHorizontal: spacing.md + 2,
    borderRadius: radius.pill,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  selected: { backgroundColor: colors.text },
  pressed: { opacity: 0.8 },
  label: { ...typography.label, color: colors.textSecondary },
  labelSelected: { color: colors.background },
}));
