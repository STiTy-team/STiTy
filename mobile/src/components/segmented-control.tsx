import { Pressable, Text, View } from 'react-native';

import { makeStyles, radius, typography } from '@/theme';

type Props<T extends string> = {
  options: readonly T[];
  value: T;
  label: (option: T) => string;
  onChange: (option: T) => void;
};

export const SegmentedControl = <T extends string>({
  options,
  value,
  label,
  onChange,
}: Props<T>) => {
  const styles = useStyles();
  return (
    <View style={styles.track} accessibilityRole="radiogroup">
      {options.map((option) => {
        const selected = option === value;
        return (
          <Pressable
            key={option}
            onPress={() => onChange(option)}
            accessibilityRole="radio"
            accessibilityState={{ selected }}
            style={[styles.segment, selected && styles.selected]}
          >
            <Text style={[styles.text, selected && styles.textSelected]} numberOfLines={1}>
              {label(option)}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  track: {
    flexDirection: 'row',
    backgroundColor: colors.surface,
    borderRadius: radius.control,
    padding: 3,
  },
  segment: {
    flex: 1,
    minHeight: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radius.segment,
    paddingHorizontal: 6,
  },
  selected: {
    backgroundColor: colors.raised,
    shadowColor: '#000000',
    shadowOpacity: 0.08,
    shadowRadius: 2,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  text: { ...typography.label, color: colors.textSecondary },
  textSelected: { color: colors.text },
}));
