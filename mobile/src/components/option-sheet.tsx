import { Ionicons } from '@expo/vector-icons';
import { ScrollView } from 'react-native';

import { BottomSheet } from '@/components/bottom-sheet';
import { ListRow, ListSection } from '@/components/list';
import { useColors } from '@/theme';

type Props<T extends string> = {
  visible: boolean;
  title: string;
  options: readonly { value: T; label: string }[];
  selected: T;
  footer?: string;
  onSelect: (value: T) => void;
  onClose: () => void;
};

/** Picks one value from a short list; closes on choice. */
export const OptionSheet = <T extends string>({
  visible,
  title,
  options,
  selected,
  footer,
  onSelect,
  onClose,
}: Props<T>) => {
  const colors = useColors();
  return (
    <BottomSheet visible={visible} title={title} onClose={onClose}>
      <ScrollView>
        <ListSection footer={footer}>
          {options.map((option) => (
            <ListRow
              key={option.value}
              title={option.label}
              accessibilityState={{ selected: option.value === selected }}
              onPress={() => {
                onSelect(option.value);
                onClose();
              }}
            >
              {option.value === selected && (
                <Ionicons name="checkmark" size={20} color={colors.accent} />
              )}
            </ListRow>
          ))}
        </ListSection>
      </ScrollView>
    </BottomSheet>
  );
};
