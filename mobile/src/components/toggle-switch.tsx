import { Switch, type SwitchProps } from 'react-native';

import { useColors } from '@/theme';

export const ToggleSwitch = (props: Omit<SwitchProps, 'trackColor' | 'thumbColor'>) => {
  const colors = useColors();
  return (
    <Switch
      {...props}
      trackColor={{ true: colors.accent, false: colors.surfaceStrong }}
      thumbColor="#FFFFFF"
    />
  );
};
