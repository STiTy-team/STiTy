import * as Haptics from 'expo-haptics';
import { Platform } from 'react-native';

export const haptics = {
  /** A translation arrived where the screen may not be watched, like a loud street. */
  translationArrived: () => {
    if (Platform.OS !== 'web')
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  },
  press: () => {
    if (Platform.OS !== 'web') void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  },
};
