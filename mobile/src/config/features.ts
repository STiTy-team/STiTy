import Constants from 'expo-constants';
import { Platform } from 'react-native';

import { resolveFeatures } from '@/config/feature-switches';

export const features = resolveFeatures(Constants.expoConfig?.extra?.features, Platform.OS);
