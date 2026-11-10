import { Platform } from 'react-native';

import type { DeviceInfo } from '@/api';
import { audioOutput } from '@/services/audio/audio-output';

/** Android names its model; iOS and the web don't without another dependency. */
export const deviceInfo = (): DeviceInfo => ({
  platform: Platform.OS,
  version: Platform.Version === undefined ? undefined : String(Platform.Version),
  model: (Platform.constants as { Model?: string } | undefined)?.Model,
  headset: audioOutput.isHeadsetConnected(),
});
