import * as ScreenOrientation from 'expo-screen-orientation';

import { createLogger } from '@/lib/logger';

const log = createLogger('orientation');

const lock = (orientation: ScreenOrientation.OrientationLock) =>
  ScreenOrientation.lockAsync(orientation).catch((error: unknown) =>
    log.warn('orientation lock unavailable', error),
  );

/** The app is portrait; only Present mode turns the phone sideways. */
export const screenOrientation = {
  portrait: () => lock(ScreenOrientation.OrientationLock.PORTRAIT_UP),
  landscape: () => lock(ScreenOrientation.OrientationLock.LANDSCAPE),
};
