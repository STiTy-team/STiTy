import { useEffect } from 'react';
import { AppState } from 'react-native';

import { restoreSession, watchSession } from '@/features/account/account-runtime';
import { screenOrientation } from '@/services/screen-orientation';
import { installCrashReporting } from '@/services/telemetry';
import { loadCapabilities, useCapabilitiesStore } from '@/stores/capabilities-store';

/** Portrait lock, crash reporting, the signed-in user and what the server can do; none of it blocks the UI. */
export const useStartup = () => {
  useEffect(() => {
    installCrashReporting();
    void screenOrientation.portrait();
    const stopWatching = watchSession();
    void restoreSession();
    void loadCapabilities();

    // A server that was down at launch is asked again when the app comes back.
    const appState = AppState.addEventListener('change', (next) => {
      if (next === 'active' && useCapabilitiesStore.getState().failed) void loadCapabilities();
    });
    return () => {
      appState.remove();
      stopWatching();
    };
  }, []);
};
