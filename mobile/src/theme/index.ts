import { useMemo, useSyncExternalStore } from 'react';
import { StyleSheet, useColorScheme } from 'react-native';

import { themes, type Colors, type ColorScheme } from './colors';

export type { Colors, ColorScheme } from './colors';
export { radius, spacing } from './spacing';
export { fonts, typography } from './typography';

type Styles = Parameters<typeof StyleSheet.create>[0];

let override: ColorScheme | null = null;
const overrideListeners = new Set<() => void>();

/** Forces the app's scheme regardless of the system; null follows the system again. */
export const setColorSchemeOverride = (scheme: ColorScheme | null) => {
  override = scheme;
  overrideListeners.forEach((listener) => listener());
};

const subscribeToOverride = (listener: () => void) => {
  overrideListeners.add(listener);
  return () => {
    overrideListeners.delete(listener);
  };
};

const currentOverride = () => override;

/** `fixed` pins a screen to one scheme, such as Present, which is dark in either theme. */
export const useColorSchemeName = (fixed?: ColorScheme): ColorScheme => {
  const system = useColorScheme();
  const forced = useSyncExternalStore(subscribeToOverride, currentOverride, currentOverride);
  return fixed ?? forced ?? (system === 'dark' ? 'dark' : 'light');
};

export const useColors = (fixed?: ColorScheme): Colors => themes[useColorSchemeName(fixed)];

/** Builds a style sheet per color scheme once, and returns the one for the current scheme. */
export const makeStyles = <S extends Styles>(factory: (colors: Colors) => S & Styles) => {
  const cache = new Map<ColorScheme, Readonly<S>>();
  return (fixed?: ColorScheme): Readonly<S> => {
    const scheme = useColorSchemeName(fixed);
    return useMemo(() => {
      let styles = cache.get(scheme);
      if (!styles) {
        styles = StyleSheet.create(factory(themes[scheme]));
        cache.set(scheme, styles);
      }
      return styles;
    }, [scheme]);
  };
};
