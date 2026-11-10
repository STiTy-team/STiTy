import { useEffect } from 'react';
import { Platform } from 'react-native';

import { useColors } from './index';

/** Browsers draw their own scrollbars, so the web build colours them from the theme. */
export const useWebScrollbarTheme = () => {
  const { surfaceStrong } = useColors();

  useEffect(() => {
    if (Platform.OS !== 'web') return;
    const style = document.createElement('style');
    style.textContent = `* { scrollbar-width: thin; scrollbar-color: ${surfaceStrong} transparent; }`;
    document.head.appendChild(style);
    return () => style.remove();
  }, [surfaceStrong]);
};
