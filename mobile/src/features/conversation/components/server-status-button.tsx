import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import { ActivityIndicator } from 'react-native';

import { TooltipButton } from '@/components/tooltip-button';
import { useServerStore } from '@/stores/server-store';
import { useColors } from '@/theme';

/** A header icon while the server is not ready; tapping it says why. Hidden once ready. */
export const ServerStatusButton = () => {
  const { t } = useTranslation();
  const colors = useColors();
  const status = useServerStore((state) => state.status);

  if (status === 'unreachable') {
    return (
      <TooltipButton message={t('server.unreachable')}>
        <Ionicons name="cloud-offline-outline" size={22} color={colors.warning} />
      </TooltipButton>
    );
  }
  if (status === 'connecting') {
    return (
      <TooltipButton message={t('server.connecting')}>
        <ActivityIndicator size="small" color={colors.textTertiary} />
      </TooltipButton>
    );
  }
  return null;
};
