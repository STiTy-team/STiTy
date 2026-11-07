import { Redirect, router, useLocalSearchParams } from 'expo-router';
import { useTranslation } from 'react-i18next';

import { LoadingView, StateView } from '@/components/state-view';
import { useServerLink } from '@/features/settings/hooks/use-server-link';

/**
 * Opened by the QR code `scripts/server/tunnel.sh` prints: stity-dev://dev-server?url=…, or the
 * web app's /dev-server?url=…. Anyone can send such a link, so a development build asks before it
 * moves the microphone stream to another server; any other build ignores the link. The app reopens
 * on this route after the switch, which then finds the server already in use and goes home.
 */
export const DevServerLinkScreen = () => {
  const { t } = useTranslation();
  const { url } = useLocalSearchParams<{ url?: string }>();
  const { allowed, target, pending } = useServerLink(url);

  if (!allowed || (target && !pending)) return <Redirect href="/" />;
  if (!target) {
    return (
      <StateView
        icon="alert-circle-outline"
        title={t('developer.linkFailed')}
        message={url}
        action={{ label: t('common.ok'), onPress: () => router.replace('/') }}
      />
    );
  }
  return <LoadingView label={target} />;
};
