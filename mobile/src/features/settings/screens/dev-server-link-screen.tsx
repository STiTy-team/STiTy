import { Redirect, router, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { showDialog } from '@/components/dialog';
import { LoadingView, StateView } from '@/components/state-view';
import { appInfo } from '@/config/app-info';
import { env } from '@/config/env';
import { normalizeServerUrl } from '@/config/server-override';
import { changeServer } from '@/features/account/account-runtime';

/**
 * Opened by the QR code `scripts/server/tunnel.sh` prints: stity-dev://dev-server?url=…, or the
 * web app's /dev-server?url=…. Anyone can send such a link, so a development build asks before it
 * moves the microphone stream to another server; any other build ignores the link. The app reopens
 * on this route after the switch, which then finds the server already in use and goes home.
 */
export const DevServerLinkScreen = () => {
  const { t } = useTranslation();
  const { url } = useLocalSearchParams<{ url?: string }>();
  const allowed = appInfo.variant === 'development';
  const target = normalizeServerUrl(url ?? '');
  const pending = allowed && target !== null && target !== env.serverUrl;
  const asked = useRef(false);

  useEffect(() => {
    if (!pending || !target || asked.current) return;
    asked.current = true;
    void (async () => {
      const confirmed = await showDialog({
        title: t('developer.switchTitle'),
        message: t('developer.switchMessage', { server: target }),
        confirmLabel: t('developer.connect'),
        cancelLabel: t('common.cancel'),
      });
      if (confirmed) await changeServer(target);
      else router.replace('/');
    })();
  }, [pending, target, t]);

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
