import { router } from 'expo-router';
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { showDialog } from '@/components/dialog';
import { appInfo } from '@/config/app-info';
import { env } from '@/config/env';
import { normalizeServerUrl } from '@/config/server-override';
import { changeServer } from '@/features/account/account-runtime';

/**
 * A link can name a server. Anyone can send such a link, so a development build asks before it
 * moves the microphone stream there, and any other build ignores it. The app reopens on the same
 * link after the switch and finds the server already in use.
 */
export const useServerLink = (url: string | undefined) => {
  const { t } = useTranslation();
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

  return { allowed, target, pending };
};
