import { useTranslation } from 'react-i18next';

import { Notice } from '@/components/notice';
import { useCapacityNotice } from '@/features/conversation/hooks/use-capacity-notice';

/**
 * Says why Start is disabled while the server is full, and when it opens up again. Connection
 * problems are shown by the header's status icon instead.
 */
export const ServerNotice = () => {
  const { t } = useTranslation();
  const { full, justFreed } = useCapacityNotice();

  const notice = full
    ? { text: t('server.full'), icon: 'hourglass-outline' as const, warn: true }
    : justFreed
      ? { text: t('server.freed'), icon: 'checkmark-circle-outline' as const, warn: false }
      : null;

  return notice && <Notice icon={notice.icon} text={notice.text} warn={notice.warn} />;
};
