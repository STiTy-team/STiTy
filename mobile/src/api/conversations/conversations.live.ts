import type { HttpClient } from '@/api/http';
import { pageSchema, recordSummarySchema, transcriptDocumentSchema } from '@/api/schemas';
import type { ConversationsApi } from '@/api/types';

const recordPath = (id: string) => `/conversations/${encodeURIComponent(id)}`;

export const createLiveConversations = (http: HttpClient): ConversationsApi => ({
  list: (query) =>
    http.request({
      path: '/conversations',
      query,
      schema: pageSchema(recordSummarySchema),
      auth: true,
    }),
  get: (id) => http.request({ path: recordPath(id), schema: transcriptDocumentSchema, auth: true }),
  rename: (id, title) =>
    http.request({ method: 'PATCH', path: recordPath(id), body: { title }, auth: true }),
  remove: (id) => http.request({ method: 'DELETE', path: recordPath(id), auth: true }),
});
