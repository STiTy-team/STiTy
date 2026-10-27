import { z } from 'zod';

import { RETENTION_OPTIONS } from '@/domain/records';

/**
 * One schema per response, shared by the live and mock adapters, so a mock can't drift from the
 * contract without a test failing. Proposed contract: "Feature specs §6" of the improvement plan.
 */

export const capabilitiesSchema = z.object({
  languages: z.array(z.string()),
  features: z.object({
    accounts: z.boolean(),
    records: z.boolean(),
    summary: z.boolean(),
    speakers: z.boolean(),
  }),
});

export const userSchema = z.object({
  id: z.string(),
  displayName: z.string(),
  email: z.string().optional(),
});

export const authResponseSchema = z.object({
  user: userSchema,
  accessToken: z.string(),
  refreshToken: z.string(),
});

export const profileSchema = userSchema.extend({
  myLanguage: z.string(),
  retention: z.enum(RETENTION_OPTIONS),
});

export const recordSummarySchema = z.object({
  id: z.string(),
  title: z.string(),
  startedAt: z.string(),
  durationMs: z.number().nonnegative(),
  languages: z.array(z.string()),
  firstLine: z.string(),
});

export const pageSchema = <T extends z.ZodType>(item: T) =>
  z.object({ items: z.array(item), nextCursor: z.string().nullable() });

export const transcriptDocumentSchema = z.object({
  id: z.string(),
  title: z.string(),
  startedAt: z.string(),
  endedAt: z.string(),
  languages: z.array(z.string()),
  myLanguage: z.string(),
  speakers: z.array(
    z.object({ id: z.string(), label: z.string(), language: z.string(), isMe: z.boolean() }),
  ),
  paragraphs: z.array(
    z.object({
      id: z.string(),
      speakerId: z.string(),
      startMs: z.number(),
      endMs: z.number(),
      language: z.string(),
      original: z.string(),
      translation: z.string(),
    }),
  ),
  summary: z.object({
    status: z.enum(['ready', 'pending', 'unavailable']),
    overview: z.string().optional(),
    keyPoints: z.array(z.string()).optional(),
    decisions: z.array(z.string()).optional(),
    actionItems: z.array(z.object({ text: z.string(), owner: z.string().optional() })).optional(),
  }),
  chapters: z.array(z.object({ title: z.string(), startMs: z.number() })).optional(),
  glossary: z.array(z.object({ term: z.string(), translation: z.string() })).optional(),
});
