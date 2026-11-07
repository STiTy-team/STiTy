import { z } from 'zod';

/** Wire contract with the ASR server — see docs/WEBSOCKET_PROTOCOL.md at the repository root. */

export const CONVERSATION_PATH = '/api/v1/ws';

export const AUDIO_FORMAT = {
  sampleRate: 16_000,
  chunkSamples: 1_600,
} as const;

const helloSchema = z.object({ type: z.literal('hello'), message: z.string().optional() });
const readySchema = z.object({
  type: z.literal('ready'),
  message: z.string().optional(),
  /** The record the server is writing for this session; null or absent when not recorded. */
  conversationId: z.string().nullable().default(null),
  /** Who this phone is in `room` messages. */
  participantId: z.string().nullable().default(null),
});

const roomSchema = z.object({
  type: z.literal('room'),
  hostId: z.string().nullable(),
  participants: z.array(z.object({ id: z.string(), number: z.number(), canSpeak: z.boolean() })),
});

const partialSchema = z.object({
  type: z.literal('partial'),
  text: z.string().default(''),
  language: z.string().default(''),
  seq: z.number().optional(),
});

const finalSchema = z.object({
  type: z.literal('final'),
  original: z.string().default(''),
  translation: z.string().default(''),
  language: z.string().default(''),
  /** Present when the server fans one segment out to several targets (one final per target). */
  targetLang: z.string().optional(),
  start: z.string().optional(),
  end: z.string().optional(),
  commitReason: z.string().optional(),
  segmentId: z.union([z.string(), z.number()]).transform(String).optional(),
  /** Present once the server separates speakers. */
  speaker: z.string().optional(),
});

const foldSchema = z.object({
  type: z.literal('fold'),
  segmentId: z.union([z.string(), z.number()]).transform(String),
  targetLang: z.string().default(''),
  original: z.string().default(''),
  /** What remains of the folded final; empty means the line disappears. */
  replacement: z.string().default(''),
});

const capacitySchema = z.object({
  type: z.literal('capacity'),
  active: z.number(),
  max: z.number().positive(),
});

/** The server refuses or abandons the session, e.g. a bad `start` or no free slot. */
const errorSchema = z.object({
  type: z.literal('error'),
  message: z.string().default(''),
  code: z.string().optional(),
});

const serverMessageSchema = z.discriminatedUnion('type', [
  helloSchema,
  readySchema,
  partialSchema,
  finalSchema,
  foldSchema,
  capacitySchema,
  roomSchema,
  errorSchema,
]);

export type ServerMessage = z.infer<typeof serverMessageSchema>;
export type PartialMessage = z.infer<typeof partialSchema>;
export type FinalMessage = z.infer<typeof finalSchema>;
export type FoldMessage = z.infer<typeof foldSchema>;
export type CapacityMessage = z.infer<typeof capacitySchema>;
export type RoomMessage = z.infer<typeof roomSchema>;
export type ErrorMessage = z.infer<typeof errorSchema>;

/** Returns null for binary frames, malformed JSON and message types the app does not use. */
export const parseServerMessage = (data: unknown): ServerMessage | null => {
  if (typeof data !== 'string') return null;
  let json: unknown;
  try {
    json = JSON.parse(data);
  } catch {
    return null;
  }
  const result = serverMessageSchema.safeParse(json);
  return result.success ? result.data : null;
};

export type DeviceInfo = {
  platform: string;
  version?: string;
  model?: string;
  headset?: boolean;
};

/** Fields after `translate` are optional additions; servers that don't know them ignore them. */
export type StartMessage = {
  type: 'start';
  lang: string;
  targetLang: string;
  translate: boolean;
  token?: string;
  save?: boolean;
  conversationId?: string;
  /** The server refuses a `conversationId` that isn't open instead of opening it. */
  join?: boolean;
  clientVersion?: string;
  /** Kept with server recordings, to tell later which phone a track came from. */
  device?: DeviceInfo;
  /** Development builds: ask the server to save this phone's audio and messages for debugging. */
  recording?: boolean;
};

export type ClientMessage =
  | StartMessage
  | { type: 'config'; lang?: string; targetLang?: string; recording?: boolean }
  | { type: 'stop' }
  | { type: 'ping' }
  | { type: 'set_mic'; participantId: string; canSpeak: boolean }
  | { type: 'set_all_mics'; canSpeak: boolean }
  | { type: 'remove_participant'; participantId: string }
  | { type: 'tts_log'; text: string; lang: string; start: string; end: string };

export const encodeClientMessage = (message: ClientMessage): string => JSON.stringify(message);
