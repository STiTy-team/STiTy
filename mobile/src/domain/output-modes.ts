/**
 * How translations are played back.
 * - speaker: one shared phone, everything through the loudspeaker.
 * - earphone: I wear an earphone and hear only the peer's speech, translated.
 * - both: I hear the peer through my earphone; the peer hears me through the speaker.
 */
export const OUTPUT_MODES = ['speaker', 'earphone', 'both'] as const;

export type OutputMode = (typeof OUTPUT_MODES)[number];
