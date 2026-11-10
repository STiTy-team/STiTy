/** Letters and digits that can't be misread when a code is read out across a room. */
const CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';

export const ROOM_CODE_LENGTH = 6;

/**
 * Every conversation can be shared by the code the server gives it; `code` is null until then, and
 * stays null for a saved conversation, whose id belongs to its record. `joining` makes the server refuse a code that
 * isn't open, so a mistyped code fails instead of opening an empty room. It clears once connected,
 * so a resume reopens the room even if everyone else has left. `shared` is set by inviting or
 * joining, and from then on the phone behaves as one of several listening in a room.
 */
export type Room = { code: string | null; joining: boolean; shared: boolean };

export type RoomParticipant = { id: string; number: number; canSpeak: boolean };

/** Who is in the room, from the server. There is one host; the room ends when they leave. */
export type Roster = { hostId: string | null; participants: RoomParticipant[] };

export const newRoom = (): Room => ({ code: null, joining: false, shared: false });

export const joinRoom = (code: string): Room => ({ code, joining: true, shared: true });

/** What the code box keeps of what was typed or pasted. */
export const normalizeRoomCode = (input: string): string =>
  input
    .toUpperCase()
    .split('')
    .filter((character) => CODE_ALPHABET.includes(character))
    .join('')
    .slice(0, ROOM_CODE_LENGTH);

export const isRoomCode = (code: string): boolean =>
  code.length === ROOM_CODE_LENGTH && normalizeRoomCode(code) === code;
