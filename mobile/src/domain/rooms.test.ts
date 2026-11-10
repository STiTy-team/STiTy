import { isRoomCode, joinRoom, newRoom, normalizeRoomCode } from './rooms';

describe('rooms', () => {
  it('starts unshared, waiting for the server to give it a code', () => {
    expect(newRoom()).toEqual({ code: null, joining: false, shared: false });
  });

  it('marks a typed code as joining', () => {
    expect(joinRoom('ABC234')).toEqual({ code: 'ABC234', joining: true, shared: true });
  });

  it('keeps only code characters of what was typed', () => {
    expect(normalizeRoomCode(' ab-c 234xyz9 ')).toBe('ABC234');
  });

  it('refuses short codes and misreadable characters', () => {
    expect(isRoomCode('ABC23')).toBe(false);
    expect(isRoomCode('ABC0O1')).toBe(false);
  });
});
