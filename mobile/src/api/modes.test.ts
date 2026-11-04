import { MockedProductionError, mockedAreas, resolveApiModes } from './modes';

describe('resolveApiModes', () => {
  it('mocks everything except live outside production by default', () => {
    const modes = resolveApiModes(undefined, 'development');
    expect(modes.live).toBe('live');
    expect(mockedAreas(modes)).toEqual(['capabilities', 'auth', 'profile', 'conversations']);
  });

  it('reads an explicit list, all and none', () => {
    expect(mockedAreas(resolveApiModes('auth, live', 'preview'))).toEqual(['auth', 'live']);
    expect(mockedAreas(resolveApiModes('all', 'development'))).toHaveLength(5);
    expect(mockedAreas(resolveApiModes('none', 'development'))).toEqual([]);
  });

  it('runs production fully live and refuses any mock', () => {
    expect(mockedAreas(resolveApiModes(undefined, 'production'))).toEqual([]);
    expect(() => resolveApiModes('conversations', 'production')).toThrow(MockedProductionError);
    expect(() => resolveApiModes('all', 'production')).toThrow(MockedProductionError);
  });

  it('rejects unknown areas instead of silently going live', () => {
    expect(() => resolveApiModes('records', 'development')).toThrow(/records/);
  });
});
