import AsyncStorage from '@react-native-async-storage/async-storage';

import { canRemoveLanguage, peerLanguageOf, usePreferencesStore } from './preferences-store';
import { LEGACY_PREFERENCES_KEY, persistedStorage } from './persisted-storage';

const store = usePreferencesStore;

beforeEach(() => store.setState({ languages: ['ko', 'en'], myLanguage: 'ko' }));

describe('conversation languages', () => {
  it('adds a language once', () => {
    store.getState().addLanguage('ja');
    store.getState().addLanguage('ja');
    expect(store.getState().languages).toEqual(['ko', 'en', 'ja']);
  });

  it('keeps at least two languages and never removes mine', () => {
    store.getState().removeLanguage('en');
    expect(store.getState().languages).toEqual(['ko', 'en']);

    store.getState().addLanguage('ja');
    store.getState().removeLanguage('ko');
    expect(store.getState().languages).toEqual(['ko', 'en', 'ja']);
    expect(canRemoveLanguage(store.getState(), 'ja')).toBe(true);

    store.getState().removeLanguage('en');
    expect(store.getState().languages).toEqual(['ko', 'ja']);
  });

  it('only accepts my language from the conversation', () => {
    store.getState().setMyLanguage('es');
    expect(store.getState().myLanguage).toBe('ko');
    store.getState().setMyLanguage('en');
    expect(store.getState().myLanguage).toBe('en');
  });

  it('pairs my language with the first other one for the session', () => {
    store.setState({ languages: ['ja', 'ko', 'en'], myLanguage: 'ko' });
    expect(peerLanguageOf(store.getState())).toBe('ja');
  });
});

describe('saved preferences', () => {
  const { migrate, merge } = store.persist.getOptions();

  it('turns a saved version 1 pair into a language list', () => {
    expect(migrate?.({ myLanguage: 'en', peerLanguage: 'ko' }, 1)).toEqual({
      myLanguage: 'en',
      languages: ['en', 'ko'],
    });
  });

  it('drops unsupported languages and refills to a usable conversation', () => {
    const restored = merge?.({ languages: ['ja', 'en'], myLanguage: 'ja' }, store.getState());
    expect(restored?.languages).toHaveLength(2);
    expect(restored?.languages).toContain('en');
    expect(restored?.languages).not.toContain('ja');
    expect(restored?.languages).toContain(restored?.myLanguage);
  });

  it('keeps only app-wide settings', () => {
    expect(
      Object.keys(store.persist.getOptions().partialize!(store.getState()) as object).sort(),
    ).toEqual(['languages', 'myLanguage', 'theme', 'uiLanguage']);
  });
});

describe('display settings', () => {
  const { partialize, merge } = store.persist.getOptions();

  it('saves and restores the app language and theme', () => {
    store.getState().setUiLanguage('ja');
    store.getState().setTheme('dark');
    const saved = partialize?.(store.getState());
    expect(saved).toMatchObject({ uiLanguage: 'ja', theme: 'dark' });

    const restored = merge?.(saved, { ...store.getState(), uiLanguage: 'system', theme: 'system' });
    expect(restored).toMatchObject({ uiLanguage: 'ja', theme: 'dark' });
  });
});

describe('restoring damaged preferences', () => {
  const { merge } = store.persist.getOptions();

  it('drops each invalid field to its default and keeps the valid ones', () => {
    const current = {
      ...store.getState(),
      uiLanguage: 'system' as const,
      theme: 'system' as const,
    };
    const restored = merge?.({ uiLanguage: 'klingon', theme: 'dark' }, current);
    expect(restored).toMatchObject({ uiLanguage: 'system', theme: 'dark' });
  });

  it('falls back to every default when the saved value is not an object', () => {
    const current = store.getState();
    const restored = merge?.('garbage', current);
    expect(restored).toMatchObject({ theme: current.theme });
    expect(restored?.languages.length).toBeGreaterThanOrEqual(2);
  });
});

describe('persistedStorage', () => {
  beforeEach(() => AsyncStorage.clear());

  it('starts a moved store from the old shared key until it writes its own', async () => {
    await AsyncStorage.setItem(
      LEGACY_PREFERENCES_KEY,
      JSON.stringify({ state: { vibrate: true, theme: 'dark' }, version: 2 }),
    );
    const storage = persistedStorage<{ vibrate: boolean }>(1)!;

    expect(await storage.getItem('stity.moved')).toEqual({
      state: { vibrate: true, theme: 'dark' },
      version: 1,
    });

    await storage.setItem('stity.moved', { state: { vibrate: false }, version: 1 });
    expect(await storage.getItem('stity.moved')).toEqual({ state: { vibrate: false }, version: 1 });
  });

  it('reads nothing when neither key holds readable JSON', async () => {
    await AsyncStorage.setItem(LEGACY_PREFERENCES_KEY, '{oops');
    expect(await persistedStorage(1)!.getItem('stity.moved')).toBeNull();
  });
});
