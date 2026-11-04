import { router } from 'expo-router';

/** Back, or home when the screen was opened straight from a link and has nothing behind it. */
export const goBack = () => (router.canGoBack() ? router.back() : router.replace('/'));

/** Back to home from anywhere in a stack; replaces the screen when home isn't under it. */
export const goHome = () => router.dismissTo('/');
