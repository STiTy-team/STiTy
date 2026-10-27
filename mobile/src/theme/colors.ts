export type ColorScheme = 'light' | 'dark';

const light = {
  background: '#FFFFFF',
  surface: '#F3F4F6',
  surfaceStrong: '#E9EBEE',
  /** Something lifted off a surface, like the selected segment. */
  raised: '#FFFFFF',
  border: '#E4E6EA',
  scrim: 'rgba(15,17,20,0.38)',

  text: '#0F1114',
  textSecondary: '#464C55',
  textTertiary: '#646B75',

  accent: '#3352C6',
  accentSoft: '#ECF0FC',
  onAccent: '#FFFFFF',

  recording: '#D93A40',
  recordingHalo: 'rgba(217,58,64,0.16)',
  onRecording: '#FFFFFF',

  danger: '#D93A40',
  dangerSoft: '#FDEEEF',
  warning: '#8A5A00',
  warningSoft: '#FFF5E0',
};

export type Colors = typeof light;

const dark: Colors = {
  background: '#0E1013',
  surface: '#181B1F',
  surfaceStrong: '#22262B',
  raised: '#30353C',
  border: '#272B31',
  scrim: 'rgba(0,0,0,0.55)',

  text: '#F1F2F4',
  textSecondary: '#B3B8C0',
  textTertiary: '#8E959F',

  accent: '#8299F2',
  accentSoft: '#1C2440',
  onAccent: '#0E1013',

  recording: '#F0575C',
  recordingHalo: 'rgba(240,87,92,0.18)',
  onRecording: '#FFFFFF',

  danger: '#F0575C',
  dangerSoft: '#2A1617',
  warning: '#F0C46B',
  warningSoft: '#2A2214',
};

export const themes: Record<ColorScheme, Colors> = { light, dark };
