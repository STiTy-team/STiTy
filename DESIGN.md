---
version: alpha
name: STiTy
description: Real-time speech translation between two people.
colors:
  primary: "{colors.blue-600}"
  fill-brand: "{colors.blue-600}"
  fill-brand-pressed: "{colors.blue-700}"
  fill-brand-weak: "{colors.blue-50}"
  fill-primary: "{colors.grey-900}"
  fill-secondary: "{colors.grey-100}"
  fill-weak: "{colors.grey-50}"
  fill-live: "{colors.olive-500}"
  fill-danger: "{colors.red-600}"
  text-primary: "{colors.grey-900}"
  text-secondary: "{colors.grey-700}"
  text-tertiary: "{colors.grey-600}"
  text-placeholder: "{colors.grey-400}"
  text-alt: "{colors.white}"
  text-brand: "{colors.blue-600}"
  text-live: "{colors.olive-700}"
  text-danger: "{colors.red-600}"
  border-primary: "{colors.blue-500}"
  border-secondary: "{colors.grey-200}"
  border-strong: "{colors.grey-400}"
  surface: "{colors.white}"
  surface-sunken: "{colors.grey-50}"
  overlay-scrim: "#00000080"
  overlay-press: "#00000042"

  # STiTy brand (from the logo)
  blue-50: "#F0F3FA"
  blue-500: "#6080C8" # logo "ST". Marks, icons, focus rings only: 3.9:1 on white, too light for text
  blue-600: "#4268BD" # every filled brand surface and brand text: 5.3:1 against white
  blue-700: "#37579F" # pressed
  olive-50: "#F4F7EA"
  olive-500: "#7A9030" # logo "T". Live / listening signal
  olive-700: "#617326" # olive text, 5.3:1 on white
  sky-300: "#9BB4D4" # logo "i". Mascot and illustration only
  cream-200: "#E8E0A0" # logo "y". Mascot and illustration only

  # Neutrals (TDS cool greys)
  white: "#FFFFFF"
  grey-50: "#F9FAFB"
  grey-100: "#F2F4F6"
  grey-200: "#E5E8EB"
  grey-300: "#D1D6DB"
  grey-400: "#B0B8C1"
  grey-500: "#8B95A1"
  grey-600: "#6B7684"
  grey-700: "#4E5968"
  grey-900: "#191F28"
  red-600: "#D91122"

  # Language hues: small marks only (dot, avatar, tag). Never a bubble fill or a page colour.
  lang-en: "{colors.blue-500}"
  lang-ko: "{colors.olive-500}"
  lang-ja: "#C87060"
  lang-zh: "#9060C8"
  lang-es: "#C8A030"
  lang-fr: "#308898"
  lang-id: "#30A070"
  lang-vi: "#B85050"
  lang-th: "#5080C0"
  lang-de: "#C09050"
  lang-ar: "#C56BA8"
  lang-other: "{colors.grey-500}"

  # Data colours: bench charts and timelines only
  data-seg: "{colors.lang-en}"
  data-vad: "{colors.lang-ko}"
  data-dot: "{colors.lang-es}"
  data-always: "{colors.lang-zh}"
  data-finish: "{colors.grey-500}"
  data-alarm: "{colors.lang-vi}"
  data-audio: "{colors.grey-300}"
  data-lane-0: "#4C9A5A"
  data-lane-1: "#D08A2A"
  data-lane-2: "#C0554F"
  data-lane-3: "#7B60C0"
  data-lane-4: "#3F8A9C"
  data-lane-5: "#A0662A"
  data-lane-6: "#6D7FA8"
  data-lane-7: "#8A8F84"

  # Dark theme (bench only; the mobile app is light only)
  dark-surface: "#202027"
  dark-surface-raised: "#2C2C35"
  dark-border: "#3C3C47"
  dark-text-primary: "#FFFFFF"
  dark-text-secondary: "#C3C3C6"
  dark-text-tertiary: "#9E9EA4"
  dark-text-brand: "#85A3DD"
  dark-text-live: "#9CB544"
  dark-text-danger: "#FF6B78"
typography:
  display:
    fontFamily: Pretendard Variable
    fontSize: 40px
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: -0.02em
  h1:
    fontFamily: Pretendard Variable
    fontSize: 28px
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: -0.02em
  h2:
    fontFamily: Pretendard Variable
    fontSize: 22px
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: -0.015em
  title:
    fontFamily: Pretendard Variable
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.45
    letterSpacing: -0.01em
  body-1:
    fontFamily: Pretendard Variable
    fontSize: 17px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: -0.005em
  body-2:
    fontFamily: Pretendard Variable
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: -0.005em
  body-3:
    fontFamily: Pretendard Variable
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.5
  label-l:
    fontFamily: Pretendard Variable
    fontSize: 17px
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: -0.005em
  label-m:
    fontFamily: Pretendard Variable
    fontSize: 15px
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: -0.005em
  label-s:
    fontFamily: Pretendard Variable
    fontSize: 13px
    fontWeight: 600
    lineHeight: 1.25
  caption:
    fontFamily: Pretendard Variable
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.4
  caption-s:
    fontFamily: Pretendard Variable
    fontSize: 11px
    fontWeight: 500
    lineHeight: 1.4
  data:
    fontFamily: Pretendard Variable
    fontSize: 13px
    fontWeight: 500
    lineHeight: 1.4
    fontFeature: '"tnum" 1'
  mono:
    fontFamily: IBM Plex Mono
    fontSize: 12px
    fontWeight: 400
    lineHeight: 1.5
spacing:
  space-1: 4px
  space-2: 8px
  space-3: 12px
  space-4: 16px
  space-5: 20px
  space-6: 24px
  space-8: 32px
  space-10: 40px
  space-12: 48px
  screen-margin: 20px
  touch-min: 44px
rounded:
  xs: 4px # small badges, bench tags
  sm: 8px # inline tags, bench inputs
  md: 12px # text fields, segmented control track
  lg: 14px # L buttons, chat bubbles
  xl: 16px # XL buttons, cards, language cards
  2xl: 20px # sheets, dialogs
  full: 999px # chips, pills, avatars, dots
elevation:
  shadow-1: 0 1px 2px #0F172A0A, 0 1px 1px #0F172A0A
  shadow-2: 0 4px 12px #0F172A0F, 0 1px 2px #0F172A0A
  shadow-3: 0 12px 32px #0F172A1A, 0 2px 6px #0F172A0F
components:
  button-primary:
    backgroundColor: "{colors.fill-brand}"
    textColor: "{colors.text-alt}"
    typography: "{typography.label-l}"
    rounded: "{rounded.xl}"
    height: 56px
  button-primary-pressed:
    backgroundColor: "{colors.fill-brand-pressed}"
    textColor: "{colors.text-alt}"
  button-secondary:
    backgroundColor: "{colors.fill-secondary}"
    textColor: "{colors.text-primary}"
    typography: "{typography.label-m}"
    rounded: "{rounded.md}"
    height: 40px
  button-danger:
    backgroundColor: "{colors.fill-danger}"
    textColor: "{colors.text-alt}"
    typography: "{typography.label-l}"
    rounded: "{rounded.xl}"
  bubble-mine:
    backgroundColor: "{colors.fill-brand}"
    textColor: "{colors.text-alt}"
    typography: "{typography.body-2}"
    rounded: "{rounded.lg}"
    padding: 12px
  bubble-peer:
    backgroundColor: "{colors.fill-secondary}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body-2}"
    rounded: "{rounded.lg}"
    padding: 12px
  bubble-translation-peer:
    backgroundColor: "{colors.fill-secondary}"
    textColor: "{colors.text-secondary}"
    typography: "{typography.body-3}"
  bubble-partial:
    backgroundColor: "{colors.fill-weak}"
    textColor: "{colors.text-secondary}"
  chip:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-secondary}"
    typography: "{typography.label-s}"
    rounded: "{rounded.full}"
    height: 28px
  chip-brand:
    backgroundColor: "{colors.fill-brand-weak}"
    textColor: "{colors.text-brand}"
  chip-live:
    backgroundColor: "{colors.olive-50}"
    textColor: "{colors.text-live}"
  live-dot:
    backgroundColor: "{colors.fill-live}"
    rounded: "{rounded.full}"
    size: 8px
  language-dot:
    backgroundColor: "{colors.lang-other}"
    rounded: "{rounded.full}"
    size: 8px
  segmented-control:
    backgroundColor: "{colors.fill-secondary}"
    textColor: "{colors.text-secondary}"
    typography: "{typography.label-s}"
    rounded: "{rounded.md}"
  segmented-control-selected:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
  error-banner:
    backgroundColor: "{colors.fill-weak}"
    textColor: "{colors.text-danger}"
    typography: "{typography.body-3}"
    rounded: "{rounded.md}"
  toast:
    backgroundColor: "{colors.fill-primary}"
    textColor: "{colors.text-alt}"
    typography: "{typography.label-m}"
    rounded: "{rounded.lg}"
  stage-caption:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    typography: "{typography.display}"
  bench-panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body-3}"
    rounded: "{rounded.xl}"
  bench-page:
    backgroundColor: "{colors.surface-sunken}"
    textColor: "{colors.text-secondary}"
  bench-panel-dark:
    backgroundColor: "{colors.dark-surface-raised}"
    textColor: "{colors.dark-text-primary}"
  bench-page-dark:
    backgroundColor: "{colors.dark-surface}"
    textColor: "{colors.dark-text-secondary}"
---

# STiTy

## Overview

STiTy puts live speech translation between two people who don't share a
language. Each person speaks in their own language, and the other reads the
translation a moment later.

The reference point is **a quick bilingual friend sitting between two people at
a café table**. The friend speaks up only when needed and otherwise stays out of
the way. The two people should look at each other, not at the screen. So the
conversation text is the only thing that stands out, and everything around it
stays quiet.

The visual language follows the **Toss Design System (TDS)**: a calm white
canvas, cool blue-tinted greys, one saturated accent per screen, generous
rounded corners, flat surfaces, thin 1px borders, and short, mechanical motion.
We take TDS's look and its rules for copy. We do not take Toss's finance
features or screens.

There are three surfaces, and they share one system:

- **Mobile app** (`STiTy-Mobile/`): the product. Light theme only. Everything in
  this file applies to it first.
- **Demo stage** (`demo-web/partial_demo/web/show.html`): captions on a
  projector for an audience. It uses the same tokens at a larger type size.
- **Bench** (`bench/replay/static/`): an internal dashboard for comparing
  pipelines. It is denser, adds the data colours and has a dark theme. It is
  still Toss-calm: no uppercase labels and no heavy colour.

## Colors

The palette is **Toss greys plus the STiTy logo**. The logo has four letters in
four colours. The UI keeps only one of them as its accent, and one as a signal.

- **Brand blue** {colors.blue-600} is the single accent. It fills the one
  primary action on a screen and the user's own chat bubbles. The logo's exact
  blue {colors.blue-500} is too light for white text (3.9:1), so it only appears
  in the logo, in icons, in focus rings and as the English language mark. It
  never sits behind text.
- **Olive** {colors.olive-500}, from the logo's "T", means **live**: the
  microphone is open and the server is listening. It appears as a small dot or
  chip, never as a button or bubble. Olive text uses {colors.olive-700}.
- **Sky** {colors.sky-300} and **cream** {colors.cream-200}, the logo's "i" and
  "y", belong to the mascot and illustrations only. They play the same role as
  Toss's yellow and brown illustration colours.
- **Greys** {colors.grey-50} to {colors.grey-900} are the TDS cool greys and
  cover every other surface. Primary text is {colors.grey-900}, never pure
  black.
- **Red** {colors.red-600} is for errors and destructive actions only.

**Language hues** (`lang-*`) tell the listener which language a line is in. They
show up only as small marks: the 8px language dot, the avatar and the language
tag. A bubble is never filled with a language hue. Your own bubble is brand blue
and your partner's is grey, whatever languages they speak. A third language in
the room gets a grey bubble with its hue on the dot. English and Korean reuse
the logo blue and olive, so the two most common languages are also the brand
colours.

**Data colours** (`data-*`) exist only inside bench charts and timelines. The
four commit reasons (`seg`, `vad`, `dot`, `always`, the `commitReason` values in
`docs/WEBSOCKET_PROTOCOL.md`) reuse language hues, so the whole project draws
from one set of hues. The `data-lane-*` colours tell pipelines apart in the
session replay.

**Dark theme** exists only on bench. It uses the TDS dark greys, which are
nearly colourless. It follows the OS setting, and `[data-theme]` can override
it. In dark mode, brand and live text switch to {colors.dark-text-brand} and
{colors.dark-text-live}. Filled brand buttons keep {colors.blue-600} with white
text in both themes.

## Typography

**Pretendard** is the one typeface. It is the free Korean and Latin font closest
to Toss Product Sans. The UI is available in Korean, English, Japanese, Chinese
and Spanish, and the conversation can be in any of the 11 languages above, so
the stack falls back by script:

```
"Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
"Noto Sans JP", "Noto Sans SC", "Noto Sans Thai", "Noto Sans Arabic", Roboto, sans-serif
```

- **Conversation text** is {typography.body-2} (15px, line height 1.5), which
  suits Korean. The original line is regular weight in the primary text colour.
  The translation below it is {typography.body-3} in the secondary text colour.
  The translation must stay readable, not faded.
- **Titles** are bold with slightly tight letter spacing ({typography.h1},
  {typography.h2}). Size steps stay modest.
- **Buttons read like sentences**: {typography.label-l} for the large bottom
  action and {typography.label-m} for the smaller ones.
- **Numbers** (latency, WER, BLEU, cost, timestamps) use {typography.data},
  which turns on tabular figures so columns line up. This replaces the old habit
  of setting every number in monospace.
- **Monospace** ({typography.mono}, IBM Plex Mono) is reserved for things that
  really are code: run IDs, config keys, file paths and model names on bench.
- **Demo stage** captions use {typography.display}, scaled to the projector.
  Weight and colour rules don't change.

## Layout

Mobile first, single column. The screen margin is {spacing.screen-margin}.
Spacing follows a 4px scale ({spacing.space-1} to {spacing.space-12}):

- {spacing.space-2} between tightly linked items, such as a label and its value.
- {spacing.space-4} between list rows and between chat bubbles from different
  speakers.
- {spacing.space-6} between sections.

Spacing is not the same everywhere. Related things sit close together and
unrelated groups sit far apart, so the gaps themselves show the grouping. A page
where every section has the same padding looks like a template.

### Where text goes

- **Left-align everything people read**: screen titles, labels, bubble text,
  captions and table text. Centre text only for a one-line empty state ("말씀해
  주세요") or a single-word button label.
- **One screen, one job**, following Toss's "one thing per page". A screen asks
  one question in its title and has one main action. The title is
  {typography.h1}, placed top left, and phrased as the question the user is
  answering, for example 누구와 대화하나요?
- **Labels sit above their values**, not beside them, and use
  {typography.caption} in {colors.text-tertiary}. There is no small "eyebrow"
  line above section titles.
- **Numbers in a column are right-aligned** and use {typography.data}, so digits
  line up.
- **Keep lines short enough to read.** Prose should stay under about 60 Latin
  characters per line (about 35 Korean characters). On a wide screen, cap the
  width rather than stretching text across it.
- **A label that can be tapped stays on one line.** If it wraps, shorten it.

### Mobile app

The app has two states, and each one owns the whole screen.

1. **Setup.** The top bar has the logo on the left and settings on the right,
   56px tall. Below it are the title question, the two language cards, the mode
   selector, and the bottom action pinned to the bottom edge. Nothing else goes
   on this screen.
2. **Conversation.** The setup panel collapses into one row of session chips (my
   language → partner's language · mode) under the top bar. The conversation
   takes the rest of the height. The stop button is pinned to the bottom.

The bottom action is full width, 56px tall, within thumb reach, and inside the
safe area. The screen margin is {spacing.screen-margin} on both sides.

### Conversation

- **Mine on the right, my partner's on the left.** Text inside a bubble is
  always left-aligned, even in my right-hand bubble. Arabic text is
  right-aligned inside its bubble.
- **Original first, translation second**, in the same bubble. The translation is
  a smaller size, not a fainter colour.
- **The newest line is at the bottom.** The view scrolls itself only while the
  user is already at the bottom. If they have scrolled up to reread, it stays
  put and a "새 메시지" pill appears above the bottom action.
- **`partial` text only grows at the end.** Words already on screen never jump,
  reflow or re-centre while the rest of the line arrives. When the line becomes
  `final`, only its colour changes. Its position stays the same.
- Bubbles from the same speaker in a row sit {spacing.space-1} apart. A change
  of speaker gets {spacing.space-4}.

### Demo stage

Captions follow live-subtitle practice (BBC). Text is left-aligned, not centred.
There are at most two lines of about 37 Latin characters, or about 20 CJK
characters. New words are added at the end of the bottom line. When the area is
full, the oldest complete line scrolls away in one step, like a teleprompter. A
line never slides sideways.

## Elevation & Depth

Flat by default. Depth comes from **tonal layers**: white content on a
{colors.grey-50} page, separated by 1px {colors.grey-200} borders. Shadows
appear only on things that float: {elevation.shadow-1} for the selected segment
and menus, {elevation.shadow-2} for tooltips, and {elevation.shadow-3} for
dialogs and sheets. Shadows are a faint navy, never grey or black. A pressed
state is a dark overlay ({colors.overlay-press}) on top of the fill, not a
shadow.

## Motion

```yaml
motion:
  ease: cubic-bezier(0.22, 0.61, 0.36, 1)
  dur-fast: 120ms # button press
  dur-base: 200ms # toggle, a new bubble rising in, partial text becoming final
  dur-slow: 320ms # sheets and dialogs
```

Quick and mechanical, with no bounce, overshoot or parallax. Nothing lasts
longer than 320ms. Two animations are specific to STiTy. A new bubble slides up
from below in 200ms. A `partial` line (still being recognised) sits dimmed, then
fades to full colour in 200ms when it becomes `final`. The live dot pulses
slowly while listening. That pulse is the only looping animation. Loading uses
three dots, not skeleton shimmer. When `prefers-reduced-motion` is set, all
durations become 0 and the pulse stops.

## Shapes

Generously rounded, never cute. The corner scale is {rounded.xs} to
{rounded.2xl}, plus {rounded.full}. Bigger elements get bigger corners: badges
{rounded.xs}, text fields {rounded.md}, bubbles {rounded.lg}, buttons and cards
{rounded.xl}, sheets {rounded.2xl}. Chips, avatars and dots are {rounded.full}.
A chat bubble has a 4px corner on the speaker's side at the bottom (bottom right
for mine, bottom left for my partner's), which acts as its tail.

Borders are 1px {colors.grey-200}. A focused input gets a 1.5px
{colors.border-primary} border. Surfaces are flat, with no gradients. The one
exception is a white-to-transparent fade above the pinned bottom button, so
scrolling text doesn't collide with it.

Icons are line icons on a 24px grid with a 1.5px stroke, and they inherit
`currentColor`. The mascot (two round faces in {colors.sky-300} and
{colors.cream-200}) is the only illustration. It appears in the empty state and
the about screen, never inside a conversation.

## Components

- **Chat bubbles**: {components.bubble-mine} and {components.bubble-peer}.
  Maximum width is 72% of the screen. The original text is on top and the
  translation below. A `partial` bubble uses {components.bubble-partial} until
  it becomes final.
- **Language card** (setup): two cards side by side, "my language" and
  "partner's language", with a swap button between them. White cards with a 1px
  border, {rounded.xl}. The language name is {typography.title}, and a
  {components.language-dot} shows the language's hue. They are not filled with
  brand colour.
- **Mode selector**: {components.segmented-control}. The selected segment is
  white with {elevation.shadow-1} and primary text, as in TDS. It is not
  coloured.
- **Bottom action**: {components.button-primary}, full width and pinned. It is
  the one blue thing on the setup screen. While a session runs it becomes a stop
  button in {components.button-secondary} style, so the conversation keeps the
  attention.
- **Status**: {components.live-dot} in olive while listening, grey while idle,
  red on error. {components.chip-live} carries a short status word next to it.
- **Session chips**: {components.chip}, with {components.chip-brand} for the
  current user's language.
- **Errors**: {components.error-banner}. Short, and says what happens next.
- **Toast**: {components.toast} for brief confirmations ("Data deleted").
- **Demo stage**: {components.stage-caption}. One caption lane per language,
  each labelled with its language dot.
- **Bench**: {components.bench-page} with {components.bench-panel}, and
  {components.bench-page-dark} / {components.bench-panel-dark} in dark mode.
  Tables use 1px row dividers and {typography.data} numbers. Charts have no
  heavy axis lines. They draw in `data-*` colours on a calm surface.

## Screens

### Bench dashboard

The dashboard answers one question: **which pipeline wins on this dataset, and
why?** Design the page from that question, not from a dashboard template.

- **The main element is one table**: runs as rows, metrics as columns, placed
  top left. There is no row of KPI tiles above it. The table already is the
  summary.
- **At most seven metric columns are visible.** Others hide behind a "더 보기"
  toggle. People can only compare a handful of numbers at once.
- **Every column header gives its unit and which way is better**, for example
  `WER ↓` or `BLEU ↑`. Each cell shows its change from the baseline run in
  {typography.caption} under the value. A better value uses {colors.text-live}
  and a worse one uses {colors.text-danger}. The value itself stays in
  {colors.text-primary}.
- **The chart comes after the table** and explains it. It never repeats the
  table.
- **When the data comes from** (dataset, run date) is always visible next to the
  page title.
- **Missing values show "—".** Never show 0 for missing data, and never show a
  made-up placeholder number.
- **One level of containment.** A panel sits on the page, and inside the panel
  only dividers and whitespace separate things. There are no cards inside cards.

### Session replay

A header block, then the timeline, then the transcript.

- **The header** has the run name, the dataset item and at most four numbers.
  The number that matters most for the item (usually latency or WER) is
  {typography.h2}. The rest are {typography.data}. Their sizes differ on
  purpose, so the tiles are not a uniform grid.
- **The timeline** reads left to right in time. Each pipeline has a lane in its
  `data-lane-*` colour. Commit reasons are marks in their `data-*` colour.
- **The transcript** is left-aligned text with timestamps in a right-aligned
  column.

### Charts

- Label lines directly at their end instead of using a legend box.
- Gridlines are thin {colors.grey-200} lines, or there are none. Axis lines are
  not drawn.
- The pipeline being looked at is {colors.data-seg}. Pipelines being compared
  are {colors.grey-400}. Use the full `data-lane-*` set only when every pipeline
  matters equally.
- No 3D, no donut charts for comparisons, and no gradient fills under lines.

## Voice

Copy follows the Toss voice.

- **Korean uses 해요체 everywhere**, including errors. Use "서버가 꽉 찼어요.
  자리가 나면 바로 시작해요.", not "…찼습니다. …시작됩니다."
- **No exclamation marks, no hype.** Use "지금 시작할 수 있어요", not "지금
  시작하세요!"
- **Buttons say what will happen**: 시작하기, 대화 끝내기, 데이터 지우기. Avoid
  vague labels such as "확인" when a specific verb fits.
- **Errors guide the user onward.** They say what to do next, not only what went
  wrong.
- **English is plain and in sentence case**, with no ALL CAPS labels. Teammates
  and many users are not native English speakers.
- The other UI languages (ja, zh, es) use the same polite-casual register as
  Korean.

## Do's and Don'ts

- **Do** keep one blue element per screen. If two things want blue, one of them
  is secondary.
- **Do** colour chat bubbles by speaker (mine is blue, my partner's is grey),
  never by language.
- **Do** use {colors.blue-600} behind white text. {colors.blue-500} is for marks
  only.
- **Do** keep olive for "live". If olive shows up on something that isn't
  listening, that's a bug.
- **Do** use tabular figures for every number that updates or sits in a column.
- **Do** keep touch targets at least {spacing.touch-min}.
- **Don't** fill surfaces with language hues. They are 8px dots, avatars and
  tags.
- **Don't** add a third accent. The yellow mode colour (`MODE_ACTIVE_COLOR`) and
  the Tailwind red and green (`#ef4444`, `#22c55e`) in `HomeScreen.tsx` get
  replaced by tokens.
- **Don't** use uppercase letter-spaced labels, including on bench.
- **Don't** put the mascot, emoji or gradients inside the conversation.
- **Don't** add dark mode to the mobile app until it has its own token table.
  Bench's dark theme doesn't carry over automatically.
- **Don't** copy Toss's product: no finance flows, no Toss logo, no Toss Product
  Sans.

### Template giveaways

These patterns show up in almost every generated UI. They make a screen look
like it could belong to any product. Before adding a layout, ask: **would this
look the same in any other app?** If yes, it came from a template and not from
STiTy.

- **Colour and effects.** No gradients on backgrounds or text. No purple. No
  glass or blur panels, glows or floating blobs. No pure black on pure white.
- **Layout.** Don't centre everything. No full-screen hero. No row of three
  identical icon cards. No cards inside cards. No thick coloured stripe down one
  side of a card. No identical padding on every section.
- **Type.** No small uppercase "eyebrow" line above headings. No single
  highlighted or coloured word in a title. No italic headings. No decorative "01
  / 02 / 03" numbering.
- **Icons.** One line-icon set only. No emoji used as icons. No ✨ or robot
  icons for anything "AI". No icon tile above every heading.
- **Motion.** No fade-up on scroll. No hover scale-up. No bounce. Never
  `transition-all`; name the property that changes. A spinner appears only after
  150ms, so fast actions don't flash one.
- **Feedback.** Success is silent. The result on screen is the confirmation, so
  there are no "🎉 완료!" toasts. Toasts are for failures and for undo. Don't
  ask "정말 삭제할까요?" for something the user can undo. Offer undo instead.
- **Content.** Never invent numbers, users or quotes to fill space. No hype
  words: seamless, powerful, next-generation, 혁신적인, 차세대.

## Known Gaps

- The code doesn't use these tokens yet. `HomeScreen.tsx` repeats hex values
  inline, and `show.html` has its own `LANG_COLORS` with different values. Both
  need to move onto this file's tokens.
- `lang-th` and `lang-en` are close in hue. They are hard to tell apart when
  Thai and English are in the same room.
- The language hues are marks, not text. Several of them fall below 4.5:1 on
  white and must not be used for text.
- Some current layouts break the rules in Layout and Screens. The app's header
  centres the logo (`headerLogoWrap` in `HomeScreen.tsx`). The demo stage's
  caption lanes need checking against the teleprompter rule.
