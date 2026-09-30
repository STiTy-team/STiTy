// Qwen3-ASR writes a `language <Name>` header at the start of each decode window, and
// again mid-text after a <SEG>. The server's partials can carry it (WORKLOG A9), and a
// header still being typed shows up as a lone `l`, `lang`… at the end. Same rule as the
// web demo's cleanAsr (demo-web/partial_demo/web/show.html).
const LANG_HEADER_RE = /\blanguage\s*:?\s*(chinese|english|cantonese|arabic|german|french|spanish|portuguese|indonesian|italian|korean|russian|thai|vietnamese|japanese|turkish|hindi|malay|dutch|swedish|danish|finnish|polish|czech|filipino|none)\b/gi;
const HEADER_STUB_RE = /(^|[.,!?;:。？！])\s*(?:l|la|lan|lang|langu|langua|languag|language)\s*$/;

export const cleanAsrText = (text: string): string =>
  String(text || '')
    .replace(/<asr_text>|<SEG>/g, ' ')
    .replace(LANG_HEADER_RE, ' ')
    .replace(HEADER_STUB_RE, '$1')
    .replace(/\s+/g, ' ')
    .trim();
