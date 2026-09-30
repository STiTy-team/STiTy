# 분절 조건별 참조 기반 BLEU — en2x/covost2/full_qwenseg

- 분절: `none_use_manifest` / 번역기 `local:google/madlad400-3b-mt:zh:ctx=False` / 부트스트랩 0회
- **언어 간 절대 BLEU 비교 금지** — 토크나이저가 다르다. 언어를 가로지르는 판독은 `retention`(자기 무분절 대비)과 `chrF2` 로 한다.
- 소스 en → 타깃 zh. 이 타깃들은 프롬프트 최적화 시 목적함수에 포함됐던 언어다 (미출현 타깃 아님).
- `k`(조각 수)는 자동·기계 분절에서는 세 타깃이 **같은 분절**을 쓰므로 동일하다. 단 `causal_align`·`mu_prefix` 는 타깃별 자원(정렬 대상·NMT)에 의존하므로 타깃마다 다르다. `laal_words` 는 정의상 타깃 길이가 들어가고 ja·zh 는 타깃 단위가 문자라 값이 커진다 — **타깃 간 laal 비교 금지**, 같은 타깃 안에서 T 방향만 읽을 것.
- `laal_ms` 는 **강제정렬 실측**이다 (`--wordtimes qwen`). 독립 정렬기 둘(wav2vec2 CTC / Qwen3-ForcedAligner)이 조건 수준 LAAL 에서 22ms 이내로 일치한다. 구 방식(발화 내 균일속도 보간)은 지연을 64~131ms 과소평가했고 정책마다 편차가 있었다.
- `*_T*` 비교군 점은 정책의 경계 **부분집합**이다 (좌→우 탐욕). 제안 `auto_T*` 만 LLM 순위로 남길 경계를 고르므로, 순위 이득을 뺀 대조는 `auto_greedy_T*` 다.

## en→zh (n=15530, tok:zh, 번역기 `local:google/madlad400-3b-mt:zh:ctx=False`)

| 조건 | k | laal_ms ↓ | laal_words ↓ | BLEU ↑ | chrF2 | retention(BLEU) | Δ vs unseg [95% CI] |
|---|---|---|---|---|---|---|---|
| unsegmented | 1.00 | 2652 | 9.10 | 40.64 | 33.89 | 1.0000 | — |
| mechanical_8 | 6.85 | 1247 | 2.57 | 2.79 | 6.46 | 0.0685 | — |
| qwenseg_th0.05 | 2.35 | 1422 | 3.77 | 29.94 | 25.46 | 0.7368 | — |
| qwenseg_th0.1 | 2.06 | 1532 | 3.98 | 31.55 | 26.75 | 0.7763 | — |
| qwenseg_th0.2 | 1.75 | 1723 | 4.99 | 33.82 | 28.52 | 0.8322 | — |
| qwenseg_th0.3 | 1.48 | 2008 | 6.31 | 36.71 | 30.68 | 0.9033 | — |
| qwenseg_th0.5 | 1.19 | 2366 | 7.93 | 39.95 | 33.25 | 0.9829 | — |

## 언어 간 안정성 (retention = 조건 / 무분절)

| 조건 | zh BLEU | zh chrF2 | BLEU 폭 | chrF2 폭 |
|---|---|---|---|---|
| mechanical_8 | 0.0685 | 0.1906 | — | — |
| qwenseg_th0.05 | 0.7368 | 0.7513 | — | — |
| qwenseg_th0.1 | 0.7763 | 0.7891 | — | — |
| qwenseg_th0.2 | 0.8322 | 0.8416 | — | — |
| qwenseg_th0.3 | 0.9033 | 0.9054 | — | — |
| qwenseg_th0.5 | 0.9829 | 0.9811 | — | — |
