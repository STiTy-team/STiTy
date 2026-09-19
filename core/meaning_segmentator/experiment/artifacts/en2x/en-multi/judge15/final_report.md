# en2x/en-multi/judge15 — 최종

채택본 test H_set **0.5612** / 오라클 0.6596 / v0 0.5713

v0 대비 Δ -0.0101 [-0.0208, +0.0013] (짝 1787 / 문장 200)

## 지연 구간별 test H_set

| 구간 | prompt | v0 | oracle |
|---|---|---|---|
| ≤3 | 0.4452 | 0.4462 | 0.5710 |
| ≤5 | 0.5737 | 0.5951 | 0.6745 |
| ≤7 | 0.6861 | 0.6993 | 0.7497 |
| ≤10 | 0.7416 | 0.7568 | 0.7969 |
| ≤99 | 0.7897 | 0.7958 | 0.8202 |

포맷 통과율 1.0

## 이터레이션

| 이터 | 결과 | Δ (dev-A) | 비고 |
|---|---|---|---|
| 1 | 선별 탈락 (후보 0) | — |  |
| 1 | 선별 탈락 (후보 1) | — |  |
| 1 | 선별 탈락 (후보 2) | — |  |
| 1 | 기각 | -0.0071 [-0.0209, +0.0062] | The C4 replacement made the model avoid cutting between a noun (or its specifier |
| 2 | 선별 탈락 (후보 0) | — |  |
| 2 | 선별 탈락 (후보 1) | — |  |
| 2 | 선별 탈락 (후보 2) | — |  |
| 2 | 기각 | -0.0100 [-0.0233, +0.0026] | Relaxing C4 to allow noun+modifier cuts when forbidding them would force very-sh |
| 3 | 선별 탈락 (후보 0) | — |  |
| 3 | 선별 탈락 (후보 1) | — |  |
| 3 | 선별 탈락 (후보 2) | — |  |
| 3 | 채택 | +0.0110 [+0.0010, +0.0214] | Adding the numeric-preservation principle made the model prefer cuts that keep a |
| 4 | 선별 탈락 (후보 1) | — |  |
| 4 | 선별 탈락 (후보 2) | — |  |
| 4 | 선별 탈락 (후보 3) | — |  |
| 4 | 기각 | -0.0031 [-0.0134, +0.0078] | The C1 change (prioritizing boundaries whose remainder would contradict the pref |
| 5 | 선별 탈락 (후보 0) | — |  |
| 5 | 선별 탈락 (후보 1) | — |  |
| 5 | 선별 탈락 (후보 2) | — |  |
| 5 | 기각 | -0.0012 [-0.0134, +0.0112] | Replacing C8 made the model more willing to isolate sentence‑initial function wo |
| 6 | 선별 탈락 (후보 0) | — |  |
| 6 | 선별 탈락 (후보 1) | — |  |
| 6 | 선별 탈락 (후보 2) | — |  |
| 6 | 기각 | -0.0052 [-0.0173, +0.0071] | Inserting the predicate–argument guideline caused the scorer to prefer keeping v |
| 7 | 선별 탈락 (후보 2) | — |  |
| 7 | 선별 탈락 (후보 3) | — |  |
| 7 | 기각 | -0.0127 [-0.0250, -0.0004] | Softening C7 made the instruction ambiguous in short‑latency contexts: the model |

비용 $42.53
