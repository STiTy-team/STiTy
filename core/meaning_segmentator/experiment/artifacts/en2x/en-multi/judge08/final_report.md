# en2x/en-multi/judge08 — 최종

채택본 test H_set **0.4919** / 오라클 0.5595 (v0 그대로 — 채택된 개정 없음)

## 지연 구간별 test H_set

| 구간 | prompt | oracle |
|---|---|---|
| ≤3 | 0.3644 | 0.4454 |
| ≤5 | 0.4800 | 0.5583 |
| ≤7 | 0.6014 | 0.6472 |
| ≤10 | 0.6864 | 0.7184 |
| ≤99 | 0.7415 | 0.7799 |

포맷 통과율 1.0

## 이터레이션

| 이터 | 결과 | Δ (dev-A) | 비고 |
|---|---|---|---|
| 1 | 반려: 길이 초과: 10831 > 10580 | — |  |
| 2 | 기각 | -0.0103 [-0.0206, +0.0006] | The new items pushed the model to cut earlier so that finite verbs, short disamb |
| 3 | 선별 탈락 (후보 1) | -0.0160 [-0.0340, +0.0006] |  |
| 3 | 선별 탈락 (후보 2) | -0.0199 [-0.0381, -0.0031] |  |
| 3 | 기각 | -0.0010 [-0.0124, +0.0113] | The new checks made the model move many cuts earlier in the sentence (hurt most  |
| 4 | 선별 탈락 (후보 1) | -0.0225 [-0.0397, -0.0075] |  |
| 4 | 선별 탈락 (후보 2) | -0.0184 [-0.0340, -0.0028] |  |
| 4 | 기각 | -0.0094 [-0.0249, +0.0067] | Replacing C8 introduced explicit checks that penalize isolating tiny right‑hand  |
| 5 | 선별 탈락 (후보 0) | -0.0104 [-0.0316, +0.0084] |  |
| 5 | 선별 탈락 (후보 2) | -0.0345 [-0.0560, -0.0141] |  |
| 5 | 기각 | -0.0077 [-0.0305, +0.0150] | The two inserted checks made the scorer prefer attaching decisive predicates/com |

비용 $27.86
