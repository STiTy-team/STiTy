# T15a Segment pairwise accuracy vs the judge

1040 rows, 20 instances. Judge = gpt-6-sol. Pairs with identical hypotheses and judge ties are skipped; a metric tie counts 0.5. CIs resample instances.

| judge_target | n_pairs | comet | comet_ci | doc_comet_w2 | doc_comet_w2_ci | doc_comet_w5 | doc_comet_w5_ci | xcomet | xcomet_ci | metricx_neg | metricx_neg_ci | chrf_pp | chrf_pp_ci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| judge_check_pass_rate | 11552 | 0.794 | [0.708, 0.867] | 0.804 | [0.721, 0.878] | 0.806 | [0.731, 0.873] | 0.841 | [0.763, 0.907] | 0.880 | [0.810, 0.937] | 0.749 | [0.653, 0.831] |
| judge_mean | 21009 | 0.757 | [0.706, 0.805] | 0.757 | [0.699, 0.811] | 0.755 | [0.698, 0.809] | 0.766 | [0.722, 0.804] | 0.792 | [0.757, 0.828] | 0.704 | [0.640, 0.763] |
| judge_contextual_correctness | 15446 | 0.763 | [0.697, 0.820] | 0.767 | [0.700, 0.828] | 0.760 | [0.698, 0.816] | 0.807 | [0.743, 0.866] | 0.829 | [0.782, 0.878] | 0.722 | [0.638, 0.792] |
| judge_meaning_preservation | 15411 | 0.799 | [0.728, 0.865] | 0.810 | [0.742, 0.870] | 0.818 | [0.758, 0.875] | 0.797 | [0.745, 0.846] | 0.850 | [0.812, 0.892] | 0.769 | [0.699, 0.831] |
| judge_naturalness | 11692 | 0.710 | [0.612, 0.796] | 0.723 | [0.649, 0.792] | 0.714 | [0.637, 0.787] | 0.671 | [0.581, 0.765] | 0.634 | [0.522, 0.733] | 0.690 | [0.610, 0.761] |
