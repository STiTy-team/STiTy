# T15b Pairwise accuracy differences

1040 rows, 20 instances. Judge = gpt-6-sol. Pairs with identical hypotheses and judge ties are skipped; a metric tie counts 0.5. CIs resample instances.

| judge_target | contrast | diff | ci_low | ci_high | excludes_0 |
|---|---|---|---|---|---|
| judge_check_pass_rate | doc_comet_w2 - comet | 0.010 | -0.025 | 0.048 | False |
| judge_check_pass_rate | xcomet - comet | 0.047 | -0.048 | 0.147 | False |
| judge_check_pass_rate | doc_comet_w2 - xcomet | -0.038 | -0.123 | 0.053 | False |
| judge_check_pass_rate | doc_comet_w5 - doc_comet_w2 | 0.002 | -0.017 | 0.021 | False |
| judge_check_pass_rate | metricx_neg - comet | 0.086 | 0.012 | 0.178 | True |
| judge_check_pass_rate | metricx_neg - doc_comet_w2 | 0.076 | 0.004 | 0.158 | True |
| judge_check_pass_rate | metricx_neg - xcomet | 0.039 | -0.011 | 0.088 | False |
| judge_mean | doc_comet_w2 - comet | 0.000 | -0.022 | 0.025 | False |
| judge_mean | xcomet - comet | 0.010 | -0.041 | 0.066 | False |
| judge_mean | doc_comet_w2 - xcomet | -0.010 | -0.073 | 0.049 | False |
| judge_mean | doc_comet_w5 - doc_comet_w2 | -0.002 | -0.014 | 0.009 | False |
| judge_mean | metricx_neg - comet | 0.035 | -0.007 | 0.083 | False |
| judge_mean | metricx_neg - doc_comet_w2 | 0.035 | -0.014 | 0.089 | False |
| judge_mean | metricx_neg - xcomet | 0.025 | -0.013 | 0.067 | False |
| judge_contextual_correctness | doc_comet_w2 - comet | 0.004 | -0.027 | 0.035 | False |
| judge_contextual_correctness | xcomet - comet | 0.044 | -0.026 | 0.127 | False |
| judge_contextual_correctness | doc_comet_w2 - xcomet | -0.040 | -0.118 | 0.033 | False |
| judge_contextual_correctness | doc_comet_w5 - doc_comet_w2 | -0.007 | -0.029 | 0.011 | False |
| judge_contextual_correctness | metricx_neg - comet | 0.065 | 0.007 | 0.137 | True |
| judge_contextual_correctness | metricx_neg - doc_comet_w2 | 0.061 | -0.002 | 0.130 | False |
| judge_contextual_correctness | metricx_neg - xcomet | 0.021 | -0.014 | 0.058 | False |
| judge_meaning_preservation | doc_comet_w2 - comet | 0.011 | -0.009 | 0.033 | False |
| judge_meaning_preservation | xcomet - comet | -0.002 | -0.060 | 0.066 | False |
| judge_meaning_preservation | doc_comet_w2 - xcomet | 0.013 | -0.054 | 0.073 | False |
| judge_meaning_preservation | doc_comet_w5 - doc_comet_w2 | 0.008 | -0.008 | 0.025 | False |
| judge_meaning_preservation | metricx_neg - comet | 0.051 | -0.005 | 0.120 | False |
| judge_meaning_preservation | metricx_neg - doc_comet_w2 | 0.040 | -0.021 | 0.107 | False |
| judge_meaning_preservation | metricx_neg - xcomet | 0.053 | 0.013 | 0.100 | True |
| judge_naturalness | doc_comet_w2 - comet | 0.013 | -0.031 | 0.053 | False |
| judge_naturalness | xcomet - comet | -0.039 | -0.102 | 0.016 | False |
| judge_naturalness | doc_comet_w2 - xcomet | 0.051 | 0.010 | 0.094 | True |
| judge_naturalness | doc_comet_w5 - doc_comet_w2 | -0.008 | -0.022 | 0.006 | False |
| judge_naturalness | metricx_neg - comet | -0.076 | -0.161 | -0.007 | True |
| judge_naturalness | metricx_neg - doc_comet_w2 | -0.089 | -0.157 | -0.031 | True |
| judge_naturalness | metricx_neg - xcomet | -0.037 | -0.110 | 0.041 | False |
