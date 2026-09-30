# T15g Score drop for defects planted in the gold reference

*_drop_sd = mean drop / within-instance SD of that metric on the real run. More negative = the metric notices the defect more. MetricX sign flipped.

| probe | n | comet_raw | doc_comet_w2_raw | xcomet_raw | metricx_neg_raw | comet_drop_sd | doc_comet_w2_drop_sd | xcomet_drop_sd | metricx_neg_drop_sd |
|---|---|---|---|---|---|---|---|---|---|
| ref (absolute score) | 20 | 0.967 | 0.935 | 0.964 | -0.851 | — | — | — | — |
| copy_context | 20 | -0.236 | -0.234 | -0.351 | -4.831 | -3.790 | -3.307 | -4.827 | -2.609 |
| drop_pronoun | 10 | -0.056 | -0.100 | -0.125 | -1.655 | -0.906 | -1.416 | -1.711 | -0.894 |
| pronoun_flip | 7 | -0.053 | -0.098 | -0.122 | -5.670 | -0.848 | -1.382 | -1.670 | -3.063 |
| truncate | 20 | -0.281 | -0.348 | -0.278 | -10.121 | -4.528 | -4.913 | -3.820 | -5.466 |
