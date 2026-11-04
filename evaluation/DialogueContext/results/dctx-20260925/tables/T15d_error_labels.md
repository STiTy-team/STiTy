# T15d Within-instance score gap (SD units) of rows with each judge error label

Negative = the metric scores labelled rows lower, as it should.

| error_label | n_rows | n_instances | comet | doc_comet_w2 | doc_comet_w5 | xcomet | metricx_neg | chrf_pp |
|---|---|---|---|---|---|---|---|---|
| mistranslation | 397 | 18 | -0.833 | -0.904 | -0.870 | -0.840 | -0.923 | -0.888 |
| wrong_coreference | 341 | 19 | -0.802 | -0.910 | -0.858 | -0.955 | -1.134 | -0.722 |
| lexical_inconsistency | 156 | 5 | -0.843 | -0.883 | -0.801 | -1.011 | -0.925 | -1.097 |
| unnatural | 110 | 16 | -0.832 | -0.899 | -0.810 | -0.853 | -0.491 | -0.740 |
| omission | 86 | 14 | -1.298 | -1.492 | -1.591 | -0.847 | -0.888 | -1.086 |
| wrong_gender | 79 | 6 | -1.114 | -1.077 | -0.974 | -0.994 | -1.481 | -0.666 |
| wrong_register | 52 | 3 | -0.229 | -0.073 | 0.008 | 0.074 | 0.144 | -0.229 |
| addition | 41 | 13 | -1.150 | -1.141 | -1.181 | -0.861 | -0.684 | -0.811 |
| context_copied | 2 | 2 | -1.045 | -1.031 | -0.726 | -3.312 | -0.098 | -0.620 |
| format_violation | 2 | 2 | -1.038 | -1.567 | -1.288 | -0.924 | -0.376 | -1.029 |
