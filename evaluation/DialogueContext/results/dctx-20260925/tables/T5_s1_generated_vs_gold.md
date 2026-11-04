# T5 S1: model's own earlier translations vs gold history

Same model, strategy and n; restricted to instances present in both.

| model | context_strategy | context_n | history | n_instances | n_rows | n_failed | comet | xcomet | judge_mean | contextual_correctness | speaker_consistency | check_pass_rate | latency_median_ms | corpus_chrf_pp | corpus_bleu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepl-quality | SRC_TGT | 3 | gold_history | 20 | 20.000 | 0.000 | 0.855 | 0.919 | 4.442 | 3.750 | 4.550 | 0.611 | 455.610 | 52.942 | 30.563 |
| deepl-quality | SRC_TGT | 3 | generated_history | 20 | 20.000 | 0.000 | 0.849 | 0.917 | 4.383 | 3.750 | 4.500 | 0.583 | — | 52.296 | 29.386 |
| deepl-quality | SRC_TGT | 3 | delta(generated-gold) | 20 | — | — | -0.007 | -0.003 | -0.058 | 0.000 | -0.050 | -0.028 | — | -0.647 | -1.177 |
| deepl-quality | TGT | 3 | gold_history | 20 | 20.000 | 0.000 | 0.848 | 0.913 | 4.333 | 3.550 | 4.350 | 0.556 | 434.925 | 54.351 | 32.132 |
| deepl-quality | TGT | 3 | generated_history | 20 | 20.000 | 0.000 | 0.842 | 0.909 | 4.258 | 3.550 | 4.250 | 0.556 | — | 54.480 | 31.137 |
| deepl-quality | TGT | 3 | delta(generated-gold) | 20 | — | — | -0.006 | -0.004 | -0.075 | 0.000 | -0.100 | 0.000 | — | 0.129 | -0.994 |
| gemma3-4b | SRC_TGT | 3 | gold_history | 20 | 20.000 | 0.000 | 0.799 | 0.852 | 3.858 | 2.900 | 3.850 | 0.417 | 239.210 | 36.901 | 15.776 |
| gemma3-4b | SRC_TGT | 3 | generated_history | 20 | 20.000 | 0.000 | 0.808 | 0.864 | 3.900 | 3.000 | 3.800 | 0.417 | — | 37.945 | 16.100 |
| gemma3-4b | SRC_TGT | 3 | delta(generated-gold) | 20 | — | — | 0.009 | 0.013 | 0.042 | 0.100 | -0.050 | 0.000 | — | 1.044 | 0.324 |
| gemma3-4b | TGT | 3 | gold_history | 20 | 20.000 | 0.000 | 0.769 | 0.844 | 3.758 | 2.700 | 3.550 | 0.361 | 239.265 | 34.228 | 12.275 |
| gemma3-4b | TGT | 3 | generated_history | 20 | 20.000 | 0.000 | 0.785 | 0.849 | 3.833 | 2.900 | 3.550 | 0.333 | — | 32.639 | 10.672 |
| gemma3-4b | TGT | 3 | delta(generated-gold) | 20 | — | — | 0.016 | 0.005 | 0.075 | 0.200 | 0.000 | -0.028 | — | -1.590 | -1.603 |
| gpt-6-luna | SRC_TGT | 3 | gold_history | 20 | 20.000 | 0.000 | 0.849 | 0.930 | 4.708 | 4.400 | 4.800 | 0.778 | 804.040 | 53.127 | 34.457 |
| gpt-6-luna | SRC_TGT | 3 | generated_history | 20 | 20.000 | 0.000 | 0.861 | 0.937 | 4.700 | 4.400 | 4.750 | 0.778 | — | 56.118 | 36.902 |
| gpt-6-luna | SRC_TGT | 3 | delta(generated-gold) | 20 | — | — | 0.012 | 0.007 | -0.008 | 0.000 | -0.050 | 0.000 | — | 2.991 | 2.445 |
| gpt-6-luna | TGT | 3 | gold_history | 20 | 20.000 | 0.000 | 0.852 | 0.928 | 4.675 | 4.300 | 4.650 | 0.778 | 773.200 | 56.354 | 39.057 |
| gpt-6-luna | TGT | 3 | generated_history | 20 | 20.000 | 0.000 | 0.855 | 0.931 | 4.633 | 4.200 | 4.600 | 0.722 | — | 56.908 | 37.727 |
| gpt-6-luna | TGT | 3 | delta(generated-gold) | 20 | — | — | 0.003 | 0.003 | -0.042 | -0.100 | -0.050 | -0.056 | — | 0.553 | -1.331 |
| qwen3.5-4b | SRC_TGT | 3 | gold_history | 20 | 20.000 | 0.000 | 0.804 | 0.876 | 4.142 | 3.350 | 4.250 | 0.583 | 300.570 | 39.989 | 17.251 |
| qwen3.5-4b | SRC_TGT | 3 | generated_history | 20 | 20.000 | 0.000 | 0.793 | 0.855 | 4.017 | 3.150 | 4.200 | 0.500 | — | 38.813 | 16.503 |
| qwen3.5-4b | SRC_TGT | 3 | delta(generated-gold) | 20 | — | — | -0.011 | -0.021 | -0.125 | -0.200 | -0.050 | -0.083 | — | -1.176 | -0.748 |
| qwen3.5-4b | TGT | 3 | gold_history | 20 | 20.000 | 0.000 | 0.792 | 0.865 | 3.958 | 3.000 | 3.900 | 0.472 | 289.675 | 39.621 | 15.508 |
| qwen3.5-4b | TGT | 3 | generated_history | 20 | 20.000 | 0.000 | 0.786 | 0.851 | 3.892 | 2.950 | 3.750 | 0.417 | — | 36.197 | 13.229 |
| qwen3.5-4b | TGT | 3 | delta(generated-gold) | 20 | — | — | -0.007 | -0.014 | -0.067 | -0.050 | -0.150 | -0.056 | — | -3.424 | -2.279 |
