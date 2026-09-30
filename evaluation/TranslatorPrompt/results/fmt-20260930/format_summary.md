## gpt-6-luna

| 조건 | N | 위반률 일반 | 위반률 스트레스 | COMET | ΔCOMET vs F0 | MetricX↓ | ΔMetricX vs F0 | Kiwi 스트레스 |
|---|---|---|---|---|---|---|---|---|
| F0 | 0 | 0.000 | 0.022 | 0.850 | - | 2.63 | - | 0.824 |
| F0 | 16 | 0.000 | 0.044 | 0.851 | - | 2.62 | - | 0.807 |
| F1 | 0 | 0.000 | 0.011 | 0.851 | +0.000 | 2.56 | -0.071 | 0.826 |
| F1 | 16 | 0.000 | 0.033 | 0.853 | +0.002 | 2.55 | -0.069 | 0.824 |
| F2 | 0 | 0.000 | 0.033 | 0.851 | +0.000 | 2.61 | -0.015 | 0.826 |
| F2 | 16 | 0.000 | 0.033 | 0.851 | +0.001 | 2.54 | -0.082* | 0.815 |
| F3 | 0 | 0.000 | 0.000 | 0.851 | +0.000 | 2.60 | -0.030 | 0.833 |
| F3 | 16 | 0.000 | 0.000 | 0.851 | +0.001 | 2.65 | +0.027 | 0.835 |
| F4 | 0 | 0.000 | 0.022 | 0.853 | +0.002 | 2.60 | -0.030 | 0.824 |
| F4 | 16 | 0.560 | 0.256 | 0.853 | +0.002 | 2.54 | -0.083* | 0.809 |

위반 유형별 건수 (일반+스트레스)

| 조건 | N | 유형별 |
|---|---|---|
| F0 | 0 | not_korean 1, answered_or_obeyed 1, prompt_leak 1, json_in_plain 1 |
| F0 | 16 | not_korean 2, answered_or_obeyed 2, length_anomaly 1, json_in_plain 1, context_echo_fuzzy 1, context_echo 1 |
| F1 | 0 | prompt_leak 1, json_in_plain 1 |
| F1 | 16 | prompt_leak 2, json_in_plain 1, context_echo_fuzzy 1 |
| F2 | 0 | prompt_leak 3, json_in_plain 1 |
| F2 | 16 | prompt_leak 2, json_in_plain 1, context_echo_fuzzy 1 |
| F3 | 0 |  |
| F3 | 16 |  |
| F4 | 0 | prompt_leak 2 |
| F4 | 16 | json_fail 242, context_echo 2, json_extra 1, prompt_leak 1 |

스트레스 범주별 위반률

| 조건 | N | address_translator | asr_noise | fragment | instruction_injection | mixed_language | numbers_urls | quoted_dialogue | very_short |
|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
| F0 | 16 | 0.00 | 0.00 | 0.00 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 |
| F1 | 0 | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 |
| F1 | 16 | 0.00 | 0.00 | 0.00 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 |
| F2 | 0 | 0.00 | 0.00 | 0.00 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 |
| F2 | 16 | 0.00 | 0.00 | 0.00 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 |
| F3 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| F3 | 16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| F4 | 0 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
| F4 | 16 | 0.00 | 0.25 | 0.45 | 0.30 | 0.33 | 0.38 | 0.17 | 0.00 |

## qwen3.5-4b-bf16

| 조건 | N | 위반률 일반 | 위반률 스트레스 | COMET | ΔCOMET vs F0 | MetricX↓ | ΔMetricX vs F0 | Kiwi 스트레스 |
|---|---|---|---|---|---|---|---|---|
| F0 | 0 | 0.018 | 0.033 | 0.823 | - | 3.38 | - | 0.805 |
| F0 | 16 | 0.028 | 0.111 | 0.829 | - | 3.32 | - | 0.776 |
| F1 | 0 | 0.023 | 0.011 | 0.823 | +0.000 | 3.39 | +0.009 | 0.806 |
| F1 | 16 | 0.025 | 0.089 | 0.829 | +0.000 | 3.35 | +0.028 | 0.779 |
| F1_rp | 0 | 0.023 | 0.022 | 0.823 | +0.000 | 3.33 | -0.053 | 0.805 |
| F1_rp | 16 | 0.025 | 0.100 | 0.829 | +0.001 | 3.24 | -0.084 | 0.778 |
| F2 | 0 | 0.023 | 0.022 | 0.819 | -0.004 | 3.45 | +0.066 | 0.796 |
| F2 | 16 | 0.023 | 0.056 | 0.826 | -0.002 | 3.39 | +0.068 | 0.785 |
| F3 | 0 | 0.025 | 0.011 | 0.822 | -0.001 | 3.48 | +0.099 | 0.805 |
| F3 | 16 | 0.025 | 0.022 | 0.826 | -0.003 | 3.41 | +0.089 | 0.803 |
| F4 | 0 | 0.513 | 0.400 | 0.821 | -0.002 | 3.41 | +0.031 | 0.814 |
| F4 | 16 | 0.633 | 0.578 | 0.829 | +0.000 | 3.41 | +0.091 | 0.790 |

위반 유형별 건수 (일반+스트레스)

| 조건 | N | 유형별 |
|---|---|---|
| F0 | 0 | foreign_script 7, not_korean 3, truncated 1, repetition 1, length_anomaly 1, answered_or_obeyed 1, json_in_plain 1 |
| F0 | 16 | foreign_script 12, length_anomaly 4, context_echo_fuzzy 4, not_korean 4, multi_line 3, answered_or_obeyed 2, context_echo 2, truncated 1, repetition 1, json_in_plain 1 |
| F1 | 0 | foreign_script 9, not_korean 1, answered_or_obeyed 1 |
| F1 | 16 | foreign_script 10, not_korean 4, context_echo 3, context_echo_fuzzy 2, answered_or_obeyed 2, length_anomaly 2, multi_line 1, json_in_plain 1 |
| F1_rp | 0 | foreign_script 10, not_korean 2, answered_or_obeyed 1 |
| F1_rp | 16 | foreign_script 11, not_korean 5, context_echo_fuzzy 2, context_echo 2, answered_or_obeyed 2, length_anomaly 2, multi_line 1, json_in_plain 1 |
| F2 | 0 | foreign_script 9, not_korean 3, truncated 1, repetition 1, length_anomaly 1, answered_or_obeyed 1 |
| F2 | 16 | foreign_script 9, not_korean 4, answered_or_obeyed 2, context_echo_fuzzy 1, context_echo 1, length_anomaly 1, json_in_plain 1 |
| F3 | 0 | foreign_script 10, quoted 1 |
| F3 | 16 | foreign_script 9, context_echo_fuzzy 2, prompt_leak 1 |
| F4 | 0 | json_fail 235, foreign_script 7, quoted 1, not_korean 1 |
| F4 | 16 | json_fail 296, quoted 45, foreign_script 10, not_korean 3, answered_or_obeyed 2, length_anomaly 1, context_echo 1 |

스트레스 범주별 위반률

| 조건 | N | address_translator | asr_noise | fragment | instruction_injection | mixed_language | numbers_urls | quoted_dialogue | very_short |
|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | 0.00 | 0.00 | 0.00 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 |
| F0 | 16 | 0.00 | 0.05 | 0.05 | 0.70 | 0.00 | 0.00 | 0.17 | 0.00 |
| F1 | 0 | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 |
| F1 | 16 | 0.00 | 0.00 | 0.05 | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 |
| F1_rp | 0 | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.12 | 0.00 | 0.00 |
| F1_rp | 16 | 0.00 | 0.00 | 0.05 | 0.70 | 0.00 | 0.00 | 0.17 | 0.00 |
| F2 | 0 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
| F2 | 16 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |
| F3 | 0 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| F3 | 16 | 0.00 | 0.00 | 0.05 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 |
| F4 | 0 | 0.20 | 0.45 | 0.40 | 0.20 | 0.33 | 0.62 | 0.67 | 0.40 |
| F4 | 16 | 0.30 | 0.70 | 0.60 | 0.30 | 0.67 | 0.62 | 0.83 | 0.60 |
