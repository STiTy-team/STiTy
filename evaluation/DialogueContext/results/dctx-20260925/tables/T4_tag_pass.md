# T4 check pass rate by challenge tag and condition

Columns: NONE baseline, context length pooled over strategies (n=k), strategy pooled over n. Long form with counts: T4_tag_pass_long.csv.

| model | tag | NONE | n=1 | n=3 | n=5 | SRC | TGT | SRC_TGT | SPK_SRC_TGT | n_checks_NONE |
|---|---|---|---|---|---|---|---|---|---|---|
| deepl-quality | register_politeness | 0.000 | 0.000 | 0.000 | 0.083 | 0.111 | 0.000 | 0.000 | 0.000 | 3 |
| deepl-quality | omitted_argument | 0.000 | 0.400 | 0.800 | 1.000 | 0.733 | 0.733 | 0.733 | 0.733 | 5 |
| deepl-quality | lexical_consistency | 0.000 | 0.000 | 0.167 | 0.250 | 0.222 | 0.111 | 0.222 | 0.000 | 3 |
| deepl-quality | gender_reference | 0.200 | 0.600 | 0.800 | 0.950 | 0.867 | 0.800 | 0.733 | 0.733 | 5 |
| deepl-quality | fragment_incremental | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| deepl-quality | entity_consistency | 0.333 | 0.167 | 0.167 | 0.333 | 0.222 | 0.333 | 0.111 | 0.222 | 3 |
| deepl-quality | discourse_connective | 0.333 | 0.417 | 0.333 | 0.333 | 0.444 | 0.333 | 0.333 | 0.333 | 3 |
| deepl-quality | context_trap | 0.750 | 0.750 | 0.750 | 0.938 | 0.833 | 0.833 | 0.833 | 0.750 | 4 |
| deepl-quality | pronoun_coreference | 0.250 | 0.375 | 0.688 | 0.688 | 0.667 | 0.583 | 0.583 | 0.500 | 4 |
| deepl-quality | word_sense | 0.333 | 0.417 | 0.833 | 0.833 | 0.778 | 0.556 | 0.889 | 0.556 | 3 |
| gemma3-4b | entity_consistency | 0.000 | 0.000 | 0.000 | 0.500 | 0.111 | 0.111 | 0.222 | 0.222 | 3 |
| gemma3-4b | fragment_incremental | 0.000 | 0.750 | 0.583 | 0.833 | 0.667 | 0.556 | 0.778 | 0.889 | 3 |
| gemma3-4b | lexical_consistency | 0.000 | 0.083 | 0.167 | 0.417 | 0.222 | 0.333 | 0.222 | 0.111 | 3 |
| gemma3-4b | omitted_argument | 0.600 | 0.450 | 0.450 | 0.600 | 0.267 | 0.467 | 0.600 | 0.667 | 5 |
| gemma3-4b | pronoun_coreference | 0.000 | 0.125 | 0.250 | 0.500 | 0.333 | 0.167 | 0.250 | 0.417 | 4 |
| gemma3-4b | register_politeness | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| gemma3-4b | word_sense | 0.000 | 0.333 | 0.583 | 0.333 | 0.333 | 0.333 | 0.444 | 0.556 | 3 |
| gemma3-4b | discourse_connective | 0.000 | 0.000 | 0.083 | 0.083 | 0.000 | 0.000 | 0.111 | 0.111 | 3 |
| gemma3-4b | context_trap | 0.750 | 0.812 | 0.688 | 0.938 | 0.750 | 0.833 | 0.917 | 0.750 | 4 |
| gemma3-4b | gender_reference | 0.000 | 0.350 | 0.750 | 0.950 | 0.533 | 0.733 | 0.733 | 0.733 | 5 |
| gpt-6-luna | word_sense | 0.333 | 0.333 | 1.000 | 1.000 | 0.778 | 0.778 | 0.778 | 0.778 | 3 |
| gpt-6-luna | lexical_consistency | 0.000 | 0.250 | 0.250 | 0.833 | 0.111 | 0.556 | 0.556 | 0.556 | 3 |
| gpt-6-luna | gender_reference | 0.000 | 0.450 | 0.750 | 0.950 | 0.600 | 0.800 | 0.733 | 0.733 | 5 |
| gpt-6-luna | fragment_incremental | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| gpt-6-luna | entity_consistency | 0.000 | 0.333 | 0.417 | 0.833 | 0.333 | 0.667 | 0.556 | 0.556 | 3 |
| gpt-6-luna | discourse_connective | 0.333 | 0.667 | 0.583 | 0.667 | 0.667 | 0.444 | 0.667 | 0.778 | 3 |
| gpt-6-luna | context_trap | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4 |
| gpt-6-luna | omitted_argument | 0.200 | 0.400 | 0.950 | 1.000 | 0.733 | 0.800 | 0.800 | 0.800 | 5 |
| gpt-6-luna | pronoun_coreference | 0.250 | 0.438 | 0.938 | 0.938 | 0.833 | 0.583 | 0.833 | 0.833 | 4 |
| gpt-6-luna | register_politeness | 0.667 | 0.333 | 0.583 | 0.667 | 0.556 | 0.556 | 0.444 | 0.556 | 3 |
| qwen3.5-4b | pronoun_coreference | 0.250 | 0.438 | 0.750 | 0.750 | 0.917 | 0.333 | 0.667 | 0.667 | 4 |
| qwen3.5-4b | gender_reference | 0.400 | 0.650 | 0.750 | 1.000 | 0.800 | 0.800 | 0.800 | 0.800 | 5 |
| qwen3.5-4b | fragment_incremental | 0.000 | 1.000 | 0.833 | 0.917 | 0.889 | 0.778 | 1.000 | 1.000 | 3 |
| qwen3.5-4b | entity_consistency | 0.000 | 0.000 | 0.083 | 0.583 | 0.111 | 0.222 | 0.222 | 0.333 | 3 |
| qwen3.5-4b | discourse_connective | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| qwen3.5-4b | context_trap | 0.750 | 0.750 | 0.750 | 1.000 | 0.833 | 0.833 | 0.833 | 0.833 | 4 |
| qwen3.5-4b | lexical_consistency | 0.000 | 0.000 | 0.250 | 0.417 | 0.000 | 0.222 | 0.333 | 0.333 | 3 |
| qwen3.5-4b | omitted_argument | 0.200 | 0.650 | 0.700 | 0.800 | 0.733 | 0.600 | 0.733 | 0.800 | 5 |
| qwen3.5-4b | word_sense | 0.333 | 0.000 | 0.667 | 0.583 | 0.444 | 0.444 | 0.333 | 0.444 | 3 |
| qwen3.5-4b | register_politeness | 0.333 | 0.083 | 0.417 | 0.167 | 0.333 | 0.222 | 0.111 | 0.222 | 3 |
| ALL | word_sense | 0.250 | 0.271 | 0.771 | 0.688 | 0.583 | 0.528 | 0.611 | 0.583 | 12 |
| ALL | register_politeness | 0.250 | 0.104 | 0.250 | 0.229 | 0.250 | 0.194 | 0.139 | 0.194 | 12 |
| ALL | pronoun_coreference | 0.188 | 0.344 | 0.656 | 0.719 | 0.688 | 0.417 | 0.583 | 0.604 | 16 |
| ALL | omitted_argument | 0.250 | 0.475 | 0.725 | 0.850 | 0.617 | 0.650 | 0.717 | 0.750 | 20 |
| ALL | lexical_consistency | 0.000 | 0.083 | 0.208 | 0.479 | 0.139 | 0.306 | 0.333 | 0.250 | 12 |
| ALL | gender_reference | 0.150 | 0.512 | 0.762 | 0.963 | 0.700 | 0.783 | 0.750 | 0.750 | 20 |
| ALL | fragment_incremental | 0.167 | 0.938 | 0.854 | 0.938 | 0.889 | 0.833 | 0.944 | 0.972 | 12 |
| ALL | entity_consistency | 0.083 | 0.125 | 0.167 | 0.562 | 0.194 | 0.333 | 0.278 | 0.333 | 12 |
| ALL | discourse_connective | 0.167 | 0.271 | 0.250 | 0.271 | 0.278 | 0.194 | 0.278 | 0.306 | 12 |
| ALL | context_trap | 0.812 | 0.828 | 0.797 | 0.969 | 0.854 | 0.875 | 0.896 | 0.833 | 16 |
