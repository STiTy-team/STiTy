# T10 check pass by cue distance (required n) x given n

Mean instance-level check pass fraction, strategies pooled at each given n. Required-n groups (from each instance's note): n=1: c01-t12, c02-t10, c02-t12, c02-t15, c03-t14, c03-t17, c04-t08, c04-t12, c05-t12; n=3: c01-t06, c01-t08, c01-t14, c03-t05, c04-t05, c05-t14; n=5: c02-t07, c03-t12, c04-t13, c05-t09, c05-t13.

| model | required_n | n_instances | given_n0 | given_n1 | given_n3 | given_n5 |
|---|---|---|---|---|---|---|
| deepl-quality | 1 | 9 | 0.167 | 0.639 | 0.667 | 0.681 |
| deepl-quality | 3 | 6 | 0.222 | 0.250 | 0.646 | 0.729 |
| deepl-quality | 5 | 5 | 0.200 | 0.175 | 0.300 | 0.525 |
| gemma3-4b | 1 | 9 | 0.111 | 0.417 | 0.375 | 0.458 |
| gemma3-4b | 3 | 6 | 0.306 | 0.222 | 0.458 | 0.542 |
| gemma3-4b | 5 | 5 | 0.100 | 0.200 | 0.225 | 0.575 |
| gpt-6-luna | 1 | 9 | 0.333 | 0.764 | 0.764 | 0.819 |
| gpt-6-luna | 3 | 6 | 0.306 | 0.243 | 0.938 | 1.000 |
| gpt-6-luna | 5 | 5 | 0.200 | 0.375 | 0.500 | 0.875 |
| qwen3.5-4b | 1 | 9 | 0.056 | 0.556 | 0.556 | 0.569 |
| qwen3.5-4b | 3 | 6 | 0.528 | 0.333 | 0.688 | 0.646 |
| qwen3.5-4b | 5 | 5 | 0.100 | 0.100 | 0.275 | 0.675 |
| ALL | 1 | 9 | 0.167 | 0.594 | 0.590 | 0.632 |
| ALL | 3 | 6 | 0.340 | 0.262 | 0.682 | 0.729 |
| ALL | 5 | 5 | 0.150 | 0.212 | 0.325 | 0.662 |
