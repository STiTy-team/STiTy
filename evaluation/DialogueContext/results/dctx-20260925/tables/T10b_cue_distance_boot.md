# T10b cue entering the window vs context beyond the cue, paired bootstrap

Paired bootstrap over the 20 instances (1000 resamples, seed 20260925). Cells: mean difference [95% CI]; '*' = CI excludes 0 (endpoints compared at 3 decimals). check_pass is the instance-level pass fraction (mean over instances of passed/checks), so it can differ slightly from the pooled check_pass_rate in T0-T7. ALL averages each instance's difference over the four models before resampling. Strategies pooled. Only instances of that required-n group.

| model | required_n | contrast | n_instances | check_pass |
|---|---|---|---|---|
| deepl-quality | 1 | cue enters: n=req - n=below | 9 | +0.472 [+0.167, +0.806]* |
| deepl-quality | 1 | beyond cue: n3 - n1 | 9 | +0.028 [-0.083, +0.167] |
| deepl-quality | 1 | beyond cue: n5 - n1 | 9 | +0.042 [-0.083, +0.208] |
| deepl-quality | 3 | cue enters: n=req - n=below | 6 | +0.396 [+0.146, +0.667]* |
| deepl-quality | 3 | before cue: n1 - NONE | 6 | +0.028 [+0.000, +0.083] |
| deepl-quality | 3 | beyond cue: n5 - n3 | 6 | +0.083 [+0.000, +0.250] |
| deepl-quality | 5 | cue enters: n=req - n=below | 5 | +0.225 [+0.025, +0.500]* |
| deepl-quality | 5 | before cue: n3 - NONE | 5 | +0.100 [-0.100, +0.350] |
| gemma3-4b | 1 | cue enters: n=req - n=below | 9 | +0.306 [+0.056, +0.583]* |
| gemma3-4b | 1 | beyond cue: n3 - n1 | 9 | -0.042 [-0.125, +0.056] |
| gemma3-4b | 1 | beyond cue: n5 - n1 | 9 | +0.042 [-0.028, +0.139] |
| gemma3-4b | 3 | cue enters: n=req - n=below | 6 | +0.236 [+0.042, +0.472]* |
| gemma3-4b | 3 | before cue: n1 - NONE | 6 | -0.083 [-0.250, +0.000] |
| gemma3-4b | 3 | beyond cue: n5 - n3 | 6 | +0.083 [+0.000, +0.208] |
| gemma3-4b | 5 | cue enters: n=req - n=below | 5 | +0.350 [-0.026, +0.775] |
| gemma3-4b | 5 | before cue: n3 - NONE | 5 | +0.125 [+0.000, +0.275] |
| gpt-6-luna | 1 | cue enters: n=req - n=below | 9 | +0.431 [+0.139, +0.722]* |
| gpt-6-luna | 1 | beyond cue: n3 - n1 | 9 | +0.000 [-0.111, +0.139] |
| gpt-6-luna | 1 | beyond cue: n5 - n1 | 9 | +0.056 [-0.056, +0.194] |
| gpt-6-luna | 3 | cue enters: n=req - n=below | 6 | +0.694 [+0.528, +0.861]* |
| gpt-6-luna | 3 | before cue: n1 - NONE | 6 | -0.062 [-0.250, +0.062] |
| gpt-6-luna | 3 | beyond cue: n5 - n3 | 6 | +0.062 [+0.000, +0.146] |
| gpt-6-luna | 5 | cue enters: n=req - n=below | 5 | +0.375 [+0.300, +0.450]* |
| gpt-6-luna | 5 | before cue: n3 - NONE | 5 | +0.300 [+0.100, +0.500]* |
| qwen3.5-4b | 1 | cue enters: n=req - n=below | 9 | +0.500 [+0.194, +0.792]* |
| qwen3.5-4b | 1 | beyond cue: n3 - n1 | 9 | +0.000 [-0.111, +0.125] |
| qwen3.5-4b | 1 | beyond cue: n5 - n1 | 9 | +0.014 [-0.083, +0.125] |
| qwen3.5-4b | 3 | cue enters: n=req - n=below | 6 | +0.354 [+0.146, +0.542]* |
| qwen3.5-4b | 3 | before cue: n1 - NONE | 6 | -0.194 [-0.472, +0.056] |
| qwen3.5-4b | 3 | beyond cue: n5 - n3 | 6 | -0.042 [-0.125, +0.000] |
| qwen3.5-4b | 5 | cue enters: n=req - n=below | 5 | +0.400 [+0.100, +0.702]* |
| qwen3.5-4b | 5 | before cue: n3 - NONE | 5 | +0.175 [+0.025, +0.325]* |
| ALL | 1 | cue enters: n=req - n=below | 9 | +0.427 [+0.181, +0.691]* |
| ALL | 1 | beyond cue: n3 - n1 | 9 | -0.003 [-0.080, +0.104] |
| ALL | 1 | beyond cue: n5 - n1 | 9 | +0.038 [-0.042, +0.153] |
| ALL | 3 | cue enters: n=req - n=below | 6 | +0.420 [+0.243, +0.573]* |
| ALL | 3 | before cue: n1 - NONE | 6 | -0.078 [-0.182, +0.026] |
| ALL | 3 | beyond cue: n5 - n3 | 6 | +0.047 [-0.010, +0.104] |
| ALL | 5 | cue enters: n=req - n=below | 5 | +0.338 [+0.131, +0.575]* |
| ALL | 5 | before cue: n3 - NONE | 5 | +0.175 [+0.019, +0.331]* |
