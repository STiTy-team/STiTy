# T12 GPU contention check for the local models

76 whole-GPU samples, 5-s interval, from 17:44:36 KST; run 17:42:42-17:50:56 KST (started_at UTC + 9 h). Each local-model row joined to the nearest sample (<= 5 s). Split at the median util 12%. gpu_util is the whole GPU, including our own job: mean util 40.2% over 22 samples taken while one of our local jobs was running, 16.0% over 54 samples while none was (the latter is roughly the other session's load). lat_resid = latency minus the model's own latency~input_tokens line. DIFF rows: median difference with 95% bootstrap CI (rows resampled within each window).

| model | window | n_rows | median_latency_ms | median_ttft_ms | median_resid_ms | mean_input_tokens | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|
| gemma3-4b | before first sample | 65 | 238.920 | 39.760 | -33.397 | 204.231 | — | — |
| gemma3-4b | util <= 12% | 92 | 248.830 | 38.690 | -22.800 | 219.533 | — | — |
| gemma3-4b | util > 12% (median) | 103 | 279.710 | 37.790 | 2.346 | 213.524 | — | — |
| gemma3-4b | DIFF high - low util | 195 | 30.880 | — | — | — | -2.723 | 64.159 |
| gemma3-4b | DIFF first 2 min - rest | 260 | -27.620 | — | — | — | -49.041 | 5.096 |
| gemma3-4b | corr(util, latency residual) | 195 | 0.078 | — | — | — | — | — |
| qwen3.5-4b | before first sample | 59 | 288.940 | 40.690 | -14.802 | 211.373 | — | — |
| qwen3.5-4b | util <= 12% | 99 | 293.480 | 39.520 | -15.945 | 218.515 | — | — |
| qwen3.5-4b | util > 12% (median) | 102 | 315.840 | 40.830 | 10.989 | 222.225 | — | — |
| qwen3.5-4b | DIFF high - low util | 201 | 22.360 | — | — | — | -10.649 | 46.642 |
| qwen3.5-4b | DIFF first 2 min - rest | 260 | -18.820 | — | — | — | -46.790 | 21.570 |
| qwen3.5-4b | corr(util, latency residual) | 201 | 0.110 | — | — | — | — | — |
