# Baseline Variance Analysis

**Baseline runs**: gpt2_pilot_baseline_s0, gpt2_rq1_baseline_s42, gpt2_rq1_baseline_s123

| Metric        | Mean   |      Std | Min    | Max    |   N |
|:--------------|:-------|---------:|:-------|:-------|----:|
| Forgetting %  | 87.00% |   0.3374 | 86.68% | 87.35% |   3 |
| PPL_A After   | 42.90  |   0.0882 | 42.84  | 43.00  |   3 |
| PPL_B After   | 21.17  |   0.0026 | 21.17  | 21.17  |   3 |
| LAMBADA After | 0.188  |   0.0494 | 0.131  | 0.217  |   3 |
| Rep-4 After   | 0.6113 |   0.0269 | 0.5810 | 0.6324 |   3 |
| Drift After   | 0.3390 |   0.0032 | 0.3370 | 0.3428 |   3 |
| Total Hours   | 3.18h  |   0.149  | 3.04h  | 3.34h  |   3 |
| Tokens/sec    | 2850   | 200.526  | 2644   | 3045   |   3 |