# Delta Table vs Baseline Mean

**Baseline mean computed from**: pilot_baseline_s0, rq1_baseline_s42, rq1_baseline_s123 (n=3)
**Baseline Forgetting**: 87.00% ± 0.34%

*Negative delta = improvement for lower-is-better metrics (forgetting, PPL, rep, drift)*  
*Positive delta = improvement for higher-is-better metrics (LAMBADA, tokens/sec)*

| Method   |   Seed | Δ Forgetting %   |   Δ PPL_A After |   Δ PPL_B After |   Δ LAMBADA After |   Δ Rep-4 After |   Δ Rep-8 After |   Δ Drift After | Δ Total Hours   |   Δ Tokens/sec |
|:---------|-------:|:-----------------|----------------:|----------------:|------------------:|----------------:|----------------:|----------------:|:----------------|---------------:|
| replay25 |     42 | -79.52%          |          -18.26 |            0.89 |             0.019 |         -0.1983 |         -0.2303 |          0.0223 | -0.19h          |           +146 |
| mer25    |     42 | -82.53%          |          -18.95 |            5.6  |             0.033 |         -0.1449 |         -0.17   |         -0.025  | -0.18h          |            +77 |
| ewc      |     42 | -53.55%          |          -12.3  |            0.9  |             0.058 |          0.0092 |          0.0087 |         -0.0027 | -0.02h          |            -51 |