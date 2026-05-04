# Table 3: Method Effectiveness vs Baseline (RQ2 / RQ4)

Delta from per-model baseline mean. Negative Δ Forgetting = better retention.

| Model        | Method        | Forgetting %   |   Δ Forgetting % |   Δ PPL_A |   Δ PPL_B |   Δ LAMBADA |
|:-------------|:--------------|:---------------|-----------------:|----------:|----------:|------------:|
| GPT-2 (124M) | EWC           | 33.44%         |           -53.55 |    -12.3  |      0.9  |       0.058 |
| GPT-2 (124M) | MER-lite      | 4.47%          |           -82.53 |    -18.95 |      5.6  |       0.033 |
| GPT-2 (124M) | Replay-25%    | 7.48%          |           -79.52 |    -18.26 |      0.89 |       0.019 |
| GPT-2 (124M) | Bandit Replay | 7.27%          |           -79.72 |    -18.3  |      0.92 |       0.026 |
| GPT-2 (124M) | RMGS          | 25.02%         |           -61.98 |    -14.24 |      6.19 |       0.043 |
| Qwen3 (0.6B) | EWC           | 12.79%         |           -31.71 |     -4.58 |      0.44 |       0.01  |
| Qwen3 (0.6B) | MER-lite      | 26.46%         |           -18.04 |     -2.61 |      1.28 |       0.026 |
| Qwen3 (0.6B) | Replay-25%    | 44.30%         |            -0.2  |     -0.02 |     -0    |       0.001 |
