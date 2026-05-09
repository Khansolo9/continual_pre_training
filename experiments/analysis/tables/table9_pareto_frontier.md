# Table 9: Pareto Frontier (forgetting × wall-time)

Pareto-optimal: no other run on the same model achieves both less forgetting AND less compute time. These are the rational method choices for a practitioner who cares about both.

| Model        | Method        |   Forget % |   Hours |   PPL_B after | Pareto-optimal   |
|:-------------|:--------------|-----------:|--------:|--------------:|:-----------------|
| GPT-2 (124M) | Baseline      |      87.35 |    3.04 |         21.17 |                  |
| GPT-2 (124M) | Baseline      |      86.68 |    3.17 |         21.17 |                  |
| GPT-2 (124M) | Baseline      |      86.95 |    3.34 |         21.17 |                  |
| GPT-2 (124M) | EWC           |      33.44 |    3.16 |         22.08 |                  |
| GPT-2 (124M) | MER-lite      |       4.47 |    3    |         26.77 | ✓                |
| GPT-2 (124M) | Replay-25%    |       7.48 |    2.99 |         22.06 | ✓                |
| GPT-2 (124M) | Bandit Replay |       7.27 |    5.73 |         22.09 |                  |
| GPT-2 (124M) | RMGS          |      25.02 |    5.73 |         27.36 |                  |
| Qwen3 (0.6B) | Baseline      |      44.49 |    8.73 |         11.14 |                  |
| Qwen3 (0.6B) | Baseline      |      44.52 |   10.36 |         11.13 |                  |
| Qwen3 (0.6B) | EWC           |      12.79 |    8.71 |         11.58 | ✓                |
| Qwen3 (0.6B) | MER-lite      |      26.46 |    9.26 |         12.41 |                  |
| Qwen3 (0.6B) | Replay-25%    |      44.3  |   10.64 |         11.13 |                  |
| Qwen3 (0.6B) | Bandit Replay |      10.39 |    8.78 |         11.31 | ✓                |
| Qwen3 (0.6B) | RMGS          |      18.44 |    8.94 |         13.58 |                  |
| Gemma3 (1B)  | Baseline      |      45.7  |   22.28 |         10.71 |                  |
| Gemma3 (1B)  | Baseline      |      45.91 |   17.03 |         10.7  |                  |
| Gemma3 (1B)  | EWC           |       8.85 |   29.24 |         10.85 | ✓                |
| Gemma3 (1B)  | MER-lite      |      23.42 |   14.13 |         11.69 |                  |
| Gemma3 (1B)  | Replay-25%    |      46.24 |   15.98 |         10.7  |                  |
| Gemma3 (1B)  | Bandit Replay |      30.07 |   12.54 |         10.87 |                  |
| Gemma3 (1B)  | RMGS          |      17.56 |   11.63 |         12.47 | ✓                |
| Llama3 (1B)  | Baseline      |      43.39 |   13.97 |          9.77 |                  |
| Llama3 (1B)  | Baseline      |      46.3  |   11.85 |          9.78 |                  |
| Llama3 (1B)  | EWC           |      45.07 |   37.52 |          9.79 |                  |
| Llama3 (1B)  | MER-lite      |      24.95 |   11.94 |         10.54 |                  |
| Llama3 (1B)  | Replay-25%    |      47.8  |   10.96 |          9.78 | ✓                |
| Llama3 (1B)  | Bandit Replay |      23.06 |   11.16 |          9.9  |                  |
| Llama3 (1B)  | RMGS          |      20.78 |   11.13 |         11.08 | ✓                |
