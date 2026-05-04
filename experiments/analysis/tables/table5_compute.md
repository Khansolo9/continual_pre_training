# Table 5: Compute Efficiency

| Model        | Method        |   N |   Avg Hours |   Avg Tok/s |   Avg RAM (GB) |
|:-------------|:--------------|----:|------------:|------------:|---------------:|
| GPT-2 (124M) | Baseline      |   3 |         3.2 |        2850 |            1   |
| GPT-2 (124M) | Replay-25%    |   1 |         3   |        2996 |            1.2 |
| GPT-2 (124M) | MER-lite      |   1 |         3   |        2927 |            1.2 |
| GPT-2 (124M) | EWC           |   1 |         3.2 |        2799 |            1.4 |
| GPT-2 (124M) | Bandit Replay |   1 |         5.7 |        1149 |            0.5 |
| GPT-2 (124M) | RMGS          |   1 |         5.7 |        1151 |            0.8 |
| Qwen3 (0.6B) | Baseline      |   2 |         9.5 |         732 |            0.8 |
| Qwen3 (0.6B) | Replay-25%    |   1 |        10.6 |         593 |            1   |
| Qwen3 (0.6B) | MER-lite      |   1 |         9.3 |         687 |            0.7 |
| Qwen3 (0.6B) | EWC           |   1 |         8.7 |         750 |            0.6 |
| Gemma3 (1B)  | Baseline      |   2 |        19.7 |         336 |            0.8 |
| Llama3 (1B)  | Baseline      |   2 |        12.9 |         483 |            0.6 |
