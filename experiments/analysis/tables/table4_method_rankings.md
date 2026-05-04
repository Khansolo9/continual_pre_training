# Table 4: Method Rankings by Forgetting (per model)

Ranked from lowest to highest forgetting within each model family.

| Model        |   Rank | Method        | Forgetting %   |   PPL_B After |   LAMBADA |
|:-------------|-------:|:--------------|:---------------|--------------:|----------:|
| GPT-2 (124M) |      1 | MER-lite      | 4.47%          |         26.77 |     0.221 |
| GPT-2 (124M) |      2 | Bandit Replay | 7.27%          |         22.09 |     0.214 |
| GPT-2 (124M) |      3 | Replay-25%    | 7.48%          |         22.06 |     0.207 |
| GPT-2 (124M) |      4 | RMGS          | 25.02%         |         27.36 |     0.231 |
| GPT-2 (124M) |      5 | EWC           | 33.44%         |         22.08 |     0.246 |
| Qwen3 (0.6B) |      1 | EWC           | 12.79%         |         11.58 |     0.394 |
| Qwen3 (0.6B) |      2 | MER-lite      | 26.46%         |         12.41 |     0.41  |
| Qwen3 (0.6B) |      3 | Replay-25%    | 44.30%         |         11.13 |     0.385 |
