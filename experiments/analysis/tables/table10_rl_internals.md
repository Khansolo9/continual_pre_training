# Table 10: RL Method Internal State

Headline patterns to look for:
- bandit_replay mean replay rate consistent across models?
- RMGS mean grad scale far from 1.0 (active throttling) or near 1.0 (rare throttling)?

| Model        | Method        |   Forget % | Mean replay rate   | Replay rate std   | Mean grad scale   | Grad scale std   |   N evaluations |
|:-------------|:--------------|-----------:|:-------------------|:------------------|:------------------|:-----------------|----------------:|
| Gemma3 (1B)  | Bandit Replay |      30.07 | 0.269              | 0.219             | —                 | —                |              24 |
| Gemma3 (1B)  | RMGS          |      17.56 | —                  | —                 | 0.986             | 0.024            |              24 |
| GPT-2 (124M) | Bandit Replay |       7.27 | 0.275              | 0.216             | —                 | —                |              24 |
| GPT-2 (124M) | RMGS          |      25.02 | —                  | —                 | 0.983             | 0.020            |              24 |
| Llama3 (1B)  | Bandit Replay |      23.06 | 0.275              | 0.216             | —                 | —                |              24 |
| Llama3 (1B)  | RMGS          |      20.78 | —                  | —                 | 0.985             | 0.027            |              24 |
| Qwen3 (0.6B) | Bandit Replay |      10.39 | 0.275              | 0.216             | —                 | —                |              24 |
| Qwen3 (0.6B) | RMGS          |      18.44 | —                  | —                 | 0.986             | 0.023            |              24 |
