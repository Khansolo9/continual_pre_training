# Table 6: Method Consistency Across Models

Lower CV = method behaves consistently across architectures. Higher CV = effectiveness depends on the model.

| Method        |   N models |   Mean forget % |   Std forget % |   CV % |   Min forget % |   Max forget % |   Range |
|:--------------|-----------:|----------------:|---------------:|-------:|---------------:|---------------:|--------:|
| RMGS          |          4 |           20.45 |           3.33 |   16.3 |          17.56 |          25.02 |    7.46 |
| MER-lite      |          4 |           19.82 |          10.31 |   52   |           4.47 |          26.46 |   21.99 |
| Replay-25%    |          4 |           36.46 |          19.37 |   53.1 |           7.48 |          47.8  |   40.32 |
| Bandit Replay |          4 |           17.7  |          10.71 |   60.5 |           7.27 |          30.07 |   22.8  |
| EWC           |          4 |           25.04 |          17.16 |   68.5 |           8.85 |          45.07 |   36.22 |
