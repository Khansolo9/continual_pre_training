# Primary Results Table

**Domain A**: wikitext-103 (10M tokens)  
**Domain B**: arxiv_abstracts (10M tokens)  
**Runs included**: 6 MPP core runs (smoke runs excluded)  
**Generated**: 2026-02-07T21:12:40Z

| Run ID            | Method   |   Seed | Forgetting %   |   PPL_A Init |   PPL_A Before |   PPL_A After |   PPL_B Init |   PPL_B After |   LAMBADA Before |   LAMBADA After |   Rep-4 Before |   Rep-4 After |   Rep-8 Before |   Rep-8 After |   Drift Before |   Drift After |   Vocab Overlap Before |   Vocab Overlap After | Total Hours   |   Tokens/sec |
|:------------------|:---------|-------:|:---------------|-------------:|---------------:|--------------:|-------------:|--------------:|-----------------:|----------------:|---------------:|--------------:|---------------:|--------------:|---------------:|--------------:|-----------------------:|----------------------:|:--------------|-------------:|
| pilot_baseline_s0 | baseline |      0 | 87.35%         |        35.83 |          22.95 |         43    |        49.03 |         21.17 |            0.149 |           0.131 |         0.419  |        0.581  |         0.2671 |        0.4639 |         0.4199 |        0.3428 |                 0.2507 |                0.2018 | 3.04h         |         3045 |
| rq1_baseline_s42  | baseline |     42 | 86.95%         |        35.83 |          22.93 |         42.87 |        49.03 |         21.17 |            0.203 |           0.217 |         0.4644 |        0.6206 |         0.3233 |        0.51   |         0.4208 |        0.3372 |                 0.2471 |                0.2111 | 3.34h         |         2644 |
| rq1_baseline_s123 | baseline |    123 | 86.68%         |        35.83 |          22.95 |         42.84 |        49.03 |         21.17 |            0.202 |           0.216 |         0.4348 |        0.6324 |         0.2871 |        0.5181 |         0.4278 |        0.337  |                 0.2383 |                0.2012 | 3.17h         |         2861 |
| rq2_replay25_s42  | replay25 |     42 | 7.48%          |        35.83 |          22.93 |         24.64 |        49.03 |         22.06 |            0.203 |           0.207 |         0.4605 |        0.413  |         0.3213 |        0.2671 |         0.4208 |        0.3613 |                 0.2471 |                0.2528 | 2.99h         |         2996 |
| rq2_mer25_s42     | mer25    |     42 | 4.47%          |        35.83 |          22.93 |         23.95 |        49.03 |         26.77 |            0.203 |           0.221 |         0.4565 |        0.4664 |         0.3153 |        0.3273 |         0.423  |        0.314  |                 0.2437 |                0.2665 | 3.00h         |         2927 |
| rq2_ewc_s42       | ewc      |     42 | 33.44%         |        35.83 |          22.93 |         30.6  |        49.03 |         22.08 |            0.202 |           0.246 |         0.4723 |        0.6206 |         0.3213 |        0.506  |         0.4207 |        0.3363 |                 0.2511 |                0.2211 | 3.16h         |         2799 |

**Notes**:
- `high_rep4` anomaly flagged on all runs (expected GPT-2 behavior)
- `peak_vram_gb` is N/A on MPS (Mac); RAM tracking used instead
- Rep-n metrics computed with sampling (temperature=0.7, top_p=0.9)
