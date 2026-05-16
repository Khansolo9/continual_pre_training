# Experiment Summary Pack

**Generated**: 2026-05-16T18:53:41Z
**Total Completed Runs**: 30
**Scope**: 4-model x 6-method matrix; per-model breakdown below.

> Authoritative cross-model synthesis with figures lives in `docs/reports/ANALYSIS_ATLAS.md` and `experiments/analysis/`. This pack is a flat machine-friendly companion (paired with `experiments/summary_table.csv`) regenerated from `experiments/run_registry.csv` + per-run `metrics.json`.

---

## Cross-Model Overview

**Total runs**: 30 (non-smoke, completed)
**Model families**: GPT-2 Small (124M), Qwen3 (0.6B), Gemma 3 (1B), Llama 3.2 (1B)
**Methods present**: , bandit_replay, baseline, ewc, mer25, replay25, rmgs

### Baseline forgetting by model

| Model | n | Mean Forgetting | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| GPT-2 Small (124M) | 3 | 87.00% | 0.34 | 86.68% | 87.35% |
| Qwen3 (0.6B) | 2 | 44.50% | 0.02 | 44.49% | 44.52% |
| Gemma 3 (1B) | 2 | 45.80% | 0.15 | 45.70% | 45.91% |
| Llama 3.2 (1B) | 2 | 44.84% | 2.06 | 43.39% | 46.30% |

> Per-model breakdowns follow. For figures, full tables (T1–T10), and synthesized analysis, see `docs/reports/ANALYSIS_ATLAS.md` and `experiments/analysis/{tables,figures}/`.

---

## GPT-2 Small (124M)

*8 run(s) for this model family.*

## Executive Summary

**Completed Runs**: 8
**Methods Tested**: bandit_replay, baseline, ewc, mer25, replay25, rmgs
**Baseline Seeds**: 3

**Baseline Forgetting**: 87.00% ± 0.34% (n=3)

### Key Findings

- **Best Forgetting**: mer25 (4.47%, Δ=+82.53% vs baseline)
- **Fastest Method**: replay25 (2.99h)
- **Highest Throughput**: replay25 (2996 tok/s)

## Full Comparison Table

| Run ID | Method | Seed | Forgetting % | PPL(A) After | PPL(B) After | LAMBADA After | Rep-4 After | Drift (JS) | Total Hours | Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2_pilot_baseline_s0 | baseline | 0 | 87.35% | 43.00 | 21.17 | 0.131 | 0.5810 | 0.3428 | 3.04h | 3045 |
| gpt2_rq1_baseline_s42 | baseline | 42 | 86.95% | 42.87 | 21.17 | 0.217 | 0.6206 | 0.3372 | 3.34h | 2644 |
| gpt2_rq1_baseline_s123 | baseline | 123 | 86.68% | 42.84 | 21.17 | 0.216 | 0.6324 | 0.3370 | 3.17h | 2861 |
| gpt2_rq2_replay25_s42 | replay25 | 42 | 7.48% | 24.64 | 22.06 | 0.207 | 0.4130 | 0.3613 | 2.99h | 2996 |
| gpt2_rq2_mer25_s42 | mer25 | 42 | 4.47% | 23.95 | 26.77 | 0.221 | 0.4664 | 0.3140 | 3.00h | 2927 |
| gpt2_rq2_ewc_s42 | ewc | 42 | 33.44% | 30.60 | 22.08 | 0.246 | 0.6206 | 0.3363 | 3.16h | 2799 |
| gpt2_rq3_bandit_replay_s42 | bandit_replay | 42 | 7.27% | 24.60 | 22.09 | 0.214 | 0.4209 | 0.3659 | 5.73h | 1149 |
| gpt2_rq3_rmgs_s42 | rmgs | 42 | 25.02% | 28.67 | 27.36 | 0.231 | 0.6522 | 0.3269 | 5.73h | 1151 |

## Baseline Variance Analysis

| Metric | Mean | Std | Min | Max | N |
|--------|------|-----|-----|-----|---|
| Forgetting % | 87.00% | 0.34 | 86.68% | 87.35% | 3 |
| PPL(A) After | 42.90 | 0.09 | 42.84 | 43.00 | 3 |
| PPL(B) After | 21.17 | 0.00 | 21.17 | 21.17 | 3 |
| LAMBADA After | 0.188 | 0.049 | 0.131 | 0.217 | 3 |
| Rep-4 After | 0.6113 | 0.0269 | 0.5810 | 0.6324 | 3 |
| Drift (JS) | 0.3390 | 0.0032 | 0.3370 | 0.3428 | 3 |
| Total Hours | 3.18h | 0.15 | 3.04h | 3.34h | 3 |
| Avg Tok/s | 2850 | 201 | 2644 | 3045 | 3 |

## Method Deltas vs Baseline Mean

*Negative delta = improvement for lower-is-better metrics (forgetting, PPL, rep, drift, time)*
*Positive delta = improvement for higher-is-better metrics (LAMBADA, tok/s)*

| Method | Seed | Δ Forgetting % | Δ PPL(A) After | Δ PPL(B) After | Δ LAMBADA After | Δ Rep-4 After | Δ Drift (JS) | Δ Total Hours | Δ Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|
| replay25 | 42 | -79.52% | -18.26 | +0.89 | +0.019 | -0.1983 | +0.0223 | -0.19h | +146 |
| mer25 | 42 | -82.53% | -18.95 | +5.60 | +0.033 | -0.1449 | -0.0250 | -0.18h | +77 |
| ewc | 42 | -53.55% | -12.30 | +0.90 | +0.058 | +0.0092 | -0.0027 | -0.02h | -51 |
| bandit_replay | 42 | -79.72% | -18.30 | +0.92 | +0.026 | -0.1904 | +0.0269 | +2.54h | -1701 |
| rmgs | 42 | -61.98% | -14.24 | +6.19 | +0.043 | +0.0408 | -0.0121 | +2.54h | -1699 |

## Pareto-Style Rankings

### Best Forgetting Reduction per Hour
1. **mer25**: 27.465%/h (Δ=+82.53% in 3.00h)
2. **replay25**: 26.568%/h (Δ=+79.52% in 2.99h)
3. **ewc**: 16.954%/h (Δ=+53.55% in 3.16h)
4. **bandit_replay**: 13.922%/h (Δ=+79.72% in 5.73h)
5. **rmgs**: 10.822%/h (Δ=+61.98% in 5.73h)

### Best Domain A Retention (Lowest Forgetting)
1. **mer25**: 4.47%
2. **bandit_replay**: 7.27%
3. **replay25**: 7.48%
4. **rmgs**: 25.02%
5. **ewc**: 33.44%

### Best General Ability (LAMBADA)
1. **ewc**: 0.246
2. **rmgs**: 0.231
3. **mer25**: 0.221
4. **bandit_replay**: 0.214
5. **replay25**: 0.207

### Best Throughput
1. **replay25**: 2996 tok/s
2. **mer25**: 2927 tok/s
3. **ewc**: 2799 tok/s
4. **rmgs**: 1151 tok/s
5. **bandit_replay**: 1149 tok/s

---

## Qwen3 (0.6B)

*7 run(s) for this model family.*

## Executive Summary

**Completed Runs**: 7
**Methods Tested**: bandit_replay, baseline, ewc, mer25, replay25, rmgs
**Baseline Seeds**: 2

**Baseline Forgetting**: 44.50% ± 0.02% (n=2)

### Key Findings

- **Best Forgetting**: bandit_replay (10.39%, Δ=+34.11% vs baseline)
- **Fastest Method**: ewc (8.71h)
- **Highest Throughput**: ewc (750 tok/s)

## Full Comparison Table

| Run ID | Method | Seed | Forgetting % | PPL(A) After | PPL(B) After | LAMBADA After | Rep-4 After | Drift (JS) | Total Hours | Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3_rq1_baseline_s42 | baseline | 42 | 44.52% | 20.89 | 11.13 | 0.386 | 0.5040 | 0.2699 | 10.36h | 724 |
| qwen3_rq1_baseline_s123 | baseline | 123 | 44.49% | 20.88 | 11.14 | 0.382 | 0.5830 | 0.2585 | 8.73h | 739 |
| qwen3_rq2_replay25_s42 | replay25 | 42 | 44.30% | 20.86 | 11.13 | 0.385 | 0.5099 | 0.2763 | 10.64h | 593 |
| qwen3_rq2_mer25_s42 | mer25 | 42 | 26.46% | 18.28 | 12.41 | 0.410 | 0.5632 | 0.2566 | 9.26h | 687 |
| qwen3_rq2_ewc_s42 | ewc | 42 | 12.79% | 16.30 | 11.58 | 0.394 | 0.5277 | 0.2450 | 8.71h | 750 |
| qwen3_rq4_bandit_replay_s42 | bandit_replay | 42 | 10.39% | 15.96 | 11.31 | 0.353 | 0.2154 | 0.2713 | 8.78h | 731 |
| qwen3_rq4_rmgs_s42 | rmgs | 42 | 18.44% | 17.12 | 13.58 | 0.417 | 0.5455 | 0.2459 | 8.94h | 716 |

## Baseline Variance Analysis

| Metric | Mean | Std | Min | Max | N |
|--------|------|-----|-----|-----|---|
| Forgetting % | 44.50% | 0.02 | 44.49% | 44.52% | 2 |
| PPL(A) After | 20.89 | 0.01 | 20.88 | 20.89 | 2 |
| PPL(B) After | 11.13 | 0.00 | 11.13 | 11.14 | 2 |
| LAMBADA After | 0.384 | 0.003 | 0.382 | 0.386 | 2 |
| Rep-4 After | 0.5435 | 0.0559 | 0.5040 | 0.5830 | 2 |
| Drift (JS) | 0.2642 | 0.0081 | 0.2585 | 0.2699 | 2 |
| Total Hours | 9.54h | 1.15 | 8.73h | 10.36h | 2 |
| Avg Tok/s | 732 | 11 | 724 | 739 | 2 |

## Method Deltas vs Baseline Mean

*Negative delta = improvement for lower-is-better metrics (forgetting, PPL, rep, drift, time)*
*Positive delta = improvement for higher-is-better metrics (LAMBADA, tok/s)*

| Method | Seed | Δ Forgetting % | Δ PPL(A) After | Δ PPL(B) After | Δ LAMBADA After | Δ Rep-4 After | Δ Drift (JS) | Δ Total Hours | Δ Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|
| replay25 | 42 | -0.20% | -0.02 | -0.00 | +0.001 | -0.0336 | +0.0121 | +1.09h | -139 |
| mer25 | 42 | -18.04% | -2.61 | +1.28 | +0.026 | +0.0198 | -0.0077 | -0.28h | -45 |
| ewc | 42 | -31.71% | -4.58 | +0.44 | +0.010 | -0.0158 | -0.0192 | -0.84h | +19 |
| bandit_replay | 42 | -34.11% | -4.93 | +0.17 | -0.031 | -0.3281 | +0.0070 | -0.76h | -1 |
| rmgs | 42 | -26.06% | -3.77 | +2.45 | +0.033 | +0.0020 | -0.0184 | -0.60h | -15 |

## Pareto-Style Rankings

### Best Forgetting Reduction per Hour
1. **bandit_replay**: 3.885%/h (Δ=+34.11% in 8.78h)
2. **ewc**: 3.642%/h (Δ=+31.71% in 8.71h)
3. **rmgs**: 2.915%/h (Δ=+26.06% in 8.94h)
4. **mer25**: 1.949%/h (Δ=+18.04% in 9.26h)
5. **replay25**: 0.019%/h (Δ=+0.20% in 10.64h)

### Best Domain A Retention (Lowest Forgetting)
1. **bandit_replay**: 10.39%
2. **ewc**: 12.79%
3. **rmgs**: 18.44%
4. **mer25**: 26.46%
5. **replay25**: 44.30%

### Best General Ability (LAMBADA)
1. **rmgs**: 0.417
2. **mer25**: 0.410
3. **ewc**: 0.394
4. **replay25**: 0.385
5. **bandit_replay**: 0.353

### Best Throughput
1. **ewc**: 750 tok/s
2. **bandit_replay**: 731 tok/s
3. **rmgs**: 716 tok/s
4. **mer25**: 687 tok/s
5. **replay25**: 593 tok/s

---

## Gemma 3 (1B)

*7 run(s) for this model family.*

## Executive Summary

**Completed Runs**: 7
**Methods Tested**: bandit_replay, baseline, ewc, mer25, replay25, rmgs
**Baseline Seeds**: 2

**Baseline Forgetting**: 45.80% ± 0.15% (n=2)

### Key Findings

- **Best Forgetting**: ewc (8.85%, Δ=+36.95% vs baseline)
- **Fastest Method**: rmgs (11.63h)
- **Highest Throughput**: rmgs (567 tok/s)

## Full Comparison Table

| Run ID | Method | Seed | Forgetting % | PPL(A) After | PPL(B) After | LAMBADA After | Rep-4 After | Drift (JS) | Total Hours | Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_rq1_baseline_s42 | baseline | 42 | 45.91% | 15.60 | 10.70 | 0.448 | 0.1937 | 0.2826 | 17.03h | 367 |
| gemma3_rq1_baseline_s123 | baseline | 123 | 45.70% | 15.56 | 10.71 | 0.451 | 0.1581 | 0.2715 | 22.28h | 305 |
| gemma3_rq2_replay25_s42 | replay25 | 42 | 46.24% | 15.64 | 10.70 | 0.458 | 0.2016 | 0.2739 | 15.98h | 402 |
| gemma3_rq2_mer25_s42 | mer25 | 42 | 23.42% | 13.19 | 11.69 | 0.476 | 0.1937 | 0.2650 | 14.13h | 452 |
| gemma3_rq2_ewc_s42 | ewc | 42 | 8.85% | 11.64 | 10.85 | 0.453 | 0.3360 | 0.2866 | 29.24h | 333 |
| gemma3_rq4_bandit_replay_s42 | bandit_replay | 42 | 30.07% | 13.90 | 10.87 | 0.462 | 0.0692 | 0.2795 | 12.54h | 522 |
| gemma3_rq4_rmgs_s42 | rmgs | 42 | 17.56% | 12.57 | 12.47 | 0.474 | 0.1542 | 0.2630 | 11.63h | 567 |

## Baseline Variance Analysis

| Metric | Mean | Std | Min | Max | N |
|--------|------|-----|-----|-----|---|
| Forgetting % | 45.80% | 0.15 | 45.70% | 45.91% | 2 |
| PPL(A) After | 15.58 | 0.03 | 15.56 | 15.60 | 2 |
| PPL(B) After | 10.71 | 0.01 | 10.70 | 10.71 | 2 |
| LAMBADA After | 0.450 | 0.002 | 0.448 | 0.451 | 2 |
| Rep-4 After | 0.1759 | 0.0252 | 0.1581 | 0.1937 | 2 |
| Drift (JS) | 0.2770 | 0.0078 | 0.2715 | 0.2826 | 2 |
| Total Hours | 19.66h | 3.71 | 17.03h | 22.28h | 2 |
| Avg Tok/s | 336 | 43 | 305 | 367 | 2 |

## Method Deltas vs Baseline Mean

*Negative delta = improvement for lower-is-better metrics (forgetting, PPL, rep, drift, time)*
*Positive delta = improvement for higher-is-better metrics (LAMBADA, tok/s)*

| Method | Seed | Δ Forgetting % | Δ PPL(A) After | Δ PPL(B) After | Δ LAMBADA After | Δ Rep-4 After | Δ Drift (JS) | Δ Total Hours | Δ Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|
| replay25 | 42 | +0.44% | +0.05 | -0.01 | +0.009 | +0.0257 | -0.0032 | -3.67h | +66 |
| mer25 | 42 | -22.39% | -2.39 | +0.98 | +0.026 | +0.0178 | -0.0120 | -5.53h | +117 |
| ewc | 42 | -36.95% | -3.94 | +0.14 | +0.004 | +0.1601 | +0.0096 | +9.59h | -3 |
| bandit_replay | 42 | -15.73% | -1.68 | +0.16 | +0.013 | -0.1067 | +0.0025 | -7.12h | +186 |
| rmgs | 42 | -28.24% | -3.02 | +1.76 | +0.024 | -0.0217 | -0.0140 | -8.03h | +231 |

## Pareto-Style Rankings

### Best Forgetting Reduction per Hour
1. **rmgs**: 2.429%/h (Δ=+28.24% in 11.63h)
2. **mer25**: 1.585%/h (Δ=+22.39% in 14.13h)
3. **ewc**: 1.264%/h (Δ=+36.95% in 29.24h)
4. **bandit_replay**: 1.254%/h (Δ=+15.73% in 12.54h)
5. **replay25**: -0.027%/h (Δ=-0.44% in 15.98h)

### Best Domain A Retention (Lowest Forgetting)
1. **ewc**: 8.85%
2. **rmgs**: 17.56%
3. **mer25**: 23.42%
4. **bandit_replay**: 30.07%
5. **replay25**: 46.24%

### Best General Ability (LAMBADA)
1. **mer25**: 0.476
2. **rmgs**: 0.474
3. **bandit_replay**: 0.462
4. **replay25**: 0.458
5. **ewc**: 0.453

### Best Throughput
1. **rmgs**: 567 tok/s
2. **bandit_replay**: 522 tok/s
3. **mer25**: 452 tok/s
4. **replay25**: 402 tok/s
5. **ewc**: 333 tok/s

---

## Llama 3.2 (1B)

*7 run(s) for this model family.*

## Executive Summary

**Completed Runs**: 7
**Methods Tested**: bandit_replay, baseline, ewc, mer25, replay25, rmgs
**Baseline Seeds**: 2

**Baseline Forgetting**: 44.84% ± 2.06% (n=2)

### Key Findings

- **Best Forgetting**: rmgs (20.78%, Δ=+24.07% vs baseline)
- **Fastest Method**: replay25 (10.96h)
- **Highest Throughput**: replay25 (568 tok/s)

## Full Comparison Table

| Run ID | Method | Seed | Forgetting % | PPL(A) After | PPL(B) After | LAMBADA After | Rep-4 After | Drift (JS) | Total Hours | Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3_rq1_baseline_s42 | baseline | 42 | 46.30% | 14.85 | 9.78 | 0.523 | 0.3281 | 0.2807 | 11.85h | 523 |
| llama3_rq1_baseline_s123 | baseline | 123 | 43.39% | 14.55 | 9.77 | 0.535 | 0.3814 | 0.2782 | 13.97h | 443 |
| llama3_rq2_replay25_s42 | replay25 | 42 | 47.80% | 15.01 | 9.78 | 0.511 | 0.2589 | 0.2815 | 10.96h | 568 |
| llama3_rq2_mer25_s42 | mer25 | 42 | 24.95% | 12.68 | 10.54 | 0.547 | 0.4545 | 0.2785 | 11.94h | 517 |
| llama3_rq2_ewc_s42 | ewc | 42 | 45.07% | 14.73 | 9.79 | 0.521 | 0.3597 | 0.2737 | 37.52h | 326 |
| llama3_rq4_bandit_replay_s42 | bandit_replay | 42 | 23.06% | 12.50 | 9.90 | 0.457 | 0.0731 | 0.2560 | 11.16h | 556 |
| llama3_rq4_rmgs_s42 | rmgs | 42 | 20.78% | 12.26 | 11.08 | 0.552 | 0.4565 | 0.2717 | 11.13h | 557 |

## Baseline Variance Analysis

| Metric | Mean | Std | Min | Max | N |
|--------|------|-----|-----|-----|---|
| Forgetting % | 44.84% | 2.06 | 43.39% | 46.30% | 2 |
| PPL(A) After | 14.70 | 0.21 | 14.55 | 14.85 | 2 |
| PPL(B) After | 9.77 | 0.01 | 9.77 | 9.78 | 2 |
| LAMBADA After | 0.529 | 0.008 | 0.523 | 0.535 | 2 |
| Rep-4 After | 0.3547 | 0.0377 | 0.3281 | 0.3814 | 2 |
| Drift (JS) | 0.2795 | 0.0017 | 0.2782 | 0.2807 | 2 |
| Total Hours | 12.91h | 1.50 | 11.85h | 13.97h | 2 |
| Avg Tok/s | 483 | 57 | 443 | 523 | 2 |

## Method Deltas vs Baseline Mean

*Negative delta = improvement for lower-is-better metrics (forgetting, PPL, rep, drift, time)*
*Positive delta = improvement for higher-is-better metrics (LAMBADA, tok/s)*

| Method | Seed | Δ Forgetting % | Δ PPL(A) After | Δ PPL(B) After | Δ LAMBADA After | Δ Rep-4 After | Δ Drift (JS) | Δ Total Hours | Δ Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|
| replay25 | 42 | +2.96% | +0.30 | +0.01 | -0.018 | -0.0958 | +0.0021 | -1.95h | +85 |
| mer25 | 42 | -19.89% | -2.02 | +0.77 | +0.018 | +0.0998 | -0.0010 | -0.97h | +34 |
| ewc | 42 | +0.23% | +0.03 | +0.02 | -0.008 | +0.0049 | -0.0057 | +24.61h | -157 |
| bandit_replay | 42 | -21.78% | -2.21 | +0.13 | -0.072 | -0.2816 | -0.0235 | -1.75h | +73 |
| rmgs | 42 | -24.07% | -2.44 | +1.30 | +0.023 | +0.1018 | -0.0078 | -1.78h | +74 |

## Pareto-Style Rankings

### Best Forgetting Reduction per Hour
1. **rmgs**: 2.162%/h (Δ=+24.07% in 11.13h)
2. **bandit_replay**: 1.952%/h (Δ=+21.78% in 11.16h)
3. **mer25**: 1.666%/h (Δ=+19.89% in 11.94h)
4. **ewc**: -0.006%/h (Δ=-0.23% in 37.52h)
5. **replay25**: -0.270%/h (Δ=-2.96% in 10.96h)

### Best Domain A Retention (Lowest Forgetting)
1. **rmgs**: 20.78%
2. **bandit_replay**: 23.06%
3. **mer25**: 24.95%
4. **ewc**: 45.07%
5. **replay25**: 47.80%

### Best General Ability (LAMBADA)
1. **rmgs**: 0.552
2. **mer25**: 0.547
3. **ewc**: 0.521
4. **replay25**: 0.511
5. **bandit_replay**: 0.457

### Best Throughput
1. **replay25**: 568 tok/s
2. **rmgs**: 557 tok/s
3. **bandit_replay**: 556 tok/s
4. **mer25**: 517 tok/s
5. **ewc**: 326 tok/s

---

## unknown

*1 run(s) for this model family.*

## Executive Summary

**Completed Runs**: 1
**Methods Tested**: 
**Baseline Seeds**: 0


### Key Findings


## Full Comparison Table

| Run ID | Method | Seed | Forgetting % | PPL(A) After | PPL(B) After | LAMBADA After | Rep-4 After | Drift (JS) | Total Hours | Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3_rq2_replay25_s42_cuda |  |  | 14.39% | 16.54 | 11.46 | 0.366 | 0.1877 | 0.2897 | 3.63h | 3809 |

## Baseline Variance Analysis

*No baseline runs available for variance analysis.*

## Method Deltas vs Baseline Mean

*No method comparisons available.*

## Pareto-Style Rankings

### Best Forgetting Reduction per Hour
1. ****: -3.962%/h (Δ=-14.39% in 3.63h)

### Best Domain A Retention (Lowest Forgetting)
1. ****: 14.39%

### Best General Ability (LAMBADA)
1. ****: 0.366

### Best Throughput
1. ****: 3809 tok/s

---

## Anomalies Summary

| Anomaly | Count | Affected Runs |
|---------|-------|---------------|
| oom_recovery | 1 | qwen3_rq2_replay25_s42_cuda |

---

## Appendix: Artifact Paths

### gpt2_pilot_baseline_s0

- **Metrics**: `experiments/runs/gpt2_pilot_baseline_s0/metrics.json`
- **Runpack**: `experiments/runs/gpt2_pilot_baseline_s0/runpack_gpt2_pilot_baseline_s0.md`
- **Config**: `experiments/runs/gpt2_pilot_baseline_s0/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gpt2_pilot_baseline_s0/checkpoints/theta_A.pt`
  - `experiments/runs/gpt2_pilot_baseline_s0/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gpt2_pilot_baseline_s0/training_log.jsonl`

### gpt2_rq1_baseline_s42

- **Metrics**: `experiments/runs/gpt2_rq1_baseline_s42/metrics.json`
- **Runpack**: `experiments/runs/gpt2_rq1_baseline_s42/runpack_gpt2_rq1_baseline_s42.md`
- **Config**: `experiments/runs/gpt2_rq1_baseline_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gpt2_rq1_baseline_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gpt2_rq1_baseline_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gpt2_rq1_baseline_s42/training_log.jsonl`

### gpt2_rq1_baseline_s123

- **Metrics**: `experiments/runs/gpt2_rq1_baseline_s123/metrics.json`
- **Runpack**: `experiments/runs/gpt2_rq1_baseline_s123/runpack_gpt2_rq1_baseline_s123.md`
- **Config**: `experiments/runs/gpt2_rq1_baseline_s123/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gpt2_rq1_baseline_s123/checkpoints/theta_A.pt`
  - `experiments/runs/gpt2_rq1_baseline_s123/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gpt2_rq1_baseline_s123/training_log.jsonl`

### gpt2_rq2_replay25_s42

- **Metrics**: `experiments/runs/gpt2_rq2_replay25_s42/metrics.json`
- **Runpack**: `experiments/runs/gpt2_rq2_replay25_s42/runpack_gpt2_rq2_replay25_s42.md`
- **Config**: `experiments/runs/gpt2_rq2_replay25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gpt2_rq2_replay25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gpt2_rq2_replay25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gpt2_rq2_replay25_s42/training_log.jsonl`

### gpt2_rq2_mer25_s42

- **Metrics**: `experiments/runs/gpt2_rq2_mer25_s42/metrics.json`
- **Runpack**: `experiments/runs/gpt2_rq2_mer25_s42/runpack_gpt2_rq2_mer25_s42.md`
- **Config**: `experiments/runs/gpt2_rq2_mer25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gpt2_rq2_mer25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gpt2_rq2_mer25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gpt2_rq2_mer25_s42/training_log.jsonl`

### gpt2_rq2_ewc_s42

- **Metrics**: `experiments/runs/gpt2_rq2_ewc_s42/metrics.json`
- **Runpack**: `experiments/runs/gpt2_rq2_ewc_s42/runpack_gpt2_rq2_ewc_s42.md`
- **Config**: `experiments/runs/gpt2_rq2_ewc_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gpt2_rq2_ewc_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gpt2_rq2_ewc_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gpt2_rq2_ewc_s42/training_log.jsonl`

### qwen3_rq1_baseline_s42

- **Metrics**: `experiments/runs/qwen3_rq1_baseline_s42/metrics.json`
- **Runpack**: `experiments/runs/qwen3_rq1_baseline_s42/runpack_qwen3_rq1_baseline_s42.md`
- **Config**: `experiments/runs/qwen3_rq1_baseline_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/qwen3_rq1_baseline_s42/checkpoints/theta_A.pt`
  - `experiments/runs/qwen3_rq1_baseline_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/qwen3_rq1_baseline_s42/training_log.jsonl`

### qwen3_rq1_baseline_s123

- **Metrics**: `experiments/runs/qwen3_rq1_baseline_s123/metrics.json`
- **Runpack**: `experiments/runs/qwen3_rq1_baseline_s123/runpack_qwen3_rq1_baseline_s123.md`
- **Config**: `experiments/runs/qwen3_rq1_baseline_s123/config.yaml`
- **Checkpoints**:
  - `experiments/runs/qwen3_rq1_baseline_s123/checkpoints/theta_A.pt`
  - `experiments/runs/qwen3_rq1_baseline_s123/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/qwen3_rq1_baseline_s123/training_log.jsonl`

### qwen3_rq2_replay25_s42

- **Metrics**: `experiments/runs/qwen3_rq2_replay25_s42/metrics.json`
- **Runpack**: `experiments/runs/qwen3_rq2_replay25_s42/runpack_qwen3_rq2_replay25_s42.md`
- **Config**: `experiments/runs/qwen3_rq2_replay25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/qwen3_rq2_replay25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/qwen3_rq2_replay25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/qwen3_rq2_replay25_s42/training_log.jsonl`

### qwen3_rq2_mer25_s42

- **Metrics**: `experiments/runs/qwen3_rq2_mer25_s42/metrics.json`
- **Runpack**: `experiments/runs/qwen3_rq2_mer25_s42/runpack_qwen3_rq2_mer25_s42.md`
- **Config**: `experiments/runs/qwen3_rq2_mer25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/qwen3_rq2_mer25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/qwen3_rq2_mer25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/qwen3_rq2_mer25_s42/training_log.jsonl`

### qwen3_rq2_ewc_s42

- **Metrics**: `experiments/runs/qwen3_rq2_ewc_s42/metrics.json`
- **Runpack**: `experiments/runs/qwen3_rq2_ewc_s42/runpack_qwen3_rq2_ewc_s42.md`
- **Config**: `experiments/runs/qwen3_rq2_ewc_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/qwen3_rq2_ewc_s42/checkpoints/theta_A.pt`
  - `experiments/runs/qwen3_rq2_ewc_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/qwen3_rq2_ewc_s42/training_log.jsonl`

### gemma3_rq1_baseline_s42

- **Metrics**: `experiments/runs/gemma3_rq1_baseline_s42/metrics.json`
- **Runpack**: `experiments/runs/gemma3_rq1_baseline_s42/runpack_gemma3_rq1_baseline_s42.md`
- **Config**: `experiments/runs/gemma3_rq1_baseline_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gemma3_rq1_baseline_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gemma3_rq1_baseline_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gemma3_rq1_baseline_s42/training_log.jsonl`

### gemma3_rq1_baseline_s123

- **Metrics**: `experiments/runs/gemma3_rq1_baseline_s123/metrics.json`
- **Runpack**: `experiments/runs/gemma3_rq1_baseline_s123/runpack_gemma3_rq1_baseline_s123.md`
- **Config**: `experiments/runs/gemma3_rq1_baseline_s123/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gemma3_rq1_baseline_s123/checkpoints/theta_A.pt`
  - `experiments/runs/gemma3_rq1_baseline_s123/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gemma3_rq1_baseline_s123/training_log.jsonl`

### gemma3_rq2_replay25_s42

- **Metrics**: `experiments/runs/gemma3_rq2_replay25_s42/metrics.json`
- **Runpack**: `experiments/runs/gemma3_rq2_replay25_s42/runpack_gemma3_rq2_replay25_s42.md`
- **Config**: `experiments/runs/gemma3_rq2_replay25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gemma3_rq2_replay25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gemma3_rq2_replay25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gemma3_rq2_replay25_s42/training_log.jsonl`

### gemma3_rq2_mer25_s42

- **Metrics**: `experiments/runs/gemma3_rq2_mer25_s42/metrics.json`
- **Runpack**: `experiments/runs/gemma3_rq2_mer25_s42/runpack_gemma3_rq2_mer25_s42.md`
- **Config**: `experiments/runs/gemma3_rq2_mer25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gemma3_rq2_mer25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gemma3_rq2_mer25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gemma3_rq2_mer25_s42/training_log.jsonl`

### gemma3_rq2_ewc_s42

- **Metrics**: `experiments/runs/gemma3_rq2_ewc_s42/metrics.json`
- **Runpack**: `experiments/runs/gemma3_rq2_ewc_s42/runpack_gemma3_rq2_ewc_s42.md`
- **Config**: `experiments/runs/gemma3_rq2_ewc_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gemma3_rq2_ewc_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gemma3_rq2_ewc_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gemma3_rq2_ewc_s42/training_log.jsonl`

### llama3_rq1_baseline_s42

- **Metrics**: `experiments/runs/llama3_rq1_baseline_s42/metrics.json`
- **Runpack**: `experiments/runs/llama3_rq1_baseline_s42/runpack_llama3_rq1_baseline_s42.md`
- **Config**: `experiments/runs/llama3_rq1_baseline_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/llama3_rq1_baseline_s42/checkpoints/theta_A.pt`
  - `experiments/runs/llama3_rq1_baseline_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/llama3_rq1_baseline_s42/training_log.jsonl`

### llama3_rq1_baseline_s123

- **Metrics**: `experiments/runs/llama3_rq1_baseline_s123/metrics.json`
- **Runpack**: `experiments/runs/llama3_rq1_baseline_s123/runpack_llama3_rq1_baseline_s123.md`
- **Config**: `experiments/runs/llama3_rq1_baseline_s123/config.yaml`
- **Checkpoints**:
  - `experiments/runs/llama3_rq1_baseline_s123/checkpoints/theta_A.pt`
  - `experiments/runs/llama3_rq1_baseline_s123/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/llama3_rq1_baseline_s123/training_log.jsonl`

### llama3_rq2_replay25_s42

- **Metrics**: `experiments/runs/llama3_rq2_replay25_s42/metrics.json`
- **Runpack**: `experiments/runs/llama3_rq2_replay25_s42/runpack_llama3_rq2_replay25_s42.md`
- **Config**: `experiments/runs/llama3_rq2_replay25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/llama3_rq2_replay25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/llama3_rq2_replay25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/llama3_rq2_replay25_s42/training_log.jsonl`

### llama3_rq2_mer25_s42

- **Metrics**: `experiments/runs/llama3_rq2_mer25_s42/metrics.json`
- **Runpack**: `experiments/runs/llama3_rq2_mer25_s42/runpack_llama3_rq2_mer25_s42.md`
- **Config**: `experiments/runs/llama3_rq2_mer25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/llama3_rq2_mer25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/llama3_rq2_mer25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/llama3_rq2_mer25_s42/training_log.jsonl`

### llama3_rq2_ewc_s42

- **Metrics**: `experiments/runs/llama3_rq2_ewc_s42/metrics.json`
- **Runpack**: `experiments/runs/llama3_rq2_ewc_s42/runpack_llama3_rq2_ewc_s42.md`
- **Config**: `experiments/runs/llama3_rq2_ewc_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/llama3_rq2_ewc_s42/checkpoints/theta_A.pt`
  - `experiments/runs/llama3_rq2_ewc_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/llama3_rq2_ewc_s42/training_log.jsonl`

### gpt2_rq3_bandit_replay_s42

- **Metrics**: `experiments/runs/gpt2_rq3_bandit_replay_s42/metrics.json`
- **Runpack**: `experiments/runs/gpt2_rq3_bandit_replay_s42/runpack_gpt2_rq3_bandit_replay_s42.md`
- **Config**: `experiments/runs/gpt2_rq3_bandit_replay_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gpt2_rq3_bandit_replay_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gpt2_rq3_bandit_replay_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gpt2_rq3_bandit_replay_s42/training_log.jsonl`

### gpt2_rq3_rmgs_s42

- **Metrics**: `experiments/runs/gpt2_rq3_rmgs_s42/metrics.json`
- **Runpack**: `experiments/runs/gpt2_rq3_rmgs_s42/runpack_gpt2_rq3_rmgs_s42.md`
- **Config**: `experiments/runs/gpt2_rq3_rmgs_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gpt2_rq3_rmgs_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gpt2_rq3_rmgs_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gpt2_rq3_rmgs_s42/training_log.jsonl`

### qwen3_rq4_bandit_replay_s42

- **Metrics**: `experiments/runs/qwen3_rq4_bandit_replay_s42/metrics.json`
- **Runpack**: `experiments/runs/qwen3_rq4_bandit_replay_s42/runpack_qwen3_rq4_bandit_replay_s42.md`
- **Config**: `experiments/runs/qwen3_rq4_bandit_replay_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/qwen3_rq4_bandit_replay_s42/checkpoints/theta_A.pt`
  - `experiments/runs/qwen3_rq4_bandit_replay_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/qwen3_rq4_bandit_replay_s42/training_log.jsonl`

### qwen3_rq4_rmgs_s42

- **Metrics**: `experiments/runs/qwen3_rq4_rmgs_s42/metrics.json`
- **Runpack**: `experiments/runs/qwen3_rq4_rmgs_s42/runpack_qwen3_rq4_rmgs_s42.md`
- **Config**: `experiments/runs/qwen3_rq4_rmgs_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/qwen3_rq4_rmgs_s42/checkpoints/theta_A.pt`
  - `experiments/runs/qwen3_rq4_rmgs_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/qwen3_rq4_rmgs_s42/training_log.jsonl`

### gemma3_rq4_bandit_replay_s42

- **Metrics**: `experiments/runs/gemma3_rq4_bandit_replay_s42/metrics.json`
- **Runpack**: `experiments/runs/gemma3_rq4_bandit_replay_s42/runpack_gemma3_rq4_bandit_replay_s42.md`
- **Config**: `experiments/runs/gemma3_rq4_bandit_replay_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gemma3_rq4_bandit_replay_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gemma3_rq4_bandit_replay_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gemma3_rq4_bandit_replay_s42/training_log.jsonl`

### gemma3_rq4_rmgs_s42

- **Metrics**: `experiments/runs/gemma3_rq4_rmgs_s42/metrics.json`
- **Runpack**: `experiments/runs/gemma3_rq4_rmgs_s42/runpack_gemma3_rq4_rmgs_s42.md`
- **Config**: `experiments/runs/gemma3_rq4_rmgs_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/gemma3_rq4_rmgs_s42/checkpoints/theta_A.pt`
  - `experiments/runs/gemma3_rq4_rmgs_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/gemma3_rq4_rmgs_s42/training_log.jsonl`

### llama3_rq4_bandit_replay_s42

- **Metrics**: `experiments/runs/llama3_rq4_bandit_replay_s42/metrics.json`
- **Runpack**: `experiments/runs/llama3_rq4_bandit_replay_s42/runpack_llama3_rq4_bandit_replay_s42.md`
- **Config**: `experiments/runs/llama3_rq4_bandit_replay_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/llama3_rq4_bandit_replay_s42/checkpoints/theta_A.pt`
  - `experiments/runs/llama3_rq4_bandit_replay_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/llama3_rq4_bandit_replay_s42/training_log.jsonl`

### llama3_rq4_rmgs_s42

- **Metrics**: `experiments/runs/llama3_rq4_rmgs_s42/metrics.json`
- **Runpack**: `experiments/runs/llama3_rq4_rmgs_s42/runpack_llama3_rq4_rmgs_s42.md`
- **Config**: `experiments/runs/llama3_rq4_rmgs_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/llama3_rq4_rmgs_s42/checkpoints/theta_A.pt`
  - `experiments/runs/llama3_rq4_rmgs_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/llama3_rq4_rmgs_s42/training_log.jsonl`

### qwen3_rq2_replay25_s42_cuda

- **Metrics**: `experiments/runs/qwen3_rq2_replay25_s42_cuda/metrics.json`
- **Runpack**: `experiments/runs/qwen3_rq2_replay25_s42_cuda/runpack_qwen3_rq2_replay25_s42_cuda.md`
- **Config**: `experiments/runs/qwen3_rq2_replay25_s42_cuda/config.yaml`
- **Checkpoints**:
  - `experiments/runs/qwen3_rq2_replay25_s42_cuda/checkpoints/theta_A.pt`
  - `experiments/runs/qwen3_rq2_replay25_s42_cuda/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/qwen3_rq2_replay25_s42_cuda/training_log.jsonl`
