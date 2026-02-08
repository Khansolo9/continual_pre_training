# Experiment Summary Pack

**Generated**: 2026-02-07T20:25:21Z
**Total Completed Runs**: 6

---

## Executive Summary

**Completed Runs**: 6
**Methods Tested**: baseline, ewc, mer25, replay25
**Baseline Seeds**: 3

**Baseline Forgetting**: 87.00% ± 0.34% (n=3)

### Key Findings

- **Best Forgetting**: mer25 (4.47%, Δ=+82.53% vs baseline)
- **Fastest Method**: replay25 (2.99h)
- **Highest Throughput**: replay25 (2996 tok/s)

---

## Full Comparison Table

| Run ID | Method | Seed | Forgetting % | PPL(A) After | PPL(B) After | LAMBADA After | Rep-4 After | Drift (JS) | Total Hours | Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| pilot_baseline_s0 | baseline | 0 | 87.35% | 43.00 | 21.17 | 0.131 | 0.5810 | 0.3428 | 3.04h | 3045 |
| rq1_baseline_s42 | baseline | 42 | 86.95% | 42.87 | 21.17 | 0.217 | 0.6206 | 0.3372 | 3.34h | 2644 |
| rq1_baseline_s123 | baseline | 123 | 86.68% | 42.84 | 21.17 | 0.216 | 0.6324 | 0.3370 | 3.17h | 2861 |
| rq2_replay25_s42 | replay25 | 42 | 7.48% | 24.64 | 22.06 | 0.207 | 0.4130 | 0.3613 | 2.99h | 2996 |
| rq2_mer25_s42 | mer25 | 42 | 4.47% | 23.95 | 26.77 | 0.221 | 0.4664 | 0.3140 | 3.00h | 2927 |
| rq2_ewc_s42 | ewc | 42 | 33.44% | 30.60 | 22.08 | 0.246 | 0.6206 | 0.3363 | 3.16h | 2799 |

---

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

---

## Method Deltas vs Baseline Mean

*Negative delta = improvement for lower-is-better metrics (forgetting, PPL, rep, drift, time)*
*Positive delta = improvement for higher-is-better metrics (LAMBADA, tok/s)*

| Method | Seed | Δ Forgetting % | Δ PPL(A) After | Δ PPL(B) After | Δ LAMBADA After | Δ Rep-4 After | Δ Drift (JS) | Δ Total Hours | Δ Avg Tok/s |
|---|---|---|---|---|---|---|---|---|---|
| replay25 | 42 | -79.52% | -18.26 | +0.89 | +0.019 | -0.1983 | +0.0223 | -0.19h | +146 |
| mer25 | 42 | -82.53% | -18.95 | +5.60 | +0.033 | -0.1449 | -0.0250 | -0.18h | +77 |
| ewc | 42 | -53.55% | -12.30 | +0.90 | +0.058 | +0.0092 | -0.0027 | -0.02h | -51 |

---

## Pareto-Style Rankings

### Best Forgetting Reduction per Hour
1. **mer25**: 27.465%/h (Δ=+82.53% in 3.00h)
2. **replay25**: 26.568%/h (Δ=+79.52% in 2.99h)
3. **ewc**: 16.954%/h (Δ=+53.55% in 3.16h)

### Best Domain A Retention (Lowest Forgetting)
1. **mer25**: 4.47%
2. **replay25**: 7.48%
3. **ewc**: 33.44%

### Best General Ability (LAMBADA)
1. **ewc**: 0.246
2. **mer25**: 0.221
3. **replay25**: 0.207

### Best Throughput
1. **replay25**: 2996 tok/s
2. **mer25**: 2927 tok/s
3. **ewc**: 2799 tok/s

---

## Anomalies Summary

| Anomaly | Count | Affected Runs |
|---------|-------|---------------|
| high_rep4 | 6 | pilot_baseline_s0, rq1_baseline_s42, rq1_baseline_s123, rq2_replay25_s42, rq2_mer25_s42 (+1 more) |

---

## Appendix: Artifact Paths

### pilot_baseline_s0

- **Metrics**: `experiments/runs/pilot_baseline_s0/metrics.json`
- **Runpack**: `experiments/runs/pilot_baseline_s0/runpack_pilot_baseline_s0.md`
- **Config**: `experiments/runs/pilot_baseline_s0/config.yaml`
- **Checkpoints**:
  - `experiments/runs/pilot_baseline_s0/checkpoints/theta_A.pt`
  - `experiments/runs/pilot_baseline_s0/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/pilot_baseline_s0/training_log.jsonl`

### rq1_baseline_s42

- **Metrics**: `experiments/runs/rq1_baseline_s42/metrics.json`
- **Runpack**: `experiments/runs/rq1_baseline_s42/runpack_rq1_baseline_s42.md`
- **Config**: `experiments/runs/rq1_baseline_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/rq1_baseline_s42/checkpoints/theta_A.pt`
  - `experiments/runs/rq1_baseline_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/rq1_baseline_s42/training_log.jsonl`

### rq1_baseline_s123

- **Metrics**: `experiments/runs/rq1_baseline_s123/metrics.json`
- **Runpack**: `experiments/runs/rq1_baseline_s123/runpack_rq1_baseline_s123.md`
- **Config**: `experiments/runs/rq1_baseline_s123/config.yaml`
- **Checkpoints**:
  - `experiments/runs/rq1_baseline_s123/checkpoints/theta_A.pt`
  - `experiments/runs/rq1_baseline_s123/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/rq1_baseline_s123/training_log.jsonl`

### rq2_replay25_s42

- **Metrics**: `experiments/runs/rq2_replay25_s42/metrics.json`
- **Runpack**: `experiments/runs/rq2_replay25_s42/runpack_rq2_replay25_s42.md`
- **Config**: `experiments/runs/rq2_replay25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/rq2_replay25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/rq2_replay25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/rq2_replay25_s42/training_log.jsonl`

### rq2_mer25_s42

- **Metrics**: `experiments/runs/rq2_mer25_s42/metrics.json`
- **Runpack**: `experiments/runs/rq2_mer25_s42/runpack_rq2_mer25_s42.md`
- **Config**: `experiments/runs/rq2_mer25_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/rq2_mer25_s42/checkpoints/theta_A.pt`
  - `experiments/runs/rq2_mer25_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/rq2_mer25_s42/training_log.jsonl`

### rq2_ewc_s42

- **Metrics**: `experiments/runs/rq2_ewc_s42/metrics.json`
- **Runpack**: `experiments/runs/rq2_ewc_s42/runpack_rq2_ewc_s42.md`
- **Config**: `experiments/runs/rq2_ewc_s42/config.yaml`
- **Checkpoints**:
  - `experiments/runs/rq2_ewc_s42/checkpoints/theta_A.pt`
  - `experiments/runs/rq2_ewc_s42/checkpoints/theta_AB.pt`
- **Training Log**: `experiments/runs/rq2_ewc_s42/training_log.jsonl`
