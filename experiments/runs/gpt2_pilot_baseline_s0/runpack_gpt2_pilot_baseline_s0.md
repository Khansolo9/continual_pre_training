# Runpack: gpt2_pilot_baseline_s0

**Version**: 1.0
**Date**: 2026-01-30

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | gpt2_pilot_baseline_s0 |
| **Method** | baseline |
| **Seed** | 0 |
| **Research Question** | RQ0 |
| **Status** | completed |

---

## Dataset Configuration

| Field | Value |
|-------|-------|
| **Domain A** | wikitext-103 |
| **Domain A Token Tier** | A-medium |
| **Domain A Tokens Used** | 10,000,000 |
| **Domain A Hash/ID** | sha256:b8a88374cf6b4619 |
| **Domain B** | arxiv_abstracts |
| **Domain B Token Tier** | B-medium |
| **Domain B Tokens Used** | 10,000,000 |
| **Domain B Hash/ID** | sha256:aa8881464afd8e23 |

---

## Prompt Set Versions

| Set | Version |
|-----|---------|
| Drift Prompts | prompts_drift_v1.json |
| Quality Prompts | prompts_quality_v1.json |
| Toxicity Prompts | null |

---

## Primary Metrics

### Perplexity

| Checkpoint | PPL_A | PPL_B |
|------------|------:|------:|
| Init (pre-training) | 35.83 | 49.03 |
| After Domain A | 22.95 | — |
| After Domain B (final) | 43.00 | 21.17 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | 87.35% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | 0.4190 | 0.5810 |
| Rep-8 | 0.2671 | 0.4639 |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | 0.4199 | 0.3428 |
| Vocab Overlap | 0.2507 | 0.2018 |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | 0.1490 |
| After CPT | 0.1310 |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | 3.04 hours |
| Domain A Time | 0.88 hours |
| Domain B Time | 0.95 hours |
| **Peak VRAM** | 0.00 GB |
| **Peak RAM** | 0.71 GB |
| Avg Tokens/sec | 3045 |

---

## Method Parameters

None (baseline method)

---

## Anomalies

- high_rep4

---

## Notes

Pilot baseline run completed successfully.

---

## Artifacts

| Artifact | Path |
|----------|------|
| Config | configs/methods/baseline.yaml |
| Metrics JSON | experiments/runs/gpt2_pilot_baseline_s0/metrics.json |
| Domain A Checkpoint | experiments/runs/gpt2_pilot_baseline_s0/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/gpt2_pilot_baseline_s0/checkpoints/theta_AB.pt |

---

**Generated**: 2026-01-30T17:01:17.363638
