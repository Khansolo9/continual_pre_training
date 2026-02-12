# Runpack: gpt2_rq1_baseline_s42

**Version**: 1.0
**Date**: 2026-02-01

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | gpt2_rq1_baseline_s42 |
| **Method** | baseline |
| **Seed** | 42 |
| **Research Question** | RQ1 |
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
| After Domain A | 22.93 | — |
| After Domain B (final) | 42.87 | 21.17 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | 86.95% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | 0.4644 | 0.6206 |
| Rep-8 | 0.3233 | 0.5100 |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | 0.4208 | 0.3372 |
| Vocab Overlap | 0.2471 | 0.2111 |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | 0.2030 |
| After CPT | 0.2170 |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | 3.34 hours |
| Domain A Time | 0.97 hours |
| Domain B Time | 1.15 hours |
| **Peak VRAM** | N/A GB |
| **Peak RAM** | 1.18 GB |
| Avg Tokens/sec | 2644 |

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
| Metrics JSON | experiments/runs/gpt2_rq1_baseline_s42/metrics.json |
| Domain A Checkpoint | experiments/runs/gpt2_rq1_baseline_s42/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/gpt2_rq1_baseline_s42/checkpoints/theta_AB.pt |

---

**Generated**: 2026-02-01T18:00:03.633625
