# Runpack: smoke_ewc_s1

**Version**: 1.0
**Date**: 2026-02-06

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | smoke_ewc_s1 |
| **Method** | ewc |
| **Seed** | 1 |
| **Research Question** | SMOKE |
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
| Init (pre-training) | 33.41 | 54.90 |
| After Domain A | 31.20 | — |
| After Domain B (final) | 30.71 | 48.39 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | -1.56% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | 0.0082 | 0.0246 |
| Rep-8 | 0.0000 | 0.0000 |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | 0.2153 | 0.2009 |
| Vocab Overlap | 0.3641 | 0.3839 |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | 0.3200 |
| After CPT | 0.3200 |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | 0.06 hours |
| Domain A Time | 0.01 hours |
| Domain B Time | 0.01 hours |
| **Peak VRAM** | N/A GB |
| **Peak RAM** | 1.93 GB |
| Avg Tokens/sec | 2953 |

---

## Method Parameters

{
  "ewc_lambda": 100,
  "fisher_samples": 100,
  "fisher_type": "diagonal"
}

---

## Anomalies

- None

---

## Notes

Pilot baseline run completed successfully.

---

## Artifacts

| Artifact | Path |
|----------|------|
| Config | configs/smoke/ewc_smoke.yaml |
| Metrics JSON | experiments/runs/smoke_ewc_s1/metrics.json |
| Domain A Checkpoint | experiments/runs/smoke_ewc_s1/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/smoke_ewc_s1/checkpoints/theta_AB.pt |

---

**Generated**: 2026-02-06T17:18:55.409028
