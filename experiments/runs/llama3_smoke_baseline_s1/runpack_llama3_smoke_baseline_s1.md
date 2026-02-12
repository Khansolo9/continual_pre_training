# Runpack: llama3_smoke_baseline_s1

**Version**: 1.0
**Date**: 2026-02-11

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | llama3_smoke_baseline_s1 |
| **Method** | baseline |
| **Seed** | 1 |
| **Research Question** | RQ1 |
| **Status** | completed |

---

## Dataset Configuration

| Field | Value |
|-------|-------|
| **Domain A** | wikitext-103 |
| **Domain A Token Tier** | A-medium |
| **Domain A Tokens Used** | 10,000,000 |
| **Domain A Hash/ID** | sha256:ccdaa25c64ccf0d1 |
| **Domain B** | arxiv_abstracts |
| **Domain B Token Tier** | B-medium |
| **Domain B Tokens Used** | 10,000,000 |
| **Domain B Hash/ID** | sha256:7e5b6ec051938d7f |

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
| Init (pre-training) | 11.74 | 16.83 |
| After Domain A | 10.76 | — |
| After Domain B (final) | 10.72 | 14.71 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | -0.36% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | 0.0410 | 0.0164 |
| Rep-8 | 0.0000 | 0.0000 |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | 0.0843 | 0.1136 |
| Vocab Overlap | 0.6585 | 0.5709 |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | 0.5400 |
| After CPT | 0.5400 |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | 0.18 hours |
| Domain A Time | 0.06 hours |
| Domain B Time | 0.05 hours |
| **Peak VRAM** | N/A GB |
| **Peak RAM** | 0.32 GB |
| Avg Tokens/sec | 503 |

---

## Method Parameters

None (baseline method)

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
| Config | experiments/runs/llama3_smoke_baseline_s1/config_merged.yaml |
| Metrics JSON | experiments/runs/llama3_smoke_baseline_s1/metrics.json |
| Domain A Checkpoint | experiments/runs/llama3_smoke_baseline_s1/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/llama3_smoke_baseline_s1/checkpoints/theta_AB.pt |

---

**Generated**: 2026-02-11T21:51:52.385564
