# Runpack: rq2_replay25_s42

**Version**: 1.0
**Date**: 2026-02-06

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | rq2_replay25_s42 |
| **Method** | replay25 |
| **Seed** | 42 |
| **Research Question** | RQ2 |
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
| After Domain B (final) | 24.64 | 22.06 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | 7.48% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | 0.4605 | 0.4130 |
| Rep-8 | 0.3213 | 0.2671 |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | 0.4208 | 0.3613 |
| Vocab Overlap | 0.2471 | 0.2528 |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | 0.2030 |
| After CPT | 0.2070 |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | 2.99 hours |
| Domain A Time | 0.91 hours |
| Domain B Time | 0.94 hours |
| **Peak VRAM** | N/A GB |
| **Peak RAM** | 1.22 GB |
| Avg Tokens/sec | 2996 |

---

## Method Parameters

{
  "buffer_size_pct": 10,
  "mixing_ratio_replay": 0.25
}

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
| Config | configs/methods/replay25.yaml |
| Metrics JSON | experiments/runs/rq2_replay25_s42/metrics.json |
| Domain A Checkpoint | experiments/runs/rq2_replay25_s42/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/rq2_replay25_s42/checkpoints/theta_AB.pt |

---

**Generated**: 2026-02-06T20:36:38.171631
