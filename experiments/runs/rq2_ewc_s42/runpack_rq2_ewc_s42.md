# Runpack: rq2_ewc_s42

**Version**: 1.0
**Date**: 2026-02-07

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | rq2_ewc_s42 |
| **Method** | ewc |
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
| After Domain B (final) | 30.60 | 22.08 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | 33.44% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | 0.4723 | 0.6206 |
| Rep-8 | 0.3213 | 0.5060 |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | 0.4207 | 0.3363 |
| Vocab Overlap | 0.2511 | 0.2211 |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | 0.2020 |
| After CPT | 0.2460 |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | 3.16 hours |
| Domain A Time | 0.96 hours |
| Domain B Time | 1.03 hours |
| **Peak VRAM** | N/A GB |
| **Peak RAM** | 1.42 GB |
| Avg Tokens/sec | 2799 |

---

## Method Parameters

{
  "ewc_lambda": 100,
  "fisher_samples": 1000,
  "fisher_type": "diagonal"
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
| Config | configs/methods/ewc.yaml |
| Metrics JSON | experiments/runs/rq2_ewc_s42/metrics.json |
| Domain A Checkpoint | experiments/runs/rq2_ewc_s42/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/rq2_ewc_s42/checkpoints/theta_AB.pt |

---

**Generated**: 2026-02-07T13:23:01.467316
