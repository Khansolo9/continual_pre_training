# Runpack: rq2_mer25_s42

**Version**: 1.0
**Date**: 2026-02-06

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | rq2_mer25_s42 |
| **Method** | mer25 |
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
| After Domain B (final) | 23.95 | 26.77 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | 4.47% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | 0.4565 | 0.4664 |
| Rep-8 | 0.3153 | 0.3273 |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | 0.4230 | 0.3140 |
| Vocab Overlap | 0.2437 | 0.2665 |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | 0.2030 |
| After CPT | 0.2210 |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | 3.00 hours |
| Domain A Time | 0.94 hours |
| Domain B Time | 0.96 hours |
| **Peak VRAM** | N/A GB |
| **Peak RAM** | 1.15 GB |
| Avg Tokens/sec | 2927 |

---

## Method Parameters

{
  "replay_rate": 0.25,
  "buffer_size_pct": 10,
  "reptile_interval": 100,
  "reptile_epsilon": 0.1
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
| Config | configs/methods/mer25.yaml |
| Metrics JSON | experiments/runs/rq2_mer25_s42/metrics.json |
| Domain A Checkpoint | experiments/runs/rq2_mer25_s42/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/rq2_mer25_s42/checkpoints/theta_AB.pt |

---

**Generated**: 2026-02-06T23:50:31.958737
