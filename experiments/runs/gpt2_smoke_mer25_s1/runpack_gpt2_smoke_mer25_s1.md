# Runpack: gpt2_smoke_mer25_s1

**Version**: 1.0
**Date**: 2026-02-11

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | gpt2_smoke_mer25_s1 |
| **Method** | mer25 |
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
| After Domain B (final) | 30.44 | 48.75 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | -2.42% |

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
| **Drift (JS divergence)** | 0.2153 | 0.1980 |
| Vocab Overlap | 0.3641 | 0.3892 |

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
| **Total Wall Time** | 0.03 hours |
| Domain A Time | 0.01 hours |
| Domain B Time | 0.01 hours |
| **Peak VRAM** | N/A GB |
| **Peak RAM** | 0.91 GB |
| Avg Tokens/sec | 3211 |

---

## Method Parameters

{
  "buffer_size_pct": 10,
  "replay_rate": 0.25,
  "reptile_epsilon": 0.1,
  "reptile_interval": 10
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
| Config | experiments/runs/gpt2_smoke_mer25_s1/config_merged.yaml |
| Metrics JSON | experiments/runs/gpt2_smoke_mer25_s1/metrics.json |
| Domain A Checkpoint | experiments/runs/gpt2_smoke_mer25_s1/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/gpt2_smoke_mer25_s1/checkpoints/theta_AB.pt |

---

**Generated**: 2026-02-11T23:14:02.204046
