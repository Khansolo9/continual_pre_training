# Runpack: qwen3_smoke_replay25_s1

**Version**: 1.0
**Date**: 2026-02-11

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | qwen3_smoke_replay25_s1 |
| **Method** | replay25 |
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
| **Domain A Hash/ID** | sha256:b6c0b8bcaeb0be65 |
| **Domain B** | arxiv_abstracts |
| **Domain B Token Tier** | B-medium |
| **Domain B Tokens Used** | 10,000,000 |
| **Domain B Hash/ID** | sha256:d04246ef71694528 |

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
| Init (pre-training) | 24.57 | 28.32 |
| After Domain A | 20.60 | — |
| After Domain B (final) | 19.79 | 22.78 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | -3.95% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | 0.0410 | 0.0082 |
| Rep-8 | 0.0000 | 0.0000 |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | 0.1020 | 0.1307 |
| Vocab Overlap | 0.4601 | 0.3889 |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | 0.3800 |
| After CPT | 0.3400 |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | 0.14 hours |
| Domain A Time | 0.04 hours |
| Domain B Time | 0.04 hours |
| **Peak VRAM** | N/A GB |
| **Peak RAM** | 0.75 GB |
| Avg Tokens/sec | 691 |

---

## Method Parameters

{
  "buffer_size_pct": 10,
  "mixing_ratio_replay": 0.25
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
| Config | experiments/runs/qwen3_smoke_replay25_s1/config_merged.yaml |
| Metrics JSON | experiments/runs/qwen3_smoke_replay25_s1/metrics.json |
| Domain A Checkpoint | experiments/runs/qwen3_smoke_replay25_s1/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/qwen3_smoke_replay25_s1/checkpoints/theta_AB.pt |

---

**Generated**: 2026-02-11T23:34:45.076446
