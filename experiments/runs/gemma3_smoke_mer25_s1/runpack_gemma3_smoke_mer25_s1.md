# Runpack: gemma3_smoke_mer25_s1

**Version**: 1.0
**Date**: 2026-02-12

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | gemma3_smoke_mer25_s1 |
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
| **Domain A Hash/ID** | sha256:3158dee9599cc2cf |
| **Domain B** | arxiv_abstracts |
| **Domain B Token Tier** | B-medium |
| **Domain B Tokens Used** | 10,000,000 |
| **Domain B Hash/ID** | sha256:9974542f257fe673 |

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
| Init (pre-training) | 26.95 | 52.51 |
| After Domain A | 14.88 | — |
| After Domain B (final) | 13.68 | 20.87 |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | -8.05% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | 0.0000 | 0.0000 |
| Rep-8 | 0.0000 | 0.0000 |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | 0.1632 | 0.1538 |
| Vocab Overlap | 0.3774 | 0.4214 |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | 0.3400 |
| After CPT | 0.3400 |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | 0.24 hours |
| Domain A Time | 0.08 hours |
| Domain B Time | 0.08 hours |
| **Peak VRAM** | N/A GB |
| **Peak RAM** | 0.81 GB |
| Avg Tokens/sec | 358 |

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
| Config | experiments/runs/gemma3_smoke_mer25_s1/config_merged.yaml |
| Metrics JSON | experiments/runs/gemma3_smoke_mer25_s1/metrics.json |
| Domain A Checkpoint | experiments/runs/gemma3_smoke_mer25_s1/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/gemma3_smoke_mer25_s1/checkpoints/theta_AB.pt |

---

**Generated**: 2026-02-12T05:03:14.240416
