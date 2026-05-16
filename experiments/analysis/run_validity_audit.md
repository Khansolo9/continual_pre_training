# Run Validity Audit

**Generated**: 2026-05-15
**Last Updated**: 2026-05-15
**Status**: Active — single source of truth for run-cell validity post-Codex-Review-1.

> This document records which existing runs are **valid for which paper claims**, following the microbatch-rounding bug discovered by Codex Review 1 (2026-05-15) and the subsequent AMENDMENT-004 fix. Per project research-integrity standards, **no existing runs are deleted or overwritten**; this audit and the validity flags below are the canonical record of what survives and what is reproduced.

## Background

On 2026-05-15, independent peer review identified that [`src/cl_methods.py:500`](../../src/cl_methods.py#L500) computed replay sample count as:

```python
n_replay = int(batch_size * replay_rate)
```

For 1B-class models (`batch_size = 2`, `replay_rate = 0.25`), this evaluates to $\text{int}(0.5) = 0$ — **zero replay samples per microbatch**, silently. Training logs confirm `replay_samples: 0` throughout Domain B for all affected cells. GPT-2 (`batch_size = 4`) was unaffected: `int(4 × 0.25) = 1` real replay sample.

AMENDMENT-004 (drafted in [`docs/protocol/PROTOCOL_AMENDMENTS.md`](../../docs/protocol/PROTOCOL_AMENDMENTS.md)) fixes this with stochastic fractional rounding. Reruns are tracked in P2 Stages 1 and 3b.

## Validity flag dictionary

| Flag | Meaning |
|---|---|
| `valid_for_baseline_claims` | Run is a valid baseline measurement for its model. |
| `valid_for_ewc_claims` | Run is a valid measurement of EWC under the locked configuration. |
| `valid_for_replay25_claims` | Run is a valid measurement of fixed 25% replay. **Only GPT-2 cells qualify pre-AMENDMENT-004.** |
| `valid_for_mer25_claims` | Run is a valid measurement of MER (replay + Reptile). **Only GPT-2 cells qualify pre-AMENDMENT-004.** |
| `valid_for_bandit_replay_full_arm_set` | bandit_replay with all 5 arms producing intended replay counts. |
| `valid_for_rmgs_claims` | Run is a valid measurement of RMGS. |
| `invalid_for_replay25_claims_due_microbatch_rounding` | The cell ran but with `replay_samples: 0`; data not interpretable as 25% replay. |
| `invalid_for_mer25_replay_component_due_microbatch_rounding` | The cell ran but the replay component of MER produced zero samples; this is Reptile-without-replay, not MER. |
| `valid_as_reptile_without_replay` | Alternative interpretation of an invalid `mer25` 1B run — meaningful as "Reptile alone" but not as MER. |
| `bandit_replay_partial_validity` | bandit_replay ran but arms 0.0/0.1/0.25 produced zero replay (only 0.5 and 0.75 worked). Comparator interpretation invalid. |

## Per-run validity (29 non-smoke runs)

### GPT-2 (124M) — `batch_size = 4`, unaffected by microbatch rounding

| Run ID | Method | Flags |
|---|---|---|
| gpt2_pilot_baseline_s0 | baseline | valid_for_baseline_claims |
| gpt2_rq1_baseline_s42 | baseline | valid_for_baseline_claims |
| gpt2_rq1_baseline_s123 | baseline | valid_for_baseline_claims |
| gpt2_rq2_replay25_s42 | replay25 | valid_for_replay25_claims |
| gpt2_rq2_mer25_s42 | mer25 | valid_for_mer25_claims |
| gpt2_rq2_ewc_s42 | ewc | valid_for_ewc_claims |
| gpt2_rq3_bandit_replay_s42 | bandit_replay | valid_for_bandit_replay_full_arm_set |
| gpt2_rq3_rmgs_s42 | rmgs | valid_for_rmgs_claims |

### Qwen3 (0.6B) — `batch_size = 2`, affected

| Run ID | Method | Flags |
|---|---|---|
| qwen3_rq1_baseline_s42 | baseline | valid_for_baseline_claims |
| qwen3_rq1_baseline_s123 | baseline | valid_for_baseline_claims |
| qwen3_rq2_replay25_s42 | replay25 | **invalid_for_replay25_claims_due_microbatch_rounding** |
| qwen3_rq2_mer25_s42 | mer25 | **invalid_for_mer25_replay_component_due_microbatch_rounding** + valid_as_reptile_without_replay |
| qwen3_rq2_ewc_s42 | ewc | valid_for_ewc_claims |
| qwen3_rq4_bandit_replay_s42 | bandit_replay | **bandit_replay_partial_validity** |
| qwen3_rq4_rmgs_s42 | rmgs | valid_for_rmgs_claims |

### Gemma3 (1B) — `batch_size = 2`, affected

| Run ID | Method | Flags |
|---|---|---|
| gemma3_rq1_baseline_s42 | baseline | valid_for_baseline_claims |
| gemma3_rq1_baseline_s123 | baseline | valid_for_baseline_claims |
| gemma3_rq2_replay25_s42 | replay25 | **invalid_for_replay25_claims_due_microbatch_rounding** |
| gemma3_rq2_mer25_s42 | mer25 | **invalid_for_mer25_replay_component_due_microbatch_rounding** + valid_as_reptile_without_replay |
| gemma3_rq2_ewc_s42 | ewc | valid_for_ewc_claims |
| gemma3_rq4_bandit_replay_s42 | bandit_replay | **bandit_replay_partial_validity** |
| gemma3_rq4_rmgs_s42 | rmgs | valid_for_rmgs_claims |

### Llama3 (1B) — `batch_size = 2`, affected

| Run ID | Method | Flags |
|---|---|---|
| llama3_rq1_baseline_s42 | baseline | valid_for_baseline_claims |
| llama3_rq1_baseline_s123 | baseline | valid_for_baseline_claims |
| llama3_rq2_replay25_s42 | replay25 | **invalid_for_replay25_claims_due_microbatch_rounding** |
| llama3_rq2_mer25_s42 | mer25 | **invalid_for_mer25_replay_component_due_microbatch_rounding** + valid_as_reptile_without_replay |
| llama3_rq2_ewc_s42 | ewc | valid_for_ewc_claims |
| llama3_rq4_bandit_replay_s42 | bandit_replay | **bandit_replay_partial_validity** |
| llama3_rq4_rmgs_s42 | rmgs | valid_for_rmgs_claims |

## Summary by paper

| Paper | Cells needed | Cells valid as-of-2026-05-15 | Cells needing rerun |
|---|---|---|---|
| Paper 1 (B) classical audit | 4 models × 4 classical methods = 16 + 3-seed GPT-2 baseline | 10 valid + 3 multi-seed = 13 | 6 (Qwen3/Gemma3/Llama3 × replay25/mer25) |
| Paper 2 (D) EWC systems | 4 model EWC runs + verifier | 4 EWC + 4 baselines = 8 (all valid) | 0 — but profiling instrumentation runs added in Stage 3a |
| Paper 3 (F) probe-driven controllers | 4 models × 2 RL methods = 8 RL + classical baselines | 5 valid (GPT-2 bandit+rmgs, 3 model RMGS) | 3 bandit reruns + 2 mandatory controls + 4 optional controls |

## Action required

1. **Stage 0**: Implement AMENDMENT-004; tests pass; logs include `effective_replay_samples` and `effective_replay_rate`.
2. **Stage 1**: Run 6 classical reruns (Qwen3/Gemma3/Llama3 × replay25/mer25). Append rows with `_cuda` (or `_v2`) suffix; do NOT overwrite existing rows.
3. **Stage 3b**: Run 3 bandit reruns + 2 mandatory controls + (recommended) 4 optional controls.
4. **Re-audit**: After reruns complete, append a "Post-AMENDMENT-004 validity" section to this file recording which new runs become the basis for which paper claims.

## Audit-trail integrity rules

1. **No invalid run is deleted.** All `metrics.json`, `runpack_*.md`, `training_log.jsonl`, and checkpoint files for the affected runs remain in `experiments/runs/`.
2. **No `summary_table.csv` row is overwritten.** New runs append new rows.
3. **No analysis figure or table from the pre-AMENDMENT-004 era is silently regenerated to look "correct."** When figures are regenerated post-Stage-1, they are timestamped 2026-05-XX+ and the pre-AMENDMENT-004 versions remain accessible via git history.
4. **This file is the single source of truth for which run supports which claim** in any paper. Paper plans cite this file by section.
5. **The `replay_samples: 0` evidence** is preserved in the training logs and will not be deleted. If a reviewer asks "how do you know replay didn't happen?", we point to the logs.

## Sign-off

| Stage | Date | Sign-off |
|---|---|---|
| Audit drafted | 2026-05-15 | Claude (post-Codex-Review-1) |
| AMENDMENT-004 implemented and tested | (pending) | |
| Stage 1 reruns complete | (pending) | |
| Stage 3b reruns complete | (pending) | |
| Post-rerun audit appended | (pending) | |
