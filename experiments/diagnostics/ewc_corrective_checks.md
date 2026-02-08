# EWC Corrective Checks Report

**Date**: 2026-02-07
**Status**: CRITICAL BUG FOUND - Fisher Underestimated by ~181x

---

## Executive Summary

EWC is failing because the **Fisher diagonal is being computed incorrectly**. The current implementation computes Fisher as `(mean_gradient)²`, but the correct formulation is `mean(gradient²)`. Due to the HuggingFace loss being averaged over all tokens, gradients are divided by sequence length before squaring, causing Fisher to be underestimated by a factor of approximately **181x** (roughly the sequence length).

**Key Evidence**:
- Actual EWC penalty during training: **0.0005** (negligible, ~0.015% of CE loss)
- Expected EWC penalty with correct Fisher: **~0.09** (meaningful, ~2.7% of CE loss)
- Forgetting with EWC: **68.88%** (essentially no better than baseline)
- Forgetting with Replay/MER: **7.48% / 4.47%** (working correctly)

---

## Corrective Checks Checklist

### Check 1: Fisher Magnitude Verification

| Item | File:Line | What to Measure | Expected | Actual | Status |
|------|-----------|-----------------|----------|--------|--------|
| Fisher mean | cl_methods.py:246 | `mean_fisher` logged | >1e-4 | 2.95e-7 | ❌ FAIL |
| Fisher max | cl_methods.py:244 | `max(F.max())` | >1.0 | 0.69 | ⚠️ LOW |
| Params tracked | trainer.py:602 | `len(fisher_diag)` | 148 | 148 | ✓ PASS |

**Why It Matters**: If Fisher values are near zero, the EWC penalty will have no effect regardless of λ.

### Check 2: Batch vs Per-Sample Gradients

| Item | Hypothesis | Measured Ratio | Status |
|------|------------|----------------|--------|
| Batch averaging | Computing (mean_grad)² vs mean(grad²) | 2.01x underestimate | ⚠️ MINOR |
| Token averaging | Computing (mean_token_grad)² vs mean(token_grad²) | **180.75x underestimate** | ❌ CRITICAL |

**Why It Matters**: HuggingFace's loss is averaged over all tokens, so gradients are already divided by sequence length (~512). Squaring these gives values ~262,144x smaller than they should be.

### Check 3: Gradient Flow Verification

| Item | File:Line | Expected | Actual | Status |
|------|-----------|----------|--------|--------|
| Penalty requires_grad | cl_methods.py:263 | True | True | ✓ PASS |
| Gradients non-zero | trainer.py:545 | 148/148 | 148/148 | ✓ PASS |
| Max gradient magnitude | diagnostic | >0.01 | 1.30 | ✓ PASS |

**Why It Matters**: If the penalty tensor is detached, it won't contribute to backpropagation.

### Check 4: Device Consistency

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| Model device | mps:0 | mps:0 | ✓ PASS |
| Anchor device | mps:0 | mps:0 | ✓ PASS |
| Fisher device | mps:0 | mps:0 | ✓ PASS |

### Check 5: Parameter Name Matching

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| Model params | 148 | 148 | ✓ PASS |
| Fisher params | 148 | 148 | ✓ PASS |
| Anchor params | 148 | 148 | ✓ PASS |
| Missing from Fisher | 0 | 0 | ✓ PASS |

### Check 6: EWC Penalty Magnitude During Training

| Metric | Value | Assessment |
|--------|-------|------------|
| Min penalty | 0.00021 | Negligible |
| Max penalty | 0.00055 | Negligible |
| Mean penalty | 0.00050 | Negligible |
| CE loss | ~3.3 | - |
| Penalty as % of CE | **0.015%** | ❌ Too small to affect training |

---

## Verified Root Cause

### The Fisher Computation Bug

**Location**: `src/cl_methods.py:170-246` (`EWC.compute_fisher`)

**Current (Incorrect) Implementation**:
```python
# Forward pass
outputs = self.model(input_ids, labels=input_ids)
loss = outputs.loss  # ← Mean over all tokens: Σ_t loss_t / T

# Backward pass
loss.backward()

# Accumulate squared gradients
for name, param in self.model.named_parameters():
    if param.requires_grad and param.grad is not None:
        self.fisher_diag[name] += param.grad.detach() ** 2  # ← (mean_grad)²
```

**Problem**:
- `outputs.loss` is the mean loss over all T tokens: `L = (1/T) Σ_t L_t`
- Gradient: `∂L/∂θ = (1/T) Σ_t ∂L_t/∂θ` (mean of per-token gradients)
- Squared: `(∂L/∂θ)² = (1/T²) (Σ_t ∂L_t/∂θ)²`

**Fisher Definition** (per Kirkpatrick2017):
- `F_i = E[(∂ log p(x|θ) / ∂θ_i)²]`
- This should be: `(1/T) Σ_t (∂L_t/∂θ)²` (mean of squared per-token gradients)

**Mathematical Inequality**:
```
E[X²] ≥ (E[X])²    (by Jensen's inequality)
```

Due to gradient cancellation across tokens:
- `(mean_grad)²` << `mean(grad²)`
- Measured underestimation: **~181x** (approximately equal to sequence length)

### Diagnostic Evidence

**From `ewc_loss_diagnostic.py`**:
```
Method 1 (HF mean loss) Fisher mean: 0.0000017163
Method 2 (sum loss) Fisher mean:     0.4481727779
Method 3 (per-token) Fisher mean:    0.0003102237

Ratio (sum / mean):      261121.27x
Ratio (per-token / mean): 180.75x

Expected sum/mean ratio (seq_len²): 261121
Actual sum/mean ratio: 261121.27

❌ BUG CONFIRMED: Token-level averaging causes massive underestimation!
```

---

## Expected vs Actual Comparison

| Metric | Expected (Working EWC) | Actual (Broken EWC) | Ratio |
|--------|------------------------|---------------------|-------|
| Fisher mean | ~1e-4 to 1e-3 | 2.95e-7 | ~300-3000x too small |
| EWC penalty | ~0.05-0.5 | 0.0005 | ~100-1000x too small |
| Forgetting % | ~15-25% | 68.88% | ~3-4x worse |
| Replay forgetting | 7.48% | 7.48% | (working correctly) |
| MER forgetting | 4.47% | 4.47% | (working correctly) |

---

## Corrective Actions (Ranked by Impact)

### 1. **CRITICAL: Fix Fisher Computation** (High Impact, High Confidence)

**Evidence**: Loss reduction diagnostic shows 181x underestimation.

**Fix**: Compute Fisher using per-token gradients instead of batch mean gradients.

**Option A: Sum loss instead of mean** (Simplest)
```python
# In compute_fisher():
outputs = self.model(input_ids)
logits = outputs.logits

# Compute per-token loss without reduction
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()

loss_per_token = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    reduction='sum'  # Sum, not mean
)

loss_per_token.backward()

# Then accumulate and divide by total tokens (not samples)
```

**Option B: Per-token backward passes** (More accurate, slower)
```python
# Compute gradient for each token position separately
# Then average the squared gradients
```

**Recommendation**: Option A is sufficient and much faster. The key is to not divide gradients by T before squaring.

### 2. **MEDIUM: Increase λ as Temporary Workaround** (Low Impact without #1)

**Evidence**: With current Fisher, even λ=10000 would give penalty of ~0.05 (still weak).

**Note**: This is NOT a proper fix. Even with very high λ, the Fisher is fundamentally wrong.

### 3. **LOW: Consider Empirical Fisher** (Alternative Approach)

**Reference**: Some implementations use `∂L/∂θ` computed on the actual data labels rather than model samples.

**Status**: Current implementation already uses empirical Fisher (labels from data). The issue is the loss reduction, not the Fisher variant.

---

## Verification Commands

### Run Diagnostic Scripts
```bash
# Activate environment
source .cpt-env/bin/activate

# Run Fisher computation diagnostic
python experiments/diagnostics/ewc_diagnostic.py

# Run loss reduction diagnostic
python experiments/diagnostics/ewc_loss_diagnostic.py

# Run Fisher bug diagnostic
python experiments/diagnostics/ewc_fisher_diagnostic.py
```

### Check Training Logs
```bash
# View EWC penalties from actual run
jq '.ewc_penalty' experiments/runs/rq2_ewc_s42/training_log.jsonl | head -20

# Compare to CE loss
jq '.loss' experiments/runs/rq2_ewc_s42/training_log.jsonl | head -20
```

---

## Key Log Snippets

### Fisher Computation (from first diagnostic)
```
Total Fisher tensors: 148
Total Fisher scalars: 124,439,808
Overall mean: 0.0000002954    ← ~3e-7, should be ~1e-4
Overall max:  0.6934626698

⚠️  WARNING: Fisher mean is very small (<1e-6)
This means EWC penalty will have minimal effect!
```

### Training Log (rq2_ewc_s42)
```json
{"step": 100, "loss": 3.404, "ewc_penalty": 0.000214}
{"step": 200, "loss": 3.657, "ewc_penalty": 0.000410}
{"step": 500, "loss": 3.199, "ewc_penalty": 0.000543}
{"step": 1200, "loss": 3.272, "ewc_penalty": 0.000548}
```

Note: EWC penalty (~0.0005) is 0.015% of CE loss (~3.3). This has no effect on training.

### Per-Token vs Mean Comparison
```
Method 1 (HF mean loss) Fisher mean: 0.0000017163
Method 3 (per-token) Fisher mean:    0.0003102237

Ratio (per-token / mean): 180.75x
```

---

## Unverified Hypotheses (Require Further Investigation)

1. **λ Scaling with Gradient Accumulation**: Does grad_accum=4 further scale down the penalty?
   - Status: UNVERIFIED
   - Evidence needed: Compare penalty with grad_accum=1 vs 4

2. **Anchor Timing**: Are weights stored before or after the final optimizer step?
   - Status: UNVERIFIED but likely correct (compute_fisher is called after train_domain returns)
   - Evidence needed: Print parameter checksums before/after

3. **dtype Precision**: Could float32 gradients underflow for very small values?
   - Status: UNVERIFIED but unlikely (Fisher max is 0.69, not underflowing)

---

## Conclusion

The EWC implementation has a **critical bug** in Fisher computation. The fix is straightforward: change the loss reduction from `mean` to `sum` (or compute per-token gradients explicitly) before squaring. This will increase Fisher values by ~181x, making the EWC penalty meaningful.

After fixing, re-run `rq2_ewc_s42` and expect:
- EWC penalty: ~0.05-0.5 (instead of 0.0005)
- Forgetting: ~15-25% (instead of 69%)

---

**Report Generated**: 2026-02-07
**Device**: MPS (Apple Silicon)
**Diagnostic Scripts**:
- `experiments/diagnostics/ewc_diagnostic.py`
- `experiments/diagnostics/ewc_fisher_diagnostic.py`
- `experiments/diagnostics/ewc_loss_diagnostic.py`
