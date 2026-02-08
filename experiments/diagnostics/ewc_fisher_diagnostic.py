#!/usr/bin/env python3
"""
EWC Fisher Computation Diagnostic

Tests the hypothesis that Fisher is incorrectly computed as (mean_grad)²
instead of mean(grad²).

Per Kirkpatrick2017: F_i = E[(∂log p(x|θ)/∂θ_i)²]
This is the expectation of squared per-sample gradients.

If we compute loss.backward() on a batch mean, we get mean gradients.
Squaring these gives (mean_grad)² which is MUCH smaller than mean(grad²)
due to cancellation effects across samples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    print("=" * 70)
    print("EWC FISHER COMPUTATION BUG DIAGNOSTIC")
    print("=" * 70)

    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create a batch of synthetic data
    torch.manual_seed(42)
    vocab_size = tokenizer.vocab_size
    batch_size = 4
    seq_len = 128  # Shorter for faster testing

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    print(f"\nTest batch: {batch_size} x {seq_len} = {batch_size * seq_len} tokens")

    # ===== METHOD 1: Current Implementation (WRONG) =====
    # Compute batch mean loss, backward, square gradients
    print("\n" + "=" * 70)
    print("METHOD 1: Current Implementation (batch mean gradient squared)")
    print("=" * 70)

    model.zero_grad()
    outputs = model(input_ids, labels=input_ids)
    batch_loss = outputs.loss  # This is already mean over all tokens
    batch_loss.backward()

    fisher_wrong = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher_wrong[name] = param.grad.detach().clone() ** 2

    # Compute statistics
    all_fisher_wrong = torch.cat([f.flatten() for f in fisher_wrong.values()])
    mean_wrong = all_fisher_wrong.mean().item()
    max_wrong = all_fisher_wrong.max().item()
    print(f"\n   Fisher mean: {mean_wrong:.10f}")
    print(f"   Fisher max:  {max_wrong:.10f}")

    # ===== METHOD 2: Correct Implementation (per-sample gradients) =====
    # Compute gradient for each sample separately, square, then average
    print("\n" + "=" * 70)
    print("METHOD 2: Correct Implementation (mean of squared per-sample gradients)")
    print("=" * 70)

    # Initialize accumulators
    fisher_correct = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Process each sample individually
    for i in range(batch_size):
        model.zero_grad()
        sample = input_ids[i:i+1]  # (1, seq_len)
        outputs = model(sample, labels=sample)
        sample_loss = outputs.loss
        sample_loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_correct[name] += param.grad.detach() ** 2

    # Average over samples
    for name in fisher_correct:
        fisher_correct[name] /= batch_size

    # Compute statistics
    all_fisher_correct = torch.cat([f.flatten() for f in fisher_correct.values()])
    mean_correct = all_fisher_correct.mean().item()
    max_correct = all_fisher_correct.max().item()
    print(f"\n   Fisher mean: {mean_correct:.10f}")
    print(f"   Fisher max:  {max_correct:.10f}")

    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    ratio = mean_correct / mean_wrong if mean_wrong > 0 else float('inf')
    print(f"\n   Correct / Wrong ratio: {ratio:.2f}x")
    print(f"\n   If ratio >> 1, this confirms the Fisher computation bug!")

    if ratio > 10:
        print("\n   ❌ BUG CONFIRMED: Fisher is being computed incorrectly!")
        print("      The current implementation squares the mean gradient,")
        print("      but should compute the mean of squared gradients.")
        print("      This underestimates Fisher by a factor of ~{:.0f}x.".format(ratio))
    elif ratio > 2:
        print("\n   ⚠️  Possible issue: Fisher underestimated by {:.1f}x".format(ratio))
    else:
        print("\n   ✓ Fisher computation appears correct (ratio ≈ 1)")

    # ===== ADDITIONAL CHECK: Variance of per-sample gradients =====
    print("\n" + "=" * 70)
    print("ADDITIONAL: Per-Sample Gradient Variance")
    print("=" * 70)

    # Check a specific layer
    test_layer = "transformer.h.0.mlp.c_fc.weight"

    # Collect per-sample gradients for this layer
    sample_grads = []
    for i in range(batch_size):
        model.zero_grad()
        sample = input_ids[i:i+1]
        outputs = model(sample, labels=sample)
        outputs.loss.backward()
        sample_grads.append(model.get_submodule("transformer.h.0.mlp.c_fc").weight.grad.detach().clone())

    stacked = torch.stack(sample_grads)  # (batch_size, out_features, in_features)
    grad_mean = stacked.mean(dim=0)
    grad_var = stacked.var(dim=0)

    print(f"\n   Layer: {test_layer}")
    print(f"   Mean gradient magnitude: {grad_mean.abs().mean().item():.8f}")
    print(f"   Gradient variance (per element): {grad_var.mean().item():.8f}")
    print(f"   Mean gradient squared: {(grad_mean ** 2).mean().item():.10f}")
    print(f"   Mean of squared gradients: {(stacked ** 2).mean().item():.10f}")

    # The difference between these two is the variance!
    # E[X²] - E[X]² = Var(X)
    # So E[X²] = E[X]² + Var(X)
    expected_sq = (grad_mean ** 2).mean().item() + grad_var.mean().item()
    actual_sq = (stacked ** 2).mean().item()
    print(f"   E[X]² + Var(X) ≈ E[X²]? {expected_sq:.10f} ≈ {actual_sq:.10f}")

    # ===== EWC PENALTY COMPARISON =====
    print("\n" + "=" * 70)
    print("EWC PENALTY IMPACT")
    print("=" * 70)

    # Simulate small weight drift
    with torch.no_grad():
        drift_std = 0.01
        diffs = {
            name: torch.randn_like(param) * drift_std
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    ewc_lambda = 100.0

    penalty_wrong = 0.0
    for name in fisher_wrong:
        penalty_wrong += (fisher_wrong[name] * diffs[name] ** 2).sum().item()
    penalty_wrong = (ewc_lambda / 2) * penalty_wrong

    penalty_correct = 0.0
    for name in fisher_correct:
        penalty_correct += (fisher_correct[name] * diffs[name] ** 2).sum().item()
    penalty_correct = (ewc_lambda / 2) * penalty_correct

    print(f"\n   With 0.01 std weight drift and λ=100:")
    print(f"   Wrong EWC penalty:   {penalty_wrong:.6f}")
    print(f"   Correct EWC penalty: {penalty_correct:.6f}")
    print(f"   Ratio: {penalty_correct/penalty_wrong if penalty_wrong > 0 else 'inf':.2f}x")

    # Compare to typical CE loss
    print(f"\n   Typical CE loss: ~3.5")
    print(f"   Wrong penalty as % of CE: {100*penalty_wrong/3.5:.4f}%")
    print(f"   Correct penalty as % of CE: {100*penalty_correct/3.5:.4f}%")

    if penalty_correct / 3.5 > 0.01:
        print("\n   ✓ With correct Fisher, EWC penalty would be meaningful")
    else:
        print("\n   ⚠️  Even with correct Fisher, penalty may be too small")
        print("      Consider increasing λ or checking if model is at true convergence")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
