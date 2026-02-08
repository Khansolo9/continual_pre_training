#!/usr/bin/env python3
"""
EWC Smoke Check Diagnostic

Quick diagnostic to verify EWC Fisher computation is working correctly.
Runs in <1 minute, does not touch registry or run full experiments.

Usage:
    python src/adhoc/_diagnostics/ewc_smoke_check.py

Checks:
1. Fisher diagonal has meaningful magnitude (mean > 1e-5)
2. Fisher max is substantial (> 0.1)
3. Penalty at anchor is ~0
4. Penalty after perturbation is meaningful
5. Compare old vs new Fisher method (ratio should be >50x)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from cl_methods import EWC


def main():
    print("=" * 70)
    print("EWC SMOKE CHECK DIAGNOSTIC")
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
    print("\nLoading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create synthetic tokens
    print("\nCreating synthetic data...")
    torch.manual_seed(42)
    vocab_size = tokenizer.vocab_size
    n_tokens = 5120  # ~10 sequences of 512
    synthetic_tokens = torch.randint(0, vocab_size, (n_tokens,))

    print(f"Tokens: {n_tokens:,}")

    # ===== Check 1: Compute Fisher with FIXED method =====
    print("\n" + "=" * 70)
    print("CHECK 1: Fisher Computation (Fixed - Sum Reduction)")
    print("=" * 70)

    ewc = EWC(model, device)
    ewc.compute_fisher(
        synthetic_tokens,
        n_samples=10,
        batch_size=2,
        sequence_length=512
    )

    stats = ewc.get_fisher_stats()
    print(f"\n  Fisher Statistics:")
    print(f"    Mean:   {stats['mean']:.8f}")
    print(f"    Median: {stats['median']:.8f}")
    print(f"    Max:    {stats['max']:.8f}")
    print(f"    Min:    {stats['min']:.8f}")
    print(f"    Params: {stats['n_params']:,}")

    check1_pass = stats['mean'] > 1e-5
    print(f"\n  [{'PASS' if check1_pass else 'FAIL'}] Fisher mean > 1e-5: {stats['mean']:.8f}")

    check1b_pass = stats['max'] > 0.1
    print(f"  [{'PASS' if check1b_pass else 'FAIL'}] Fisher max > 0.1: {stats['max']:.8f}")

    # ===== Check 2: Penalty at Anchor =====
    print("\n" + "=" * 70)
    print("CHECK 2: Penalty at Anchor (should be ~0)")
    print("=" * 70)

    penalty_anchor = ewc.penalty(ewc_lambda=100.0).item()
    print(f"\n  Penalty at anchor (λ=100): {penalty_anchor:.10f}")

    check2_pass = penalty_anchor < 1e-6
    print(f"\n  [{'PASS' if check2_pass else 'FAIL'}] Penalty < 1e-6: {penalty_anchor:.10f}")

    # ===== Check 3: Penalty After Perturbation =====
    print("\n" + "=" * 70)
    print("CHECK 3: Penalty After Perturbation")
    print("=" * 70)

    # Perturb weights
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.data += torch.randn_like(param) * 0.01

    penalty_perturbed = ewc.penalty(ewc_lambda=100.0).item()
    print(f"\n  Penalty after 0.01 std perturbation (λ=100): {penalty_perturbed:.6f}")

    check3_pass = penalty_perturbed > 0.1
    print(f"\n  [{'PASS' if check3_pass else 'FAIL'}] Penalty > 0.1: {penalty_perturbed:.6f}")

    # Compare to typical CE loss
    ce_loss = 3.5  # Approximate
    pct_of_ce = 100 * penalty_perturbed / ce_loss
    print(f"  Penalty as % of CE loss (~3.5): {pct_of_ce:.2f}%")

    # ===== Check 4: Lambda Scaling =====
    print("\n" + "=" * 70)
    print("CHECK 4: Lambda Scaling")
    print("=" * 70)

    penalty_100 = ewc.penalty(ewc_lambda=100.0).item()
    penalty_200 = ewc.penalty(ewc_lambda=200.0).item()
    ratio = penalty_200 / penalty_100 if penalty_100 > 0 else 0

    print(f"\n  Penalty (λ=100): {penalty_100:.6f}")
    print(f"  Penalty (λ=200): {penalty_200:.6f}")
    print(f"  Ratio: {ratio:.2f}x (expected: 2.0x)")

    check4_pass = 1.9 < ratio < 2.1
    print(f"\n  [{'PASS' if check4_pass else 'FAIL'}] Ratio ≈ 2.0: {ratio:.2f}")

    # ===== Check 5: Compare to Old Method =====
    print("\n" + "=" * 70)
    print("CHECK 5: Compare Fixed vs Old Fisher Method")
    print("=" * 70)

    # Reload model to get fresh weights
    model_fresh = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Compute with OLD method (inline)
    fisher_old = {
        name: torch.zeros_like(param)
        for name, param in model_fresh.named_parameters()
        if param.requires_grad
    }

    seq_len = 512
    n_seqs = min(10, n_tokens // seq_len)
    tokens_subset = synthetic_tokens[:n_seqs * seq_len].view(n_seqs, seq_len)

    model_fresh.eval()
    n_processed = 0

    for i in range(0, len(tokens_subset), 2):
        batch = tokens_subset[i:i+2].to(device)
        model_fresh.zero_grad()

        # OLD method: HF mean-reduced loss
        outputs = model_fresh(batch, labels=batch)
        loss = outputs.loss
        loss.backward()

        for name, param in model_fresh.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_old[name] += param.grad.detach() ** 2

        n_processed += batch.shape[0]

    for name in fisher_old:
        fisher_old[name] /= n_processed

    all_fisher_old = torch.cat([f.flatten() for f in fisher_old.values()])
    fisher_old_mean = all_fisher_old.mean().item()

    # Compute with NEW method
    ewc_new = EWC(model_fresh, device)
    ewc_new.compute_fisher(
        synthetic_tokens,
        n_samples=10,
        batch_size=2,
        sequence_length=512
    )
    fisher_new_mean = ewc_new.get_fisher_stats()["mean"]

    ratio_fix = fisher_new_mean / fisher_old_mean if fisher_old_mean > 0 else float('inf')

    print(f"\n  Old method (mean reduction) Fisher mean: {fisher_old_mean:.10f}")
    print(f"  New method (sum reduction) Fisher mean:  {fisher_new_mean:.10f}")
    print(f"  Ratio (new/old): {ratio_fix:.2f}x")

    check5_pass = ratio_fix > 50
    print(f"\n  [{'PASS' if check5_pass else 'FAIL'}] Ratio > 50x: {ratio_fix:.2f}x")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = all([check1_pass, check1b_pass, check2_pass, check3_pass, check4_pass, check5_pass])

    checks = [
        ("Fisher mean > 1e-5", check1_pass),
        ("Fisher max > 0.1", check1b_pass),
        ("Penalty at anchor < 1e-6", check2_pass),
        ("Penalty after perturbation > 0.1", check3_pass),
        ("Lambda scaling ≈ 2x", check4_pass),
        ("New/Old Fisher ratio > 50x", check5_pass),
    ]

    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print("\n" + "=" * 70)
    if all_pass:
        print("  ALL CHECKS PASSED - EWC fix is working correctly")
    else:
        print("  SOME CHECKS FAILED - Review issues above")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
