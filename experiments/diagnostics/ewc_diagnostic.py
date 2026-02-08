#!/usr/bin/env python3
"""
EWC Diagnostic Script

Verifies EWC implementation by checking:
1. Fisher diagonal computation and magnitudes
2. Anchor weight storage
3. Parameter name matching
4. Penalty gradient flow
5. Device consistency
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from cl_methods import EWC

def main():
    print("=" * 70)
    print("EWC DIAGNOSTIC REPORT")
    print("=" * 70)

    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n[1] Device: {device}")

    # Load model
    print("\n[2] Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Count trainable parameters
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for n, p in trainable_params)
    print(f"   Trainable parameter tensors: {len(trainable_params)}")
    print(f"   Total trainable scalars: {total_params:,}")

    # Create synthetic "Domain A" data (1000 samples for Fisher)
    print("\n[3] Creating synthetic training data...")
    seq_len = 512
    n_samples = 100  # Small for diagnostic

    # Generate random token IDs (simulating domain A data)
    torch.manual_seed(42)
    vocab_size = tokenizer.vocab_size
    synthetic_tokens = torch.randint(0, vocab_size, (n_samples * seq_len,))
    print(f"   Synthetic tokens: {len(synthetic_tokens):,}")

    # Initialize EWC and compute Fisher
    print("\n[4] Computing Fisher diagonal...")
    ewc = EWC(model, device)
    ewc.compute_fisher(
        synthetic_tokens,
        n_samples=n_samples,
        batch_size=4,
        sequence_length=seq_len
    )

    # === CHECK 1: Fisher Statistics ===
    print("\n" + "=" * 70)
    print("CHECK 1: Fisher Diagonal Statistics")
    print("=" * 70)

    fisher_stats = []
    for name, fisher in ewc.fisher_diag.items():
        f_min = fisher.min().item()
        f_max = fisher.max().item()
        f_mean = fisher.mean().item()
        f_std = fisher.std().item()
        f_nonzero = (fisher > 1e-10).sum().item()
        fisher_stats.append({
            'name': name,
            'shape': tuple(fisher.shape),
            'numel': fisher.numel(),
            'min': f_min,
            'max': f_max,
            'mean': f_mean,
            'std': f_std,
            'nonzero_pct': 100.0 * f_nonzero / fisher.numel()
        })

    # Overall Fisher statistics
    all_fisher = torch.cat([f.flatten() for f in ewc.fisher_diag.values()])
    overall_mean = all_fisher.mean().item()
    overall_max = all_fisher.max().item()
    overall_std = all_fisher.std().item()

    print(f"\n   Total Fisher tensors: {len(ewc.fisher_diag)}")
    print(f"   Total Fisher scalars: {all_fisher.numel():,}")
    print(f"   Overall mean: {overall_mean:.10f}")
    print(f"   Overall max:  {overall_max:.10f}")
    print(f"   Overall std:  {overall_std:.10f}")

    # Check if Fisher is too small
    if overall_mean < 1e-6:
        print("\n   ⚠️  WARNING: Fisher mean is very small (<1e-6)")
        print("   This means EWC penalty will have minimal effect!")

    # Top 5 largest mean Fisher values
    print("\n   Top 5 layers by mean Fisher:")
    sorted_stats = sorted(fisher_stats, key=lambda x: x['mean'], reverse=True)
    for i, s in enumerate(sorted_stats[:5]):
        print(f"   {i+1}. {s['name']}: mean={s['mean']:.8f}, max={s['max']:.8f}")

    # Bottom 5
    print("\n   Bottom 5 layers by mean Fisher:")
    for i, s in enumerate(sorted_stats[-5:]):
        print(f"   {i+1}. {s['name']}: mean={s['mean']:.12f}")

    # === CHECK 2: Anchor Weight Storage ===
    print("\n" + "=" * 70)
    print("CHECK 2: Anchor Weight Storage (theta_star)")
    print("=" * 70)

    print(f"\n   Anchor tensors stored: {len(ewc.theta_star)}")
    print(f"   Anchor tensors match Fisher tensors: {set(ewc.theta_star.keys()) == set(ewc.fisher_diag.keys())}")

    # Check that anchor matches current weights (should be same since no training between)
    anchor_matches = 0
    for name, anchor in ewc.theta_star.items():
        current = dict(model.named_parameters())[name]
        if torch.allclose(anchor, current.detach()):
            anchor_matches += 1
    print(f"   Anchor weights match current model: {anchor_matches}/{len(ewc.theta_star)}")

    # === CHECK 3: Penalty Computation ===
    print("\n" + "=" * 70)
    print("CHECK 3: Penalty Computation")
    print("=" * 70)

    # Penalty should be 0 when weights haven't changed
    penalty_same = ewc.penalty(ewc_lambda=100.0)
    print(f"\n   Penalty with same weights (should be ~0): {penalty_same.item():.10f}")

    # Now simulate training by modifying some weights
    print("\n   Simulating weight drift...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data += torch.randn_like(param) * 0.01  # Small perturbation

    # Recompute penalty
    penalty_perturbed = ewc.penalty(ewc_lambda=100.0)
    print(f"   Penalty after 0.01 std perturbation: {penalty_perturbed.item():.6f}")

    # Expected penalty magnitude calculation
    # If Fisher mean ~ F and weight diff std ~ 0.01
    # Then penalty ~ (lambda/2) * sum(F * 0.01^2 * n_params)
    # = (100/2) * F * 0.0001 * 124M ~ 50 * F * 12400
    expected_order = 50 * overall_mean * 0.0001 * total_params
    print(f"   Expected order of magnitude: {expected_order:.6f}")

    # === CHECK 4: Gradient Flow ===
    print("\n" + "=" * 70)
    print("CHECK 4: Gradient Flow Through Penalty")
    print("=" * 70)

    # Reset gradients
    model.zero_grad()

    # Compute penalty and backward
    penalty = ewc.penalty(ewc_lambda=100.0)
    print(f"\n   Penalty value: {penalty.item():.6f}")
    print(f"   Penalty requires_grad: {penalty.requires_grad}")

    if penalty.requires_grad:
        penalty.backward()

        # Check if gradients exist
        grad_count = 0
        grad_nonzero = 0
        grad_max = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                if param.grad.abs().max() > 1e-10:
                    grad_nonzero += 1
                grad_max = max(grad_max, param.grad.abs().max().item())

        print(f"   Parameters with gradients: {grad_count}/{len(trainable_params)}")
        print(f"   Parameters with non-zero gradients: {grad_nonzero}")
        print(f"   Max gradient magnitude: {grad_max:.8f}")

        if grad_nonzero == 0:
            print("\n   ⚠️  WARNING: All gradients are zero!")
            print("   EWC penalty is not contributing to optimization!")
    else:
        print("\n   ❌ CRITICAL: Penalty does not require grad!")
        print("   Check if fisher_diag or theta_star tensors break the graph")

    # === CHECK 5: Device Consistency ===
    print("\n" + "=" * 70)
    print("CHECK 5: Device Consistency")
    print("=" * 70)

    model_device = next(model.parameters()).device
    anchor_devices = set(t.device for t in ewc.theta_star.values())
    fisher_devices = set(t.device for t in ewc.fisher_diag.values())

    print(f"\n   Model device: {model_device}")
    print(f"   Anchor devices: {anchor_devices}")
    print(f"   Fisher devices: {fisher_devices}")

    if len(anchor_devices) > 1 or model_device not in anchor_devices:
        print("   ⚠️  WARNING: Device mismatch detected!")

    # === CHECK 6: Parameter Name Matching ===
    print("\n" + "=" * 70)
    print("CHECK 6: Parameter Name Matching")
    print("=" * 70)

    model_params = set(n for n, p in model.named_parameters() if p.requires_grad)
    anchor_params = set(ewc.theta_star.keys())
    fisher_params = set(ewc.fisher_diag.keys())

    missing_from_fisher = model_params - fisher_params
    extra_in_fisher = fisher_params - model_params

    print(f"\n   Model trainable params: {len(model_params)}")
    print(f"   Anchor params: {len(anchor_params)}")
    print(f"   Fisher params: {len(fisher_params)}")
    print(f"   Missing from Fisher: {len(missing_from_fisher)}")
    print(f"   Extra in Fisher: {len(extra_in_fisher)}")

    if missing_from_fisher:
        print(f"\n   ⚠️  Missing params: {list(missing_from_fisher)[:5]}...")

    # === CHECK 7: Compare with actual EWC run ===
    print("\n" + "=" * 70)
    print("CHECK 7: Compare with Actual Run (rq2_ewc_s42)")
    print("=" * 70)

    # Load actual training log
    import json
    log_path = os.path.join(os.path.dirname(__file__), '..', 'runs', 'rq2_ewc_s42', 'training_log.jsonl')
    if os.path.exists(log_path):
        with open(log_path) as f:
            entries = [json.loads(line) for line in f]

        ewc_entries = [e for e in entries if e.get('domain') == 'domain_b']
        if ewc_entries:
            penalties = [e.get('ewc_penalty', 0) for e in ewc_entries]
            print(f"\n   EWC penalties from actual run:")
            print(f"   Min:  {min(penalties):.8f}")
            print(f"   Max:  {max(penalties):.8f}")
            print(f"   Mean: {np.mean(penalties):.8f}")

            # Estimate what penalty SHOULD be
            # After training on Domain B, weights should drift
            # If forgetting is 68.88%, there's significant drift
            print(f"\n   Analysis:")
            print(f"   With λ=100, penalty of ~0.0005 means:")
            print(f"   Raw Σ F_i(θ-θ*)² = 0.0005 * 2 / 100 = {0.0005 * 2 / 100:.8f}")
            print(f"   This is EXTREMELY small for 124M parameters!")
            print(f"   Either Fisher values are ~0, or weight diff is ~0")
    else:
        print(f"\n   Could not find training log at {log_path}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    issues = []

    if overall_mean < 1e-6:
        issues.append("Fisher diagonal values are near zero (mean < 1e-6)")

    if penalty_same.item() > 0.01:
        issues.append("Penalty is non-zero even with unchanged weights")

    if penalty.requires_grad and grad_nonzero == 0:
        issues.append("Penalty has zero gradients")

    if not penalty.requires_grad:
        issues.append("Penalty does not require gradients (broken graph)")

    if missing_from_fisher:
        issues.append(f"Missing {len(missing_from_fisher)} params from Fisher")

    if issues:
        print("\n   ❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"      {i}. {issue}")
    else:
        print("\n   ✓ No obvious issues found in basic checks")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
