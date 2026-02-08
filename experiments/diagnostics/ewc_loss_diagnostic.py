#!/usr/bin/env python3
"""
EWC Loss Reduction Diagnostic

Tests whether the HuggingFace loss reduction (mean over tokens) is causing
Fisher underestimation.

When loss = mean(per_token_losses), gradient = mean(per_token_gradients)
Squaring this gives (mean_grad)² << mean(grad²)

For Fisher, we want E[(∂log p(x|θ)/∂θ)²]
But with mean reduction, we get (E[∂log p(x|θ)/∂θ])² which is much smaller.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    print("=" * 70)
    print("EWC LOSS REDUCTION DIAGNOSTIC")
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

    # Create a single sequence
    torch.manual_seed(42)
    vocab_size = tokenizer.vocab_size
    seq_len = 512

    input_ids = torch.randint(0, vocab_size, (1, seq_len)).to(device)
    print(f"\nTest sequence: 1 x {seq_len} tokens")

    # ===== METHOD 1: Standard HF loss (mean over tokens) =====
    print("\n" + "=" * 70)
    print("METHOD 1: Standard HuggingFace loss (mean over tokens)")
    print("=" * 70)

    model.zero_grad()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss  # Already mean over all tokens
    print(f"\n   Loss value: {loss.item():.6f}")
    loss.backward()

    fisher_mean_loss = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher_mean_loss[name] = param.grad.detach().clone() ** 2

    all_fisher_mean = torch.cat([f.flatten() for f in fisher_mean_loss.values()])
    print(f"   Fisher mean: {all_fisher_mean.mean().item():.10f}")
    print(f"   Fisher max:  {all_fisher_mean.max().item():.10f}")

    # ===== METHOD 2: Sum loss over tokens (no reduction) =====
    print("\n" + "=" * 70)
    print("METHOD 2: Sum loss over tokens (scaled up by seq_len)")
    print("=" * 70)

    model.zero_grad()
    outputs = model(input_ids)
    logits = outputs.logits

    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Compute per-token loss
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    )

    # Sum (not mean) over tokens
    loss_sum = loss_per_token.sum()
    print(f"\n   Loss value (sum): {loss_sum.item():.6f}")
    print(f"   Equivalent mean loss: {loss_sum.item() / (seq_len - 1):.6f}")

    loss_sum.backward()

    fisher_sum_loss = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher_sum_loss[name] = param.grad.detach().clone() ** 2

    all_fisher_sum = torch.cat([f.flatten() for f in fisher_sum_loss.values()])
    print(f"   Fisher mean: {all_fisher_sum.mean().item():.10f}")
    print(f"   Fisher max:  {all_fisher_sum.max().item():.10f}")

    # ===== METHOD 3: Per-token Fisher (average of squared per-token gradients) =====
    print("\n" + "=" * 70)
    print("METHOD 3: Per-token Fisher (correct: mean of squared per-token grads)")
    print("=" * 70)

    fisher_per_token = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Get logits
    outputs = model(input_ids)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    n_tokens = seq_len - 1
    print(f"\n   Processing {n_tokens} tokens individually...")

    for t in range(n_tokens):
        model.zero_grad()

        # Single token loss
        token_logit = shift_logits[0, t:t+1, :]
        token_label = shift_labels[0, t:t+1]
        token_loss = F.cross_entropy(token_logit, token_label)

        # Backward for this token
        # Need to retain graph for subsequent tokens
        if t < n_tokens - 1:
            token_loss.backward(retain_graph=True)
        else:
            token_loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_per_token[name] += param.grad.detach() ** 2

    # Average over tokens
    for name in fisher_per_token:
        fisher_per_token[name] /= n_tokens

    all_fisher_per_token = torch.cat([f.flatten() for f in fisher_per_token.values()])
    print(f"   Fisher mean: {all_fisher_per_token.mean().item():.10f}")
    print(f"   Fisher max:  {all_fisher_per_token.max().item():.10f}")

    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    mean_ratio_sum = all_fisher_sum.mean().item() / all_fisher_mean.mean().item()
    mean_ratio_per_token = all_fisher_per_token.mean().item() / all_fisher_mean.mean().item()

    print(f"\n   Method 1 (HF mean loss) Fisher mean: {all_fisher_mean.mean().item():.10f}")
    print(f"   Method 2 (sum loss) Fisher mean:     {all_fisher_sum.mean().item():.10f}")
    print(f"   Method 3 (per-token) Fisher mean:    {all_fisher_per_token.mean().item():.10f}")

    print(f"\n   Ratio (sum / mean):      {mean_ratio_sum:.2f}x")
    print(f"   Ratio (per-token / mean): {mean_ratio_per_token:.2f}x")

    # The sum method should give gradients scaled by seq_len
    # So Fisher should be scaled by seq_len²
    expected_sum_ratio = (seq_len - 1) ** 2
    print(f"\n   Expected sum/mean ratio (seq_len²): {expected_sum_ratio}")
    print(f"   Actual sum/mean ratio: {mean_ratio_sum:.2f}")

    if mean_ratio_per_token > 10:
        print("\n   ❌ BUG CONFIRMED: Token-level averaging causes massive underestimation!")
        print(f"      Fisher is underestimated by ~{mean_ratio_per_token:.0f}x due to loss reduction.")
        print("      The gradient of mean(losses) is NOT the same as mean(individual gradients²)!")
    else:
        print("\n   Per-token ratio is not dramatically higher.")
        print("   The main issue may be elsewhere (model at convergence, etc.)")

    # ===== EWC PENALTY COMPARISON =====
    print("\n" + "=" * 70)
    print("EWC PENALTY WITH DIFFERENT FISHER METHODS")
    print("=" * 70)

    # Simulate weight drift
    with torch.no_grad():
        drift_std = 0.01
        diffs = {
            name: torch.randn_like(param) * drift_std
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    ewc_lambda = 100.0

    penalty_mean = sum(
        (fisher_mean_loss.get(name, torch.zeros_like(diffs[name])) * diffs[name] ** 2).sum().item()
        for name in diffs
    )
    penalty_mean = (ewc_lambda / 2) * penalty_mean

    penalty_per_token = sum(
        (fisher_per_token.get(name, torch.zeros_like(diffs[name])) * diffs[name] ** 2).sum().item()
        for name in diffs
    )
    penalty_per_token = (ewc_lambda / 2) * penalty_per_token

    print(f"\n   With 0.01 std drift and λ=100:")
    print(f"   Standard HF loss Fisher penalty:  {penalty_mean:.6f}")
    print(f"   Per-token Fisher penalty:         {penalty_per_token:.6f}")
    print(f"   Ratio: {penalty_per_token/penalty_mean if penalty_mean > 0 else 'inf':.2f}x")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
