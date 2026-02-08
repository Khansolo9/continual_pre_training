#!/usr/bin/env python3
"""
EWC-Specific Tests

Tests the EWC implementation to verify:
1. Fisher diagonal has meaningful magnitude (not near-zero)
2. Penalty at anchor is zero/near-zero
3. Penalty increases with parameter perturbation
4. Penalty scales linearly with lambda
5. Fisher computed with sum reduction is >> Fisher with mean reduction

These tests do NOT touch Replay/MER code paths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config


# Import only EWC-related code
from cl_methods import EWC


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="module")
def tiny_gpt2(device):
    """Create a tiny GPT-2 model for fast testing."""
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model = model.to(device)
    return model


@pytest.fixture
def synthetic_tokens(device):
    """Create synthetic token data for testing."""
    torch.manual_seed(42)
    n_tokens = 2048  # Small but enough for a few sequences
    vocab_size = 1000
    return torch.randint(0, vocab_size, (n_tokens,))


# =============================================================================
# TEST: Fisher Magnitude
# =============================================================================

class TestFisherMagnitude:
    """Tests that Fisher diagonal has meaningful (non-zero) values."""

    def test_fisher_mean_is_meaningful(self, tiny_gpt2, synthetic_tokens, device):
        """Fisher mean should be > 1e-6 (not near-zero)."""
        ewc = EWC(tiny_gpt2, device)
        ewc.compute_fisher(
            synthetic_tokens,
            n_samples=10,
            batch_size=2,
            sequence_length=64
        )

        stats = ewc.get_fisher_stats()
        assert stats["mean"] > 1e-6, f"Fisher mean too small: {stats['mean']}"

    def test_fisher_max_is_substantial(self, tiny_gpt2, synthetic_tokens, device):
        """Fisher max should be > 0.01 (at least some important params)."""
        ewc = EWC(tiny_gpt2, device)
        ewc.compute_fisher(
            synthetic_tokens,
            n_samples=10,
            batch_size=2,
            sequence_length=64
        )

        stats = ewc.get_fisher_stats()
        assert stats["max"] > 0.01, f"Fisher max too small: {stats['max']}"

    def test_fisher_all_params_covered(self, tiny_gpt2, synthetic_tokens, device):
        """All trainable params should be in fisher_diag."""
        ewc = EWC(tiny_gpt2, device)
        ewc.compute_fisher(
            synthetic_tokens,
            n_samples=10,
            batch_size=2,
            sequence_length=64
        )

        trainable_params = {n for n, p in tiny_gpt2.named_parameters() if p.requires_grad}
        fisher_params = set(ewc.fisher_diag.keys())

        assert trainable_params == fisher_params, "Parameter mismatch"


# =============================================================================
# TEST: Penalty Behavior
# =============================================================================

class TestPenaltyBehavior:
    """Tests that EWC penalty behaves correctly."""

    def test_penalty_at_anchor_is_zero(self, tiny_gpt2, synthetic_tokens, device):
        """Penalty should be ~0 when weights haven't changed from anchor."""
        ewc = EWC(tiny_gpt2, device)
        ewc.compute_fisher(
            synthetic_tokens,
            n_samples=10,
            batch_size=2,
            sequence_length=64
        )

        penalty = ewc.penalty(ewc_lambda=100.0)
        assert penalty.item() < 1e-6, f"Penalty at anchor should be ~0, got {penalty.item()}"

    def test_penalty_increases_with_perturbation(self, tiny_gpt2, synthetic_tokens, device):
        """Penalty should increase when weights are perturbed."""
        ewc = EWC(tiny_gpt2, device)
        ewc.compute_fisher(
            synthetic_tokens,
            n_samples=10,
            batch_size=2,
            sequence_length=64
        )

        # Penalty at anchor
        penalty_before = ewc.penalty(ewc_lambda=100.0).item()

        # Perturb weights
        with torch.no_grad():
            for param in tiny_gpt2.parameters():
                if param.requires_grad:
                    param.data += torch.randn_like(param) * 0.01

        # Penalty after perturbation
        penalty_after = ewc.penalty(ewc_lambda=100.0).item()

        assert penalty_after > penalty_before, \
            f"Penalty should increase after perturbation: {penalty_before} -> {penalty_after}"
        assert penalty_after > 0.01, \
            f"Penalty after perturbation should be meaningful, got {penalty_after}"

    def test_penalty_scales_linearly_with_lambda(self, tiny_gpt2, synthetic_tokens, device):
        """Penalty should scale linearly with lambda."""
        ewc = EWC(tiny_gpt2, device)
        ewc.compute_fisher(
            synthetic_tokens,
            n_samples=10,
            batch_size=2,
            sequence_length=64
        )

        # Perturb weights first
        with torch.no_grad():
            for param in tiny_gpt2.parameters():
                if param.requires_grad:
                    param.data += torch.randn_like(param) * 0.01

        penalty_100 = ewc.penalty(ewc_lambda=100.0).item()
        penalty_200 = ewc.penalty(ewc_lambda=200.0).item()

        # Should be 2x
        ratio = penalty_200 / penalty_100 if penalty_100 > 0 else 0
        assert 1.9 < ratio < 2.1, f"Expected 2x scaling, got {ratio}x"

    def test_penalty_requires_grad(self, tiny_gpt2, synthetic_tokens, device):
        """Penalty tensor should require grad for backprop."""
        ewc = EWC(tiny_gpt2, device)
        ewc.compute_fisher(
            synthetic_tokens,
            n_samples=10,
            batch_size=2,
            sequence_length=64
        )

        # Perturb weights
        with torch.no_grad():
            for param in tiny_gpt2.parameters():
                if param.requires_grad:
                    param.data += torch.randn_like(param) * 0.01

        penalty = ewc.penalty(ewc_lambda=100.0)
        assert penalty.requires_grad, "Penalty should require grad"


# =============================================================================
# TEST: Fisher Scaling Fix Verification
# =============================================================================

class TestFisherScalingFix:
    """
    Tests that the sum-reduction Fisher is significantly larger than
    mean-reduction Fisher, confirming the fix is effective.
    """

    def test_sum_reduction_fisher_is_larger(self, tiny_gpt2, synthetic_tokens, device):
        """
        Compare Fisher computed with sum reduction (fixed) vs mean reduction (old).
        Sum reduction should give Fisher values ~T larger where T is seq length.
        """
        torch.manual_seed(42)
        seq_len = 64
        batch_size = 2
        n_samples = 10

        # ===== Compute Fisher with FIXED method (sum reduction) =====
        ewc_fixed = EWC(tiny_gpt2, device)
        ewc_fixed.compute_fisher(
            synthetic_tokens,
            n_samples=n_samples,
            batch_size=batch_size,
            sequence_length=seq_len
        )
        fisher_fixed_mean = ewc_fixed.get_fisher_stats()["mean"]

        # ===== Compute Fisher with OLD method (mean reduction) - inline =====
        # Recreate old buggy computation
        fisher_old = {
            name: torch.zeros_like(param)
            for name, param in tiny_gpt2.named_parameters()
            if param.requires_grad
        }

        n_tokens = len(synthetic_tokens)
        n_seqs = min(n_samples, n_tokens // seq_len)
        tokens_subset = synthetic_tokens[:n_seqs * seq_len].view(n_seqs, seq_len)

        tiny_gpt2.eval()
        n_processed = 0

        for i in range(0, len(tokens_subset), batch_size):
            if n_processed >= n_samples:
                break

            batch = tokens_subset[i:i+batch_size].to(device)
            tiny_gpt2.zero_grad()

            # OLD method: use HF loss (mean-reduced)
            outputs = tiny_gpt2(batch, labels=batch)
            loss = outputs.loss
            loss.backward()

            for name, param in tiny_gpt2.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_old[name] += param.grad.detach() ** 2

            n_processed += batch.shape[0]

        # OLD method normalized by samples
        for name in fisher_old:
            fisher_old[name] /= n_processed

        all_fisher_old = torch.cat([f.flatten() for f in fisher_old.values()])
        fisher_old_mean = all_fisher_old.mean().item()

        tiny_gpt2.train()

        # ===== Compare =====
        ratio = fisher_fixed_mean / fisher_old_mean if fisher_old_mean > 0 else float('inf')

        # Fixed Fisher should be significantly larger (>50x for seq_len=64)
        assert ratio > 50, \
            f"Fixed Fisher should be >>50x larger than old. Got ratio={ratio:.2f}x. " \
            f"Fixed mean={fisher_fixed_mean:.8f}, Old mean={fisher_old_mean:.8f}"

        print(f"\n  Fisher scaling verification:")
        print(f"    Old (mean reduction) Fisher mean: {fisher_old_mean:.10f}")
        print(f"    Fixed (sum reduction) Fisher mean: {fisher_fixed_mean:.10f}")
        print(f"    Ratio (fixed/old): {ratio:.2f}x")


# =============================================================================
# TEST: Anchor Storage
# =============================================================================

class TestAnchorStorage:
    """Tests that anchor weights are stored correctly."""

    def test_anchor_matches_model_at_computation(self, tiny_gpt2, synthetic_tokens, device):
        """Anchor (theta_star) should match model weights when Fisher is computed."""
        ewc = EWC(tiny_gpt2, device)
        ewc.compute_fisher(
            synthetic_tokens,
            n_samples=10,
            batch_size=2,
            sequence_length=64
        )

        for name, param in tiny_gpt2.named_parameters():
            if name in ewc.theta_star:
                assert torch.allclose(ewc.theta_star[name], param.detach()), \
                    f"Anchor mismatch for {name}"

    def test_anchor_is_detached(self, tiny_gpt2, synthetic_tokens, device):
        """Anchor weights should not require grad (detached)."""
        ewc = EWC(tiny_gpt2, device)
        ewc.compute_fisher(
            synthetic_tokens,
            n_samples=10,
            batch_size=2,
            sequence_length=64
        )

        for name, tensor in ewc.theta_star.items():
            assert not tensor.requires_grad, f"Anchor {name} should not require grad"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
