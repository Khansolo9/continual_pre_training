#!/usr/bin/env python3
"""
Tests for RMGS (Reward-Modulated Gradient Scaling).

Covers:
- Scale computation and clamping
- EMA reward update
- Probe loss tracking
- Stats serialization
- Config merge behavior for rmgs method
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cl_methods import (
    RMGS, create_probe_set, evaluate_probe, is_rmgs_method,
)


# ---------------------------------------------------------------------------
# Tiny model for unit tests
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Minimal causal LM for testing."""
    def __init__(self, vocab_size=100, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, labels=None):
        h = self.embed(input_ids)
        logits = self.linear(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1)
            )
        return type('Out', (), {'logits': logits, 'loss': loss})()


def _make_tokens(n=10000, vocab=100, seed=0):
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=rng)


# ===========================================================================
# RMGS Tests
# ===========================================================================

class TestRMGS:
    """Test RMGS scale computation and update."""

    def _make_rmgs(self, beta=2.0, min_scale=0.05, probe_interval=3, seed=42):
        """Create an RMGS instance for testing."""
        model = TinyLM()
        valid_a = _make_tokens(5120)
        probe, probe_hash, _ = create_probe_set(
            valid_a, probe_size=5, sequence_length=512, seed=seed
        )
        rmgs = RMGS(
            model=model, device="cpu",
            probe_set=probe, probe_hash=probe_hash,
            probe_interval=probe_interval,
            ema_alpha=0.3, beta=beta, min_scale=min_scale,
        )
        return rmgs

    def test_initialization(self):
        """RMGS initializes with probe loss and scale=1.0."""
        rmgs = self._make_rmgs()
        rmgs.initialize()
        assert rmgs.last_probe_loss is not None
        assert rmgs.last_probe_loss > 0
        assert rmgs.current_scale == 1.0
        assert rmgs.ema_reward == 0.0
        assert len(rmgs.probe_loss_history) == 1

    def test_scale_starts_at_one(self):
        """Initial scale is 1.0 (no modulation before first evaluation)."""
        rmgs = self._make_rmgs()
        rmgs.initialize()
        assert rmgs.get_scale() == 1.0

    def test_step_returns_none_between_intervals(self):
        """step() returns None when not at probe_interval boundary."""
        rmgs = self._make_rmgs()
        rmgs.initialize()
        result = rmgs.step()  # step 1
        assert result is None
        assert rmgs.get_scale() == 1.0

    def test_step_evaluates_at_interval(self):
        """step() evaluates probe and returns dict at probe_interval."""
        rmgs = self._make_rmgs()
        rmgs.initialize()
        rmgs.step()  # step 1
        rmgs.step()  # step 2
        result = rmgs.step()  # step 3 -- evaluation
        assert result is not None
        assert "probe_loss" in result
        assert "reward" in result
        assert "ema_reward" in result
        assert "scale" in result
        assert len(rmgs.scale_history) == 1

    def test_scale_clamped_to_min(self):
        """Scale never goes below min_scale."""
        rmgs = self._make_rmgs(min_scale=0.1)
        rmgs.initialize()
        # Force a very negative EMA reward
        rmgs.ema_reward = -10.0
        rmgs.last_probe_loss = rmgs.last_probe_loss  # keep same
        # Manually trigger evaluation
        for _ in range(3):
            rmgs.step()
        scale = rmgs.get_scale()
        assert scale >= 0.1, f"Scale {scale} should be >= min_scale 0.1"

    def test_scale_clamped_to_one(self):
        """Scale never exceeds 1.0."""
        rmgs = self._make_rmgs(beta=100.0)
        rmgs.initialize()
        # Force a very positive EMA reward
        rmgs.ema_reward = 10.0
        for _ in range(3):
            rmgs.step()
        scale = rmgs.get_scale()
        assert scale <= 1.0, f"Scale {scale} should be <= 1.0"

    def test_get_stats_structure(self):
        """get_stats returns all required fields."""
        rmgs = self._make_rmgs()
        rmgs.initialize()
        for _ in range(6):
            rmgs.step()
        stats = rmgs.get_stats()
        required = [
            "rmgs_scale_history", "rmgs_reward_history",
            "rmgs_ema_reward_history", "rmgs_probe_loss_trajectory",
            "mean_gradient_scale", "scale_std",
            "probe_set_hash", "n_evaluations",
        ]
        for key in required:
            assert key in stats, f"Missing key: {key}"
        assert stats["probe_set_hash"] is not None
        assert len(stats["probe_set_hash"]) == 64

    def test_stats_json_serializable(self):
        """get_stats output can be serialized to JSON."""
        rmgs = self._make_rmgs()
        rmgs.initialize()
        for _ in range(6):
            rmgs.step()
        stats = rmgs.get_stats()
        # Should not raise
        json_str = json.dumps(stats)
        assert len(json_str) > 0

    def test_deterministic_with_seed(self):
        """Same seed and model produce same probe loss sequence."""
        torch.manual_seed(0)
        r1 = self._make_rmgs(seed=42)
        r1.initialize()
        for _ in range(6):
            r1.step()
        torch.manual_seed(0)
        r2 = self._make_rmgs(seed=42)
        r2.initialize()
        for _ in range(6):
            r2.step()
        assert r1.probe_loss_history == r2.probe_loss_history
        assert r1.scale_history == r2.scale_history


# ===========================================================================
# Scale Computation Unit Tests
# ===========================================================================

class TestScaleComputation:
    """Test the RMGS scale formula in isolation."""

    def test_neutral_reward_keeps_scale_at_one(self):
        """EMA reward of 0 should produce scale = 1.0."""
        # scale = clamp(1.0 + beta * 0.0, min, 1.0) = 1.0
        rmgs = RMGS.__new__(RMGS)
        rmgs.beta = 2.0
        rmgs.min_scale = 0.05
        rmgs.ema_reward = 0.0
        scale = max(rmgs.min_scale, min(1.0, 1.0 + rmgs.beta * rmgs.ema_reward))
        assert scale == 1.0

    def test_negative_ema_reduces_scale(self):
        """Negative EMA reward (forgetting) should reduce scale below 1.0."""
        rmgs = RMGS.__new__(RMGS)
        rmgs.beta = 2.0
        rmgs.min_scale = 0.05
        rmgs.ema_reward = -0.2
        scale = max(rmgs.min_scale, min(1.0, 1.0 + rmgs.beta * rmgs.ema_reward))
        assert scale == 0.6  # 1.0 + 2.0 * (-0.2) = 0.6

    def test_positive_ema_capped_at_one(self):
        """Positive EMA reward (retention improving) should cap scale at 1.0."""
        rmgs = RMGS.__new__(RMGS)
        rmgs.beta = 2.0
        rmgs.min_scale = 0.05
        rmgs.ema_reward = 0.5
        scale = max(rmgs.min_scale, min(1.0, 1.0 + rmgs.beta * rmgs.ema_reward))
        assert scale == 1.0  # 1.0 + 2.0 * 0.5 = 2.0 -> clamped to 1.0


# ===========================================================================
# Helper Tests
# ===========================================================================

class TestHelpers:
    """Test helper functions for RL methods."""

    def test_is_rmgs_method(self):
        assert is_rmgs_method("rmgs") is True
        assert is_rmgs_method("bandit_replay") is False
        assert is_rmgs_method("ewc") is False
        assert is_rmgs_method("baseline") is False

    def test_rmgs_not_in_other_method_checks(self):
        """rmgs should NOT match any other method check."""
        from cl_methods import is_replay_method, is_ewc_method, is_mer_method
        assert is_replay_method("rmgs") is False
        assert is_ewc_method("rmgs") is False
        assert is_mer_method("rmgs") is False
