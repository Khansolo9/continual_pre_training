#!/usr/bin/env python3
"""
Tests for BanditReplay (Bandit-Guided Replay Scheduling).

Covers:
- Deterministic probe construction and stable hash
- EXP3 arm selection and update
- Replay rate tracking and history
- Config merge behavior for bandit_replay method
"""

import sys
import random
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cl_methods import (
    BanditReplay, ReplayBuffer, create_replay_buffer, create_probe_set,
    evaluate_probe, is_bandit_replay_method, create_mixed_batch,
)


# ---------------------------------------------------------------------------
# Tiny model for unit tests (avoids loading GPT-2)
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
    """Create reproducible random tokens."""
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=rng)


# ===========================================================================
# Probe Set Tests
# ===========================================================================

class TestProbeSet:
    """Test deterministic probe set construction."""

    def test_deterministic_same_seed(self):
        """Same seed produces identical probe sets."""
        tokens = _make_tokens(20000)
        p1, h1, i1 = create_probe_set(tokens, probe_size=20, sequence_length=512, seed=42)
        p2, h2, i2 = create_probe_set(tokens, probe_size=20, sequence_length=512, seed=42)
        assert h1 == h2, "Hashes should match for same seed"
        assert i1 == i2, "Indices should match for same seed"
        assert torch.equal(p1, p2), "Probe tensors should be identical"

    def test_different_seeds(self):
        """Different seeds produce different probe sets."""
        tokens = _make_tokens(20000)
        _, h1, i1 = create_probe_set(tokens, probe_size=20, sequence_length=512, seed=42)
        _, h2, i2 = create_probe_set(tokens, probe_size=20, sequence_length=512, seed=123)
        assert h1 != h2, "Different seeds should produce different hashes"
        assert i1 != i2, "Different seeds should produce different indices"

    def test_hash_is_sha256(self):
        """Probe hash is a valid SHA256 hex digest."""
        tokens = _make_tokens(20000)
        _, h, _ = create_probe_set(tokens, probe_size=10, sequence_length=512, seed=1)
        assert len(h) == 64, f"SHA256 hex should be 64 chars, got {len(h)}"
        assert all(c in '0123456789abcdef' for c in h)

    def test_probe_shape(self):
        """Probe tensor has correct shape."""
        tokens = _make_tokens(20000)
        p, _, _ = create_probe_set(tokens, probe_size=15, sequence_length=512, seed=1)
        assert p.shape == (15, 512)

    def test_probe_from_validation_not_coupled(self):
        """Probe creation doesn't affect global random state."""
        tokens = _make_tokens(20000)
        random.seed(99)
        r1 = random.random()
        random.seed(99)
        _ = create_probe_set(tokens, probe_size=10, sequence_length=512, seed=42)
        r2 = random.random()
        assert r1 == r2, "Global random state should not be affected by probe creation"


# ===========================================================================
# Evaluate Probe Tests
# ===========================================================================

class TestEvaluateProbe:
    """Test probe evaluation function."""

    def test_returns_finite_loss(self):
        """Probe evaluation returns finite NLL."""
        model = TinyLM()
        tokens = _make_tokens(5120)
        probe, _, _ = create_probe_set(tokens, probe_size=5, sequence_length=512, seed=1)
        loss = evaluate_probe(model, probe, "cpu")
        assert loss > 0 and loss < 100, f"Loss should be finite positive, got {loss}"

    def test_deterministic(self):
        """Same model and probe produce same loss."""
        model = TinyLM()
        tokens = _make_tokens(5120)
        probe, _, _ = create_probe_set(tokens, probe_size=5, sequence_length=512, seed=1)
        l1 = evaluate_probe(model, probe, "cpu")
        l2 = evaluate_probe(model, probe, "cpu")
        assert abs(l1 - l2) < 1e-6, f"Losses should match: {l1} vs {l2}"


# ===========================================================================
# BanditReplay Tests
# ===========================================================================

class TestBanditReplay:
    """Test BanditReplay arm selection and update."""

    def _make_bandit(self, seed=42):
        """Create a BanditReplay instance for testing."""
        model = TinyLM()
        tokens_a = _make_tokens(10000)
        valid_a = _make_tokens(5120)
        buffer = create_replay_buffer(tokens_a, buffer_size_pct=10, sequence_length=512)
        probe, probe_hash, _ = create_probe_set(valid_a, probe_size=5, sequence_length=512, seed=seed)
        bandit = BanditReplay(
            model=model, device="cpu", replay_buffer=buffer,
            arms=[0.0, 0.1, 0.25, 0.5, 0.75],
            initial_weights=[1, 1, 2, 1, 1],
            probe_set=probe, probe_hash=probe_hash,
            probe_interval=3, exp3_gamma=0.1, seed=seed,
        )
        return bandit

    def test_initialization(self):
        """Bandit initializes with probe loss and selects first arm."""
        bandit = self._make_bandit()
        bandit.initialize()
        assert bandit.last_probe_loss is not None
        assert bandit.last_probe_loss > 0
        assert bandit.current_arm_idx is not None
        assert bandit.current_rate in bandit.arms

    def test_probabilities_sum_to_one(self):
        """EXP3 probabilities sum to 1."""
        bandit = self._make_bandit()
        probs = bandit._get_probabilities()
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_warm_start_bias(self):
        """Warm-start weights bias toward arm 2 (0.25 rate)."""
        bandit = self._make_bandit()
        probs = bandit._get_probabilities()
        # Arm 2 (index 2, rate=0.25) should have highest probability
        assert probs[2] > probs[0], "0.25 arm should be more likely than 0.0"
        assert probs[2] > probs[4], "0.25 arm should be more likely than 0.75"

    def test_step_returns_none_between_intervals(self):
        """step() returns None when not at probe_interval boundary."""
        bandit = self._make_bandit()
        bandit.initialize()
        result = bandit.step()  # step 1, interval is 3
        assert result is None

    def test_step_evaluates_at_interval(self):
        """step() evaluates probe and returns dict at probe_interval."""
        bandit = self._make_bandit()
        bandit.initialize()
        bandit.step()  # step 1
        bandit.step()  # step 2
        result = bandit.step()  # step 3 -- evaluation point
        assert result is not None
        assert "probe_loss" in result
        assert "reward" in result
        assert len(bandit.arm_history) == 1

    def test_get_stats_structure(self):
        """get_stats returns all required fields."""
        bandit = self._make_bandit()
        bandit.initialize()
        for _ in range(6):
            bandit.step()
        stats = bandit.get_stats()
        required = [
            "bandit_arm_history", "bandit_reward_history",
            "bandit_arm_weights_final", "mean_replay_rate",
            "replay_rate_std", "probe_loss_trajectory",
            "probe_set_hash", "n_evaluations", "arms",
        ]
        for key in required:
            assert key in stats, f"Missing key: {key}"
        assert stats["probe_set_hash"] is not None
        assert len(stats["probe_set_hash"]) == 64

    def test_deterministic_with_seed(self):
        """Same seed produces same arm selection sequence."""
        b1 = self._make_bandit(seed=42)
        b1.initialize()
        for _ in range(9):
            b1.step()
        b2 = self._make_bandit(seed=42)
        b2.initialize()
        for _ in range(9):
            b2.step()
        assert b1.arm_history == b2.arm_history

    def test_mixed_batch_with_dynamic_rate(self):
        """create_mixed_batch works with bandit-selected rates."""
        bandit = self._make_bandit()
        bandit.initialize()
        rate = bandit.get_current_rate()
        batch = torch.randint(0, 100, (4, 512))
        mixed, n_new, n_replay = create_mixed_batch(batch, bandit.replay_buffer, rate)
        assert mixed.shape[0] == 4
        assert n_new + n_replay == 4


# ===========================================================================
# Helper Tests
# ===========================================================================

class TestHelpers:
    """Test helper functions for RL methods."""

    def test_is_bandit_replay_method(self):
        assert is_bandit_replay_method("bandit_replay") is True
        assert is_bandit_replay_method("replay25") is False
        assert is_bandit_replay_method("baseline") is False
        assert is_bandit_replay_method("rmgs") is False

    def test_bandit_replay_not_in_replay_method(self):
        """bandit_replay should NOT match is_replay_method."""
        from cl_methods import is_replay_method
        assert is_replay_method("bandit_replay") is False
