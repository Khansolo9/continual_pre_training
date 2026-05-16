#!/usr/bin/env python3
"""
Tests for AMENDMENT-004 (replay-rounding fix).

The pre-AMENDMENT-004 implementation computed `n_replay = int(batch_size *
replay_rate)`. For batch_size=2, replay_rate=0.25 this is `int(0.5) == 0`,
silently producing zero replay samples per microbatch on all 1B-class runs.
See `experiments/analysis/run_validity_audit.md` for which runs were
invalidated by this bug.

The fix is stochastic fractional rounding:

    expected = batch_size * replay_rate
    n_floor  = floor(expected)
    frac     = expected - n_floor
    n_replay = n_floor + Bernoulli(frac)

so that E[n_replay] == batch_size * replay_rate exactly, with a single
microbatch's count an integer (the model can only consume integer batches).

These tests cover:
1. The previously-broken case (batch=2, rate=0.25) now yields ≈25% replay
   in expectation, with both 0 and 1 outcomes observed.
2. Integer cases (batch=4, rate=0.25; batch=2, rate=0.5) are deterministic.
3. Boundary rates (0.0, 1.0) are deterministic.
4. Empty buffer short-circuits to no replay regardless of rate.
5. RNG is reproducible across runs with the same seed.
6. The fractional draw uses the supplied `random.Random` only, leaving
   torch's global generator state untouched.
"""

import sys
import random
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cl_methods import create_mixed_batch, ReplayBuffer


def _make_buffer(seq_len: int = 8, capacity: int = 64):
    buf = ReplayBuffer(capacity=capacity, sequence_length=seq_len)
    buf.add_batch(torch.zeros(capacity, seq_len, dtype=torch.long))
    return buf


def _make_new_batch(batch_size: int, seq_len: int = 8) -> torch.Tensor:
    # Non-zero token ids so a mixed batch is distinguishable from pure replay
    # (replay buffer is zero-filled above).
    return torch.full((batch_size, seq_len), 7, dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. The bug-trigger case: batch=2, rate=0.25
# ---------------------------------------------------------------------------

def test_batch2_rate25_yields_quarter_in_expectation():
    """Mean realized count over many draws should converge to 0.5 = 2*0.25.

    Tolerance: ±0.05 (i.e. effective rate in [0.20, 0.30]) over 10_000
    trials with a Bernoulli(0.5) generator. Loose enough to not flake,
    tight enough to fail the pre-AMENDMENT-004 implementation (which
    would deterministically yield 0).
    """
    rng = random.Random(0xA40004)
    buf = _make_buffer()
    counts = []
    for _ in range(10_000):
        _, _, n_replay = create_mixed_batch(
            _make_new_batch(2), buf, replay_rate=0.25, rng=rng
        )
        counts.append(n_replay)
    mean = sum(counts) / len(counts)
    assert 0.45 <= mean <= 0.55, f"Mean replay count {mean:.3f} far from 0.5"
    # Sanity: both outcomes occur.
    assert 0 in counts and 1 in counts, "Bernoulli not producing both outcomes"


# ---------------------------------------------------------------------------
# 2. Integer cases must be deterministic (no spurious randomness)
# ---------------------------------------------------------------------------

def test_batch4_rate25_always_one():
    """batch=4 * 0.25 = 1.0 exactly: must always return 1, no Bernoulli draw."""
    rng = random.Random(1)
    buf = _make_buffer()
    for _ in range(100):
        _, _, n_replay = create_mixed_batch(
            _make_new_batch(4), buf, replay_rate=0.25, rng=rng
        )
        assert n_replay == 1


def test_batch2_rate50_always_one():
    rng = random.Random(1)
    buf = _make_buffer()
    for _ in range(100):
        _, _, n_replay = create_mixed_batch(
            _make_new_batch(2), buf, replay_rate=0.5, rng=rng
        )
        assert n_replay == 1


# ---------------------------------------------------------------------------
# 3. Boundary rates
# ---------------------------------------------------------------------------

def test_rate_zero_always_zero():
    rng = random.Random(1)
    buf = _make_buffer()
    for _ in range(50):
        _, _, n_replay = create_mixed_batch(
            _make_new_batch(2), buf, replay_rate=0.0, rng=rng
        )
        assert n_replay == 0


def test_rate_one_always_full_batch():
    rng = random.Random(1)
    buf = _make_buffer()
    for _ in range(50):
        _, _, n_replay = create_mixed_batch(
            _make_new_batch(2), buf, replay_rate=1.0, rng=rng
        )
        assert n_replay == 2


# ---------------------------------------------------------------------------
# 4. Empty replay buffer short-circuits
# ---------------------------------------------------------------------------

def test_empty_buffer_short_circuits():
    rng = random.Random(1)
    empty = ReplayBuffer(capacity=64, sequence_length=8)
    mixed, n_new, n_replay = create_mixed_batch(
        _make_new_batch(2), empty, replay_rate=0.25, rng=rng
    )
    assert n_replay == 0
    assert n_new == 2
    assert mixed.shape == (2, 8)


# ---------------------------------------------------------------------------
# 5. Reproducibility under a fixed seed
# ---------------------------------------------------------------------------

def test_seeded_rng_reproducible():
    buf = _make_buffer()
    trace_a = []
    rng_a = random.Random(123)
    for _ in range(500):
        _, _, n = create_mixed_batch(_make_new_batch(2), buf, 0.25, rng=rng_a)
        trace_a.append(n)
    trace_b = []
    rng_b = random.Random(123)
    for _ in range(500):
        _, _, n = create_mixed_batch(_make_new_batch(2), buf, 0.25, rng=rng_b)
        trace_b.append(n)
    assert trace_a == trace_b, "Same seed produced different replay traces"


# ---------------------------------------------------------------------------
# 6. Bernoulli draw does not perturb torch's global RNG
# ---------------------------------------------------------------------------

def test_does_not_touch_torch_global_rng():
    """Replay draws must not consume entropy from torch's global generator.

    Reason: data-shuffle and dropout streams rely on torch's global generator
    being in a deterministic state at every training step. If the fractional
    draw advanced that state, two otherwise-identical runs (with the bug fix
    vs. without) would diverge in unrelated ways and the audit-trail of
    "what changed" would be muddied.
    """
    torch.manual_seed(0)
    expected = torch.randn(8)
    torch.manual_seed(0)
    # Burn some entropy from a private random.Random
    rng = random.Random(999)
    buf = _make_buffer()
    for _ in range(50):
        create_mixed_batch(_make_new_batch(2), buf, 0.25, rng=rng)
    after = torch.randn(8)
    assert torch.equal(expected, after), (
        "Replay draw consumed entropy from torch's global generator"
    )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
