#!/usr/bin/env python3
"""
Minimal Tests for Continual Learning Method Implementations

These tests verify that EWC, Replay, and MER are correctly implemented
using tiny models and fake data. Each test should complete in < 30 seconds.

Run with: python3 src/adhoc/_diagnostics/test_cl_methods.py
"""

import sys
import random
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cl_methods import (
    ReplayBuffer, create_replay_buffer, create_mixed_batch,
    EWC, MER
)


def header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  [{symbol}] {name}: {status}")
    if details:
        for line in details.split("\n"):
            print(f"      {line}")
    return passed


# =============================================================================
# Tiny Model for Testing
# =============================================================================

class TinyLM(nn.Module):
    """Minimal language model for testing (embedding + linear)."""

    def __init__(self, vocab_size=100, embed_dim=32, seq_len=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def forward(self, input_ids, labels=None):
        # input_ids: (batch, seq_len)
        x = self.embed(input_ids)  # (batch, seq_len, embed_dim)
        logits = self.linear(x)    # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            # Cross-entropy loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

        # Return object with .loss attribute like HuggingFace models
        class Output:
            pass
        out = Output()
        out.loss = loss
        out.logits = logits
        return out


def create_fake_tokens(n_tokens, vocab_size=100):
    """Create random token tensor."""
    return torch.randint(0, vocab_size, (n_tokens,))


# =============================================================================
# TEST 1: Replay Buffer
# =============================================================================

def test_replay_buffer():
    header("TEST 1: Replay Buffer")

    all_passed = True

    # Test 1.1: Buffer fills correctly
    buffer = ReplayBuffer(capacity=10, sequence_length=16)
    for i in range(15):
        seq = torch.randint(0, 100, (16,))
        buffer.add(seq)

    passed = len(buffer) == 10
    all_passed &= check("Buffer respects capacity", passed,
                        f"capacity=10, len={len(buffer)}")

    # Test 1.2: Sampling works
    samples = buffer.sample(5)
    passed = samples.shape == (5, 16)
    all_passed &= check("Sample returns correct shape", passed,
                        f"expected (5, 16), got {samples.shape}")

    # Test 1.3: create_replay_buffer from tokens
    tokens = create_fake_tokens(1000)
    buffer = create_replay_buffer(tokens, buffer_size_pct=10, sequence_length=16)
    expected_seqs = 1000 // 16  # 62 total sequences
    expected_capacity = int(expected_seqs * 0.10)  # ~6
    passed = len(buffer) <= expected_capacity + 1  # Allow +1 for rounding
    all_passed &= check("create_replay_buffer capacity", passed,
                        f"expected ~{expected_capacity}, got {len(buffer)}")

    # Test 1.4: Mixed batch creation
    tokens = create_fake_tokens(640)  # 40 sequences of 16
    buffer = create_replay_buffer(tokens, buffer_size_pct=50, sequence_length=16)

    new_batch = torch.randint(0, 100, (8, 16))  # batch_size=8
    mixed, n_new, n_replay = create_mixed_batch(new_batch, buffer, replay_rate=0.25)

    passed = mixed.shape == (8, 16)
    all_passed &= check("Mixed batch shape preserved", passed)

    passed = n_replay == 2  # 25% of 8 = 2
    all_passed &= check("Replay ratio honored", passed,
                        f"expected n_replay=2, got {n_replay}")

    return all_passed


# =============================================================================
# TEST 2: EWC Implementation
# =============================================================================

def test_ewc():
    header("TEST 2: EWC (Elastic Weight Consolidation)")

    all_passed = True
    device = "cpu"

    # Create tiny model
    model = TinyLM(vocab_size=100, embed_dim=32, seq_len=16)
    model.to(device)

    # Create fake Domain A data
    tokens_a = create_fake_tokens(320)  # 20 sequences

    # Test 2.1: Fisher computation
    ewc = EWC(model, device)
    ewc.compute_fisher(tokens_a, n_samples=10, batch_size=2, sequence_length=16)

    passed = ewc.is_computed
    all_passed &= check("Fisher computed", passed)

    passed = ewc.theta_star is not None and len(ewc.theta_star) > 0
    all_passed &= check("Anchor weights stored", passed,
                        f"stored {len(ewc.theta_star) if ewc.theta_star else 0} params")

    passed = ewc.fisher_diag is not None and len(ewc.fisher_diag) > 0
    all_passed &= check("Fisher diagonal stored", passed)

    # Check Fisher values are non-zero
    fisher_sum = sum(f.sum().item() for f in ewc.fisher_diag.values())
    passed = fisher_sum > 0
    all_passed &= check("Fisher values non-zero", passed,
                        f"sum={fisher_sum:.6f}")

    # Test 2.2: Penalty is zero when theta == theta*
    penalty_at_anchor = ewc.penalty(ewc_lambda=100.0)
    passed = penalty_at_anchor.item() < 1e-6
    all_passed &= check("Penalty=0 at anchor", passed,
                        f"penalty={penalty_at_anchor.item():.8f}")

    # Test 2.3: Penalty increases when weights change
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(0.1)  # Shift all weights by 0.1

    penalty_after_shift = ewc.penalty(ewc_lambda=100.0)
    passed = penalty_after_shift.item() > penalty_at_anchor.item()
    all_passed &= check("Penalty increases after weight shift", passed,
                        f"before={penalty_at_anchor.item():.4f}, after={penalty_after_shift.item():.4f}")

    # Test 2.4: Penalty scales with lambda
    penalty_lambda_10 = ewc.penalty(ewc_lambda=10.0)
    penalty_lambda_100 = ewc.penalty(ewc_lambda=100.0)
    ratio = penalty_lambda_100.item() / penalty_lambda_10.item()
    passed = abs(ratio - 10.0) < 0.1
    all_passed &= check("Penalty scales with lambda", passed,
                        f"ratio={ratio:.2f} (expected 10.0)")

    # Test 2.5: Penalty affects gradients
    model2 = TinyLM(vocab_size=100, embed_dim=32, seq_len=16)
    model2.to(device)
    ewc2 = EWC(model2, device)
    ewc2.compute_fisher(tokens_a, n_samples=10, batch_size=2, sequence_length=16)

    # Forward pass with EWC penalty
    input_ids = torch.randint(0, 100, (2, 16)).to(device)
    outputs = model2(input_ids, labels=input_ids)
    ce_loss = outputs.loss
    ewc_penalty = ewc2.penalty(ewc_lambda=100.0)
    total_loss = ce_loss + ewc_penalty

    # Backward
    model2.zero_grad()
    total_loss.backward()

    # Check gradients exist
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in model2.parameters())
    passed = has_grads
    all_passed &= check("EWC loss produces gradients", passed)

    return all_passed


# =============================================================================
# TEST 3: MER (Reptile Updates)
# =============================================================================

def test_mer():
    header("TEST 3: MER (Reptile Meta-Updates)")

    all_passed = True
    device = "cpu"

    # Create tiny model
    model = TinyLM(vocab_size=100, embed_dim=32, seq_len=16)
    model.to(device)

    # Initialize MER
    mer = MER(model, reptile_interval=5, reptile_epsilon=0.5)

    # Test 3.1: Snapshot creation
    mer.snapshot()
    passed = mer.theta_old is not None
    all_passed &= check("Snapshot taken", passed)

    # Save original weights
    original_weights = {name: param.clone() for name, param in model.named_parameters()}

    # Test 3.2: Simulate training steps
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for step in range(10):
        # Fake training step
        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # MER step
        mer.step()

    # Test 3.3: Reptile updates were applied
    passed = mer.update_counter == 2  # intervals at step 5 and 10
    all_passed &= check("Reptile updates counted", passed,
                        f"expected 2, got {mer.update_counter}")

    # Test 3.4: Weights changed but interpolated toward old
    # After Reptile: θ = θ_old + ε(θ - θ_old)
    # With ε=0.5, the new θ should be halfway between old snapshot and SGD result

    # Since we did 10 SGD steps, weights should have changed
    weights_changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, original_weights[name], atol=1e-6):
            weights_changed = True
            break
    passed = weights_changed
    all_passed &= check("Weights modified during training", passed)

    # Test 3.5: Reset works
    mer.reset()
    passed = mer.theta_old is None and mer.step_counter == 0 and mer.update_counter == 0
    all_passed &= check("MER reset works", passed)

    # Test 3.6: Reptile interpolation correctness
    model3 = TinyLM(vocab_size=100, embed_dim=32, seq_len=16)
    mer3 = MER(model3, reptile_interval=1, reptile_epsilon=0.5)

    # Take snapshot
    mer3.snapshot()
    theta_old = {name: param.clone() for name, param in model3.named_parameters()}

    # Manually shift weights
    with torch.no_grad():
        for param in model3.parameters():
            param.add_(1.0)  # Add 1.0 to all weights

    theta_before_reptile = {name: param.clone() for name, param in model3.named_parameters()}

    # Apply Reptile (should happen since step_counter will be 1)
    mer3.step()  # step 1, interval=1, so Reptile fires

    # Check: θ_new = θ_old + 0.5 * (θ_current - θ_old)
    #       = θ_old + 0.5 * 1.0 = θ_old + 0.5
    for name, param in model3.named_parameters():
        expected = theta_old[name] + 0.5  # θ_old + ε * shift
        passed = torch.allclose(param, expected, atol=1e-6)
        if not passed:
            diff = (param - expected).abs().max().item()
            all_passed &= check(f"Reptile interpolation for {name}", passed,
                              f"max diff={diff:.6f}")
            break
    else:
        all_passed &= check("Reptile interpolation correct", True,
                          "θ = θ_old + ε(θ - θ_old) verified")

    return all_passed


# =============================================================================
# TEST 4: Integration with Training Loop
# =============================================================================

def test_integration():
    header("TEST 4: Integration Test (Fake Training Loop)")

    all_passed = True
    device = "cpu"

    # Create model and fake data
    model = TinyLM(vocab_size=100, embed_dim=32, seq_len=16)
    model.to(device)

    tokens_a = create_fake_tokens(320)  # Domain A
    tokens_b = create_fake_tokens(320)  # Domain B

    # Simulate EWC + Replay + MER combined
    ewc = EWC(model, device)
    ewc.compute_fisher(tokens_a, n_samples=10, batch_size=2, sequence_length=16)

    replay_buffer = create_replay_buffer(tokens_a, buffer_size_pct=50, sequence_length=16)

    mer = MER(model, reptile_interval=3, reptile_epsilon=0.3)

    # Training on Domain B
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    total_ewc_penalty = 0.0
    replay_samples = 0

    # Reshape tokens_b into sequences
    n_seqs = len(tokens_b) // 16
    tokens_b_seqs = tokens_b[:n_seqs * 16].view(n_seqs, 16)

    for step in range(10):
        # Get batch from Domain B
        batch_idx = step % (n_seqs // 4)
        new_batch = tokens_b_seqs[batch_idx * 4:(batch_idx + 1) * 4]

        # Mix with replay
        mixed_batch, n_new, n_replay = create_mixed_batch(
            new_batch, replay_buffer, replay_rate=0.25
        )
        replay_samples += n_replay

        # Forward
        outputs = model(mixed_batch, labels=mixed_batch)
        ce_loss = outputs.loss
        ewc_penalty = ewc.penalty(ewc_lambda=50.0)
        total_loss = ce_loss + ewc_penalty
        total_ewc_penalty += ewc_penalty.item()

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # MER step
        mer.step()

    # Verify all components were exercised
    passed = total_ewc_penalty > 0
    all_passed &= check("EWC penalty accumulated", passed,
                        f"total={total_ewc_penalty:.4f}")

    passed = replay_samples > 0
    all_passed &= check("Replay samples used", passed,
                        f"count={replay_samples}")

    # At step 3: first snapshot taken (no update yet)
    # At step 6: first Reptile update
    # At step 9: second Reptile update
    passed = mer.update_counter >= 2
    all_passed &= check("MER updates applied", passed,
                        f"updates={mer.update_counter}")

    return all_passed


# =============================================================================
# TEST 5: Forgetting Calculation (from diagnostics)
# =============================================================================

def test_forgetting_formula():
    header("TEST 5: Forgetting Formula Verification")

    all_passed = True

    # Import the formula
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from metrics import MetricsComputer

    # Test case 1: Known values
    ppl_before = 22.93
    ppl_after = 42.87
    expected = ((42.87 - 22.93) / 22.93) * 100  # ~86.95%
    computed = MetricsComputer.compute_forgetting_pct(ppl_before, ppl_after)

    passed = abs(computed - expected) < 0.01
    all_passed &= check("Forgetting formula", passed,
                        f"expected={expected:.4f}%, computed={computed:.4f}%")

    # Test case 2: No forgetting
    ppl_before = 25.0
    ppl_after = 25.0
    computed = MetricsComputer.compute_forgetting_pct(ppl_before, ppl_after)
    passed = abs(computed) < 0.01
    all_passed &= check("Zero forgetting case", passed,
                        f"computed={computed:.4f}%")

    # Test case 3: Negative forgetting (improvement)
    ppl_before = 30.0
    ppl_after = 25.0
    computed = MetricsComputer.compute_forgetting_pct(ppl_before, ppl_after)
    passed = computed < 0
    all_passed &= check("Negative forgetting (improvement)", passed,
                        f"computed={computed:.4f}%")

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  CONTINUAL LEARNING METHOD TESTS")
    print("=" * 60)

    # Set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    results = []

    results.append(("Replay Buffer", test_replay_buffer()))
    results.append(("EWC", test_ewc()))
    results.append(("MER", test_mer()))
    results.append(("Integration", test_integration()))
    results.append(("Forgetting Formula", test_forgetting_formula()))

    # Summary
    header("SUMMARY")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n  Tests passed: {passed}/{total}")
    print()

    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  [{symbol}] {name}: {status}")

    print()

    if passed == total:
        print("  All tests passed! CL methods implemented correctly.")
    else:
        print("  Some tests failed. Check implementation.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
