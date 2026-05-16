#!/usr/bin/env python3
"""
Integration smoke for AMENDMENT-004 (replay-rounding fix).

The pre-fix bug only manifested when `batch_size * replay_rate` was
non-integer, which for the locked protocol happens on 1B-class runs
(`batch_size=2`, `replay_rate=0.25`). This file drives the actual
`CPTTrainer.train_domain_b` loop at that exact configuration with a tiny
synthetic model, asserts the realised replay count is non-zero, and
verifies the new per-run telemetry fields are present in both the periodic
training-log entries and the final stats block.

This stands in for the AMENDMENT-004 §Verification-Gate-3 smoke ("Qwen3
replay25 shows replay_samples > 0 averaged over an 8-step accumulation
window") until the GCP CUDA environment is up; the real Qwen3 smoke runs
as the first step of Stage 1 once cloud access is available. The intent
is identical: prove that the end-to-end plumbing (helper → trainer → log)
yields a non-zero effective replay count at `batch_size=2, replay_rate=0.25`.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trainer import CPTTrainer  # noqa: E402


class TinyLM(nn.Module):
    """Minimal causal LM compatible with the trainer's `model(input_ids, labels=...)` call."""

    def __init__(self, vocab_size: int = 64, hidden: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, labels=None):
        h = self.embed(input_ids)
        logits = self.linear(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1)
            )
        return type("Out", (), {"logits": logits, "loss": loss})()


def _config(batch_size: int, replay_rate: float, seed: int = 7):
    return {
        "method": "replay25",
        "method_params": {
            "buffer_size_pct": 20,
            "mixing_ratio_replay": replay_rate,
        },
        "data": {"sequence_length": 16},
        "logging": {"eval_steps": 100_000},
        "seed": seed,
        "model": {"gradient_checkpointing": False},
    }


def _domain_b_cfg(batch_size: int):
    return {
        "learning_rate": 1.0e-3,
        "batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
    }


def _run(batch_size: int, replay_rate: float, total_microbatches: int = 32):
    torch.manual_seed(0)
    model = TinyLM()
    tokenizer = None
    cfg = _config(batch_size, replay_rate)
    trainer = CPTTrainer(model=model, tokenizer=tokenizer, config=cfg, device="cpu")

    # Manually wire up the replay buffer (the full setup pipeline lives in
    # train_with_methods; we exercise just the train_domain_b loop).
    from cl_methods import create_replay_buffer
    tokens_a = torch.randint(0, 64, (batch_size * total_microbatches * 16,))
    trainer.replay_buffer = create_replay_buffer(
        tokens_a, buffer_size_pct=20, sequence_length=16
    )

    # Domain B tokens: enough sequences to cover total_microbatches at batch_size.
    n_seqs = batch_size * total_microbatches
    tokens_b = torch.randint(0, 64, (n_seqs * 16,))

    domain_cfg = _domain_b_cfg(batch_size)
    grad_accum = domain_cfg["gradient_accumulation_steps"]
    max_steps = total_microbatches // grad_accum  # gradient steps

    stats = trainer.train_domain_b(
        tokens_b, domain_cfg, max_steps=max_steps, log_steps=1
    )
    return trainer, stats


def test_batch2_rate25_yields_nonzero_replay_in_trainer_loop():
    """The bug-trigger config: batch=2 * rate=0.25 must produce >0 replay samples."""
    trainer, stats = _run(batch_size=2, replay_rate=0.25, total_microbatches=32)
    assert stats.get("effective_replay_samples", 0) > 0, (
        "AMENDMENT-004 regression: effective_replay_samples is 0 at "
        f"batch_size=2, rate=0.25. Stats: {stats}"
    )
    # Long-run effective rate should be near nominal 0.25 within tolerance.
    # 32 microbatches × 2 slots = 64 Bernoulli trials at p=0.5; std ≈ 0.06.
    eff = stats["effective_replay_rate"]
    assert 0.10 <= eff <= 0.40, (
        f"effective_replay_rate {eff:.3f} far from nominal 0.25 over 32 microbatches"
    )
    # The seed is recorded so future audits can reproduce the trace.
    assert "replay_rng_seed" in stats


def test_batch4_rate25_integer_case_unaffected():
    """batch=4 * rate=0.25 = 1.0 exactly: pre-fix and post-fix produce identical counts."""
    _, stats = _run(batch_size=4, replay_rate=0.25, total_microbatches=16)
    # 16 microbatches × 1 replay each = 16 replay samples total.
    assert stats["effective_replay_samples"] == 16
    assert abs(stats["effective_replay_rate"] - 0.25) < 1e-9


def test_training_log_entries_carry_new_fields():
    """Periodic log entries (training_log.jsonl content) must expose the new telemetry."""
    trainer, _ = _run(batch_size=2, replay_rate=0.25, total_microbatches=32)
    replay_entries = [e for e in trainer.training_log if e.get("domain") == "domain_b"]
    assert replay_entries, "No domain_b log entries produced"
    sample = replay_entries[-1]
    for required in (
        "nominal_replay_rate",
        "effective_replay_samples",
        "effective_replay_rate",
        "replay_rng_seed",
    ):
        assert required in sample, f"Missing telemetry field: {required}"
    assert sample["nominal_replay_rate"] == 0.25


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
