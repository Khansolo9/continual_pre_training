#!/usr/bin/env python3
"""
PPL Batch-Size Invariance Test

Verifies that compute_ppl() returns the same value regardless of batch_size,
since the implementation uses token-weighted NLL (not batch-averaged).

Uses a known GPT-2 smoke checkpoint to compare batch_size=1 vs batch_size=8.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

from metrics import MetricsComputer


PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="module")
def gpt2_checkpoint(device):
    """Load the GPT-2 smoke baseline checkpoint (theta_A)."""
    ckpt_path = PROJECT_ROOT / "experiments" / "runs" / "gpt2_smoke_baseline_s1" / "checkpoints" / "theta_A.pt"
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def valid_tokens():
    """Load GPT-2 validation tokens."""
    path = PROJECT_ROOT / "data" / "eval" / "wikitext103_valid_tokens.pt"
    if not path.exists():
        pytest.skip(f"Validation tokens not found: {path}")
    tokens = torch.load(path, weights_only=True)
    # Use a small subset to keep the test fast
    return tokens[:8192]


# =============================================================================
# TESTS
# =============================================================================

class TestPPLBatchInvariance:
    """PPL should be identical across batch sizes for token-weighted NLL."""

    def test_batch_1_vs_8(self, gpt2_checkpoint, valid_tokens, device):
        """Core invariance test: batch_size=1 vs batch_size=8."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        mc = MetricsComputer(model=gpt2_checkpoint, tokenizer=tokenizer, device=device)

        result_bs1 = mc.compute_ppl(valid_tokens, batch_size=1, sequence_length=512)
        result_bs8 = mc.compute_ppl(valid_tokens, batch_size=8, sequence_length=512)

        ppl_bs1 = result_bs1["ppl_primary"]
        ppl_bs8 = result_bs8["ppl_primary"]

        # Token counts must be identical
        assert result_bs1["total_tokens"] == result_bs8["total_tokens"]

        # PPL values should be identical (token-weighted, not batch-averaged)
        rel_diff = abs(ppl_bs1 - ppl_bs8) / ppl_bs8
        assert rel_diff < 1e-4, (
            f"PPL mismatch: bs=1 gave {ppl_bs1:.6f}, bs=8 gave {ppl_bs8:.6f}, "
            f"relative diff={rel_diff:.2e}"
        )

    def test_batch_1_vs_4(self, gpt2_checkpoint, valid_tokens, device):
        """Additional invariance check with batch_size=4."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        mc = MetricsComputer(model=gpt2_checkpoint, tokenizer=tokenizer, device=device)

        result_bs1 = mc.compute_ppl(valid_tokens, batch_size=1, sequence_length=512)
        result_bs4 = mc.compute_ppl(valid_tokens, batch_size=4, sequence_length=512)

        rel_diff = abs(result_bs1["ppl_primary"] - result_bs4["ppl_primary"]) / result_bs4["ppl_primary"]
        assert rel_diff < 1e-4
