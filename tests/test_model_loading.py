#!/usr/bin/env python3
"""
Multi-Model Loading Tests

Verifies that each model family config can:
1. Load tokenizer with correct settings
2. Load model with correct dtype
3. Run a forward pass producing finite logits of the expected shape
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "models"

# Model configs to test — parametrize by filename
MODEL_CONFIGS = [
    "gpt2.yaml",
    "gemma3_1b.yaml",
    "qwen3_0.6b.yaml",
    "llama3.2_1b.yaml",
]


def load_model_config(filename: str) -> dict:
    """Load a model config YAML."""
    path = CONFIGS_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# TEST: Config Structure
# =============================================================================

class TestConfigStructure:
    """Verify all model configs have required fields."""

    @pytest.mark.parametrize("config_file", MODEL_CONFIGS)
    def test_required_fields(self, config_file):
        cfg = load_model_config(config_file)
        model = cfg["model"]
        assert "name" in model, f"Missing model.name in {config_file}"
        assert "family" in model, f"Missing model.family in {config_file}"
        assert "params_m" in model, f"Missing model.params_m in {config_file}"
        assert "dtype" in model, f"Missing model.dtype in {config_file}"
        assert "trust_remote_code" in model, f"Missing model.trust_remote_code in {config_file}"

    @pytest.mark.parametrize("config_file", MODEL_CONFIGS)
    def test_evaluation_overrides_present(self, config_file):
        cfg = load_model_config(config_file)
        assert "evaluation_overrides" in cfg, f"Missing evaluation_overrides in {config_file}"
        eo = cfg["evaluation_overrides"]
        assert eo.get("ppl_batch_size") == 1, f"ppl_batch_size != 1 in {config_file}"
        assert eo.get("ppl_sequence_length") == 512, f"ppl_sequence_length != 512 in {config_file}"

    @pytest.mark.parametrize("config_file", ["gemma3_1b.yaml", "qwen3_0.6b.yaml", "llama3.2_1b.yaml"])
    def test_training_overrides_present(self, config_file):
        cfg = load_model_config(config_file)
        assert "training_overrides" in cfg, f"Missing training_overrides in {config_file}"
        to = cfg["training_overrides"]
        assert to["batch_size"] == 2
        assert to["gradient_accumulation_steps"] == 8


# =============================================================================
# TEST: Tokenizer Loading
# =============================================================================

class TestTokenizerLoading:
    """Verify tokenizers load and have pad_token."""

    @pytest.mark.parametrize("config_file", MODEL_CONFIGS)
    def test_tokenizer_loads(self, config_file):
        cfg = load_model_config(config_file)
        model_cfg = cfg["model"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        assert tokenizer is not None

        # Guard pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token is not None

    @pytest.mark.parametrize("config_file", MODEL_CONFIGS)
    def test_tokenizer_encodes(self, config_file):
        cfg = load_model_config(config_file)
        model_cfg = cfg["model"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)


# =============================================================================
# TEST: Model Loading and Forward Pass
# =============================================================================

class TestModelForwardPass:
    """Verify models load and produce finite logits."""

    @pytest.mark.parametrize("config_file", MODEL_CONFIGS)
    def test_model_loads_and_produces_logits(self, config_file, device):
        cfg = load_model_config(config_file)
        model_cfg = cfg["model"]

        dtype_str = model_cfg.get("dtype", "float32")
        torch_dtype = getattr(torch, dtype_str, torch.float32)

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            torch_dtype=torch_dtype,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        model = model.to(device)
        model.eval()

        # Tiny forward pass
        input_ids = torch.randint(0, 100, (1, 16), device=device)
        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs.logits
        assert logits.shape[0] == 1, "Batch dim mismatch"
        assert logits.shape[1] == 16, "Sequence length mismatch"
        assert logits.shape[2] > 0, "Vocab dim should be > 0"
        assert torch.isfinite(logits).all(), f"Non-finite logits in {config_file}"

        # Cleanup
        del model, outputs, logits
        if device == "mps":
            torch.mps.empty_cache()
