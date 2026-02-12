#!/usr/bin/env python3
"""
Manifest Path Resolution Tests

Verifies that _manifest_path():
1. Prefers data/manifests/{model_family}/domain_a.json when it exists
2. Falls back to data/manifests/domain_a.json when model-specific dir is missing
3. Handles empty model_family gracefully
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from pathlib import Path
from unittest.mock import MagicMock


# We can't easily instantiate ExperimentRunner (needs model download),
# so we test the resolution logic directly by extracting it.

def manifest_path_logic(data_dir: Path, model_family: str, filename: str) -> Path:
    """Replicate _manifest_path() logic from run_experiment.py for testing."""
    if model_family:
        model_path = data_dir / "manifests" / model_family / filename
        if model_path.exists():
            return model_path
    return data_dir / "manifests" / filename


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# TESTS
# =============================================================================

class TestManifestPathResolution:
    """Test the manifest path fallback logic."""

    def test_model_specific_path_preferred(self):
        """When data/manifests/gpt2/domain_a.json exists, use it."""
        result = manifest_path_logic(DATA_DIR, "gpt2", "domain_a.json")
        expected = DATA_DIR / "manifests" / "gpt2" / "domain_a.json"
        assert result == expected
        assert result.exists(), f"Expected {result} to exist on disk"

    def test_all_model_families_have_manifests(self):
        """Every model family should have both domain manifests."""
        for family in ("gpt2", "gemma3", "qwen3", "llama3"):
            for domain in ("domain_a.json", "domain_b.json"):
                result = manifest_path_logic(DATA_DIR, family, domain)
                expected = DATA_DIR / "manifests" / family / domain
                assert result == expected, f"Expected model-specific path for {family}/{domain}"
                assert result.exists(), f"Manifest missing: {result}"

    def test_fallback_when_family_dir_missing(self, tmp_path):
        """When model-specific dir doesn't exist, fall back to root manifests."""
        # Create only root-level manifest
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        (manifests_dir / "domain_a.json").write_text("{}")

        result = manifest_path_logic(tmp_path, "nonexistent_model", "domain_a.json")
        expected = tmp_path / "manifests" / "domain_a.json"
        assert result == expected

    def test_fallback_when_family_empty_string(self, tmp_path):
        """When model_family is empty, skip model-specific check."""
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        (manifests_dir / "domain_a.json").write_text("{}")

        result = manifest_path_logic(tmp_path, "", "domain_a.json")
        expected = tmp_path / "manifests" / "domain_a.json"
        assert result == expected

    def test_legacy_root_manifests_exist(self):
        """Legacy root manifests should still exist for backward compat."""
        assert (DATA_DIR / "manifests" / "domain_a.json").exists()
        assert (DATA_DIR / "manifests" / "domain_b.json").exists()
