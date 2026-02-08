#!/usr/bin/env python3
"""
Experiment Runner for Continual Pretraining

Executes a single run according to docs/RUNBOOK.md:
1. Load config and register run
2. Train on Domain A
3. Evaluate and save checkpoint
4. Train on Domain B
5. Final evaluation
6. Write metrics.json and runpack

Usage:
    python src/run_experiment.py --run-id pilot_baseline_s0 --config configs/methods/baseline.yaml
"""

import os
import sys
import json
import yaml
import shutil
import argparse
import logging
import csv
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pickle

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from trainer import CPTTrainer, create_model
from metrics import MetricsComputer, load_prompts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load dataset manifest."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def update_run_registry(
    registry_path: Path,
    run_id: str,
    updates: Dict[str, Any]
):
    """Update run registry CSV with run status."""
    rows = []
    headers = None

    # Read existing
    with open(registry_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            if row['run_id'] == run_id:
                row.update(updates)
            rows.append(row)

    # Write back
    with open(registry_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Registry updated for {run_id}")


class CPUNotAllowedError(Exception):
    """Raised when CPU-only mode is detected but --allow-cpu not specified."""
    pass


def choose_device(allow_cpu: bool = False) -> str:
    """
    Choose compute device with priority: CUDA > MPS > CPU.

    Args:
        allow_cpu: If True, allow CPU fallback. If False, raise error when no GPU.

    Returns:
        Device string: "cuda", "mps", or "cpu"

    Raises:
        CPUNotAllowedError: If no GPU available and allow_cpu=False
    """
    # Priority 1: CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return "cuda"

    # Priority 2: MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"

    # Priority 3: CPU (requires --allow-cpu flag)
    if not allow_cpu:
        raise CPUNotAllowedError(
            "\n" + "=" * 60 + "\n"
            "ERROR: No GPU detected (CUDA or MPS unavailable).\n"
            "CPU-only training is VERY slow and not recommended.\n\n"
            "To proceed anyway, re-run with: --allow-cpu\n"
            + "=" * 60
        )

    return "cpu"


def log_device_info(device: str) -> None:
    """
    Log GPU preflight information.

    Args:
        device: Selected device ("cuda", "mps", or "cpu")
    """
    print("\n" + "=" * 60)
    print("PRE-RUN GPU PREFLIGHT")
    print("=" * 60)

    # PyTorch version
    print(f"  PyTorch version:    {torch.__version__}")

    # CUDA info
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available:     {cuda_available}")
    if cuda_available:
        print(f"  CUDA device count:  {torch.cuda.device_count()}")
        print(f"  CUDA device name:   {torch.cuda.get_device_name(0)}")
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  Total VRAM:         {total_vram:.2f} GB")
        except Exception:
            print(f"  Total VRAM:         (unable to query)")

    # MPS info (Apple Silicon)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"  MPS available:      {mps_available}")
    if mps_available and device == "mps":
        print(f"  MPS in use:         True")

    # Selected device
    print("-" * 60)
    print(f"  SELECTED DEVICE:    {device.upper()}")

    # Warning banner for CPU
    if device == "cpu":
        print("-" * 60)
        print("  ⚠️  WARNING: Running on CPU only!")
        print("  ⚠️  Training will be significantly slower.")
        print("  ⚠️  Consider using a GPU-enabled machine.")

    print("=" * 60 + "\n")


class ExperimentRunner:
    """Orchestrates a single continual pretraining experiment."""

    def __init__(
        self,
        run_id: str,
        config_path: Path,
        project_root: Path,
        seed: int = 0,
        allow_cpu: bool = False
    ):
        self.run_id = run_id
        self.config_path = config_path
        self.project_root = project_root
        self.seed = seed

        # Load config
        self.config = load_config(config_path)
        self.config['seed'] = seed

        # Paths
        self.output_dir = project_root / "experiments" / "runs" / run_id
        self.data_dir = project_root / "data"
        self.eval_dir = self.data_dir / "eval"

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy frozen config
        shutil.copy(config_path, self.output_dir / "config.yaml")

        # Set seeds for full reproducibility
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Note: torch.use_deterministic_algorithms(True) was removed due to 3-4x slowdown.
        # Seed-based reproducibility (manual seeds + cudnn settings above) is sufficient.

        # Device selection with preflight check
        self.device = choose_device(allow_cpu=allow_cpu)
        log_device_info(self.device)

        # Initialize model and tokenizer
        logger.info("Loading model and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = create_model(
            self.config.get("model", {}).get("name", "gpt2"),
            self.config.get("model", {}).get("gradient_checkpointing", True)
        )

        # Initialize trainer
        self.trainer = CPTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            output_dir=self.output_dir
        )

        # Initialize metrics computer
        self.metrics_computer = MetricsComputer(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

        # Results storage
        self.results = {
            "run": {},
            "inputs": {},
            "metrics": {},
            "resources": {},
            "method_params": {},
            "training_curves": None,
            "anomalies": [],
            "notes": ""
        }

        # Reference distributions for drift
        self.ref_distributions = None

    def load_data(self) -> tuple:
        """Load tokenized training and evaluation data."""
        logger.info("Loading data...")

        # Load manifests
        manifest_a = load_manifest(self.data_dir / "manifests" / "domain_a.json")
        manifest_b = load_manifest(self.data_dir / "manifests" / "domain_b.json")

        # Load tokenized training data
        tokens_a = torch.load(self.project_root / manifest_a["train_path"])
        tokens_b = torch.load(self.project_root / manifest_b["train_path"])

        # Load validation data
        valid_tokens_a = torch.load(self.project_root / manifest_a["valid_path"])
        valid_tokens_b = torch.load(self.project_root / manifest_b["valid_path"])

        # SMOKE MODE: Truncate tokens if configured
        smoke_cfg = self.config.get("smoke_mode", {})
        if smoke_cfg.get("enabled", False):
            max_tokens = smoke_cfg.get("max_tokens_per_domain", 100000)
            logger.info(f"[SMOKE MODE] Truncating to {max_tokens:,} tokens per domain")
            tokens_a = tokens_a[:max_tokens]
            tokens_b = tokens_b[:max_tokens]
            # Also limit validation for speed
            max_valid = min(50000, max_tokens // 2)
            valid_tokens_a = valid_tokens_a[:max_valid]
            valid_tokens_b = valid_tokens_b[:max_valid]

        logger.info(f"Domain A: {len(tokens_a):,} train tokens, {len(valid_tokens_a):,} valid tokens")
        logger.info(f"Domain B: {len(tokens_b):,} train tokens, {len(valid_tokens_b):,} valid tokens")

        return (tokens_a, tokens_b, valid_tokens_a, valid_tokens_b, manifest_a, manifest_b)

    def evaluate_checkpoint(
        self,
        valid_tokens_a: torch.Tensor,
        valid_tokens_b: Optional[torch.Tensor],
        checkpoint_name: str,
        compute_drift: bool = True
    ) -> Dict[str, Any]:
        """Run full evaluation at a checkpoint."""
        logger.info(f"Evaluating checkpoint: {checkpoint_name}")
        results = {}

        # PPL on Domain A
        ppl_a = self.metrics_computer.compute_ppl(
            valid_tokens_a,
            batch_size=self.config.get("evaluation", {}).get("ppl_batch_size", 8)
        )
        results["ppl_a"] = ppl_a["ppl_primary"]
        results["ppl_a_median_batch"] = ppl_a["ppl_median_batch"]

        # PPL on Domain B (if available)
        if valid_tokens_b is not None:
            ppl_b = self.metrics_computer.compute_ppl(
                valid_tokens_b,
                batch_size=self.config.get("evaluation", {}).get("ppl_batch_size", 8)
            )
            results["ppl_b"] = ppl_b["ppl_primary"]
            results["ppl_b_median_batch"] = ppl_b["ppl_median_batch"]

        # Load prompts
        drift_prompts = load_prompts(self.eval_dir / "prompts_drift_v1.json")
        quality_prompts = load_prompts(self.eval_dir / "prompts_quality_v1.json")

        # SMOKE MODE: Limit prompts
        smoke_cfg = self.config.get("smoke_mode", {})
        if smoke_cfg.get("enabled", False):
            max_drift = smoke_cfg.get("max_drift_prompts", 10)
            max_quality = smoke_cfg.get("max_quality_prompts", 20)
            drift_prompts = drift_prompts[:max_drift]
            quality_prompts = quality_prompts[:max_quality]

        # Rep-n metrics
        eval_cfg = self.config.get("evaluation", {}).get("quality", {})
        n_quality = min(200, len(quality_prompts))
        rep_results = self.metrics_computer.compute_repetition(
            quality_prompts[:n_quality],
            max_new_tokens=eval_cfg.get("max_new_tokens", 256),
            temperature=eval_cfg.get("temperature", 0.7),
            top_p=eval_cfg.get("top_p", 0.9),
            do_sample=eval_cfg.get("do_sample", True)
        )
        results["rep4"] = rep_results["rep4"]
        results["rep8"] = rep_results["rep8"]

        # Drift metrics
        if compute_drift:
            drift_cfg = self.config.get("evaluation", {}).get("drift", {})
            drift_results, current_dist = self.metrics_computer.compute_drift_metrics(
                drift_prompts,
                reference_distributions=self.ref_distributions,
                max_new_tokens=drift_cfg.get("max_new_tokens", 128),
                do_sample=drift_cfg.get("do_sample", False)
            )

            if self.ref_distributions is None:
                # First evaluation - save as reference
                self.ref_distributions = current_dist
                results["drift_js"] = 0.0
                results["vocab_overlap"] = 1.0
            else:
                results["drift_js"] = drift_results.get("js_divergence", 0.0)
                results["vocab_overlap"] = drift_results.get("vocab_overlap", 1.0)

        # LAMBADA
        lambada_path = self.eval_dir / "lambada_test.json"
        max_lambada = smoke_cfg.get("max_lambada", 1000) if smoke_cfg.get("enabled", False) else 1000
        if lambada_path.exists():
            lambada_results = self.metrics_computer.compute_lambada_accuracy(
                lambada_path,
                max_examples=max_lambada
            )
            results["lambada_acc"] = lambada_results["accuracy"]

        logger.info(f"Checkpoint {checkpoint_name} - PPL_A: {results.get('ppl_a', 'N/A'):.2f}, "
                    f"Rep-4: {results.get('rep4', 'N/A'):.4f}")

        return results

    def _utc_iso_timestamp(self, dt: datetime = None) -> str:
        """Return ISO8601 timestamp with Z suffix (UTC)."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")

    def _fmt(self, val: Any, fmt: str = ".2f") -> str:
        """Safely format numeric values, returning 'N/A' for None or invalid."""
        if val is None:
            return "N/A"
        try:
            return f"{val:{fmt}}"
        except (ValueError, TypeError):
            return str(val)

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment."""
        start_time = datetime.now(timezone.utc)
        logger.info("=" * 60)
        logger.info(f"Starting experiment: {self.run_id}")
        logger.info("=" * 60)

        # Update registry
        registry_path = self.project_root / "experiments" / "run_registry.csv"
        update_run_registry(registry_path, self.run_id, {
            "status": "running",
            "timestamp_start": self._utc_iso_timestamp(start_time)
        })

        try:
            # Load data
            tokens_a, tokens_b, valid_a, valid_b, manifest_a, manifest_b = self.load_data()

            # Extract drift and quality config
            drift_cfg = self.config.get("evaluation", {}).get("drift", {})
            quality_cfg = self.config.get("evaluation", {}).get("quality", {})

            # Determine drift decoding mode
            drift_do_sample = drift_cfg.get("do_sample", False)
            drift_decoding_mode = drift_cfg.get("decoding_mode",
                                                 "sampling" if drift_do_sample else "deterministic")

            # Determine quality decoding mode
            quality_do_sample = quality_cfg.get("do_sample", True)
            quality_decoding_mode = quality_cfg.get("decoding_mode",
                                                     "sampling" if quality_do_sample else "deterministic")

            # Store inputs with disambiguated drift/quality settings
            self.results["inputs"] = {
                "domain_a_name": manifest_a["name"],
                "domain_b_name": manifest_b["name"],
                "domain_a_token_tier": manifest_a["tier"],
                "domain_b_token_tier": manifest_b["tier"],
                "domain_a_tokens_used": manifest_a["tokens_used"],
                "domain_b_tokens_used": manifest_b["tokens_used"],
                "domain_a_id": manifest_a["hash"],
                "domain_b_id": manifest_b["hash"],
                "prompts_drift_version": "prompts_drift_v1.json",
                "prompts_quality_version": "prompts_quality_v1.json",
                "prompts_toxicity_version": None,
                # Drift generation settings
                "gen_drift_max_new_tokens": drift_cfg.get("max_new_tokens", 128),
                "gen_drift_decoding_mode": drift_decoding_mode,
                "gen_drift_do_sample": drift_do_sample,
                # Quality generation settings
                "gen_quality_max_new_tokens": quality_cfg.get("max_new_tokens", 256),
                "gen_quality_decoding_mode": quality_decoding_mode,
                "gen_quality_do_sample": quality_do_sample,
                "gen_quality_temperature": quality_cfg.get("temperature", 0.7),
                "gen_quality_top_p": quality_cfg.get("top_p", 0.9),
            }

            # Add drift sampling params only if drift uses sampling
            if drift_do_sample:
                self.results["inputs"]["gen_drift_temperature"] = drift_cfg.get("temperature", 0.7)
                self.results["inputs"]["gen_drift_top_p"] = drift_cfg.get("top_p", 0.9)

            # ===== INIT EVALUATION =====
            logger.info("\n[1/5] Initial evaluation (pre-training)...")
            init_results = self.evaluate_checkpoint(valid_a, valid_b, "init")
            self.results["metrics"]["ppl_a_init"] = init_results["ppl_a"]
            self.results["metrics"]["ppl_b_init"] = init_results.get("ppl_b")
            lambada_init = init_results.get("lambada_acc")

            # ===== DOMAIN A TRAINING =====
            logger.info("\n[2/5] Training on Domain A...")
            domain_a_start = datetime.now()
            domain_a_stats = self.trainer.train_domain(
                tokens_a,
                self.config.get("training", {}).get("domain_a", {}),
                "domain_a",
                log_steps=self.config.get("logging", {}).get("log_steps", 100)
            )
            domain_a_hours = (datetime.now() - domain_a_start).total_seconds() / 3600

            # Save Domain A checkpoint
            self.trainer.save_checkpoint("theta_A", {"domain": "A"})

            # ===== SETUP CL METHODS (after Domain A) =====
            method = self.config.get("method", "baseline")
            if method != "baseline":
                logger.info(f"\n[2.5/5] Setting up CL method: {method}...")
                self.trainer.setup_cl_after_domain_a(tokens_a)

            # ===== POST-A EVALUATION =====
            logger.info("\n[3/5] Evaluation after Domain A...")
            post_a_results = self.evaluate_checkpoint(valid_a, None, "post_A")
            self.results["metrics"]["ppl_a_before"] = post_a_results["ppl_a"]
            self.results["metrics"]["ppl_a_before_median_batch"] = post_a_results["ppl_a_median_batch"]
            self.results["metrics"]["rep4_before"] = post_a_results["rep4"]
            self.results["metrics"]["rep8_before"] = post_a_results["rep8"]
            lambada_before = post_a_results.get("lambada_acc", lambada_init)
            self.results["metrics"]["lambada_before"] = lambada_before

            # Store reference for drift
            drift_before = post_a_results.get("drift_js", 0.0)
            vocab_before = post_a_results.get("vocab_overlap", 1.0)

            # ===== DOMAIN B TRAINING =====
            logger.info("\n[4/5] Training on Domain B...")
            domain_b_start = datetime.now()
            # Use train_domain_b which applies CL methods (EWC/Replay/MER) if configured
            domain_b_stats = self.trainer.train_domain_b(
                tokens_b,
                self.config.get("training", {}).get("domain_b", {}),
                log_steps=self.config.get("logging", {}).get("log_steps", 100)
            )
            domain_b_hours = (datetime.now() - domain_b_start).total_seconds() / 3600

            # Save final checkpoint
            self.trainer.save_checkpoint("theta_AB", {"domain": "AB"})

            # ===== FINAL EVALUATION =====
            logger.info("\n[5/5] Final evaluation (post-CPT)...")
            final_results = self.evaluate_checkpoint(valid_a, valid_b, "final")

            # Store final metrics
            self.results["metrics"]["ppl_a_after"] = final_results["ppl_a"]
            self.results["metrics"]["ppl_a_after_median_batch"] = final_results["ppl_a_median_batch"]
            self.results["metrics"]["ppl_b_after"] = final_results.get("ppl_b")
            self.results["metrics"]["ppl_b_after_median_batch"] = final_results.get("ppl_b_median_batch")
            self.results["metrics"]["rep4_after"] = final_results["rep4"]
            self.results["metrics"]["rep8_after"] = final_results["rep8"]
            self.results["metrics"]["lambada_after"] = final_results.get("lambada_acc")

            # Drift metrics
            self.results["metrics"]["drift_metric_name"] = "js_divergence"
            self.results["metrics"]["drift_value_before"] = drift_before
            self.results["metrics"]["drift_value_after"] = final_results.get("drift_js", 0.0)
            self.results["metrics"]["vocab_overlap_before"] = vocab_before
            self.results["metrics"]["vocab_overlap_after"] = final_results.get("vocab_overlap", 1.0)

            # Compute forgetting
            ppl_before = self.results["metrics"]["ppl_a_before"]
            ppl_after = self.results["metrics"]["ppl_a_after"]
            forgetting = MetricsComputer.compute_forgetting_pct(ppl_before, ppl_after)
            self.results["metrics"]["forgetting_pct"] = forgetting

            # ===== FINALIZE =====
            end_time = datetime.now(timezone.utc)
            total_hours = (end_time - start_time).total_seconds() / 3600

            # Resource metrics
            resource_stats = self.trainer.get_resource_stats()
            self.results["resources"] = {
                "total_hours": total_hours,
                "domain_a_hours": domain_a_hours,
                "domain_b_hours": domain_b_hours,
                "peak_vram_gb": resource_stats["peak_vram_gb"],
                "peak_ram_gb": resource_stats["peak_ram_gb"],
                "avg_tokens_per_sec": (domain_a_stats["tokens_per_sec"] + domain_b_stats["tokens_per_sec"]) / 2
            }

            # Run metadata
            self.results["run"] = {
                "run_id": self.run_id,
                "research_question": "RQ0" if "pilot" in self.run_id else "RQ1",
                "method": self.config.get("method", "baseline"),
                "seed": self.seed,
                "timestamp_start": self._utc_iso_timestamp(start_time),
                "timestamp_end": self._utc_iso_timestamp(end_time),
                "status": "completed",
                "config_file": str(self.config_path.relative_to(self.project_root)),
                "runpack_file": f"runpack_{self.run_id}.md"
            }

            # Method params (empty for baseline)
            self.results["method_params"] = self.config.get("method_params", {})

            # Check for anomalies (None-safe)
            rep4_after = self.results["metrics"].get("rep4_after")
            if rep4_after is not None and rep4_after > 0.25:
                self.results["anomalies"].append("high_rep4")
            peak_vram = resource_stats.get("peak_vram_gb")
            if peak_vram is not None and peak_vram > 7.5:
                self.results["anomalies"].append("oom_recovery")

            self.results["notes"] = "Pilot baseline run completed successfully."

            # Save outputs
            self._save_outputs()

            # Update registry
            update_run_registry(registry_path, self.run_id, {
                "status": "completed",
                "timestamp_end": self._utc_iso_timestamp(end_time),
                "metrics_file": f"experiments/runs/{self.run_id}/metrics.json"
            })

            logger.info("\n" + "=" * 60)
            logger.info(f"Experiment {self.run_id} completed!")
            logger.info(f"Forgetting: {forgetting:.2f}%")
            logger.info(f"Total time: {total_hours:.2f} hours")
            logger.info("=" * 60)

            return self.results

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            update_run_registry(registry_path, self.run_id, {
                "status": "failed",
                "timestamp_end": self._utc_iso_timestamp(),
                "notes": str(e)
            })
            raise

    def _save_outputs(self):
        """Save all output files."""
        # metrics.json
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Metrics saved: {metrics_path}")

        # Training log
        self.trainer.save_training_log(self.output_dir / "training_log.jsonl")

        # Runpack
        self._generate_runpack()

    def _generate_runpack(self):
        """Generate runpack markdown summary."""
        r = self.results
        m = r["metrics"]
        inp = r["inputs"]
        res = r["resources"]

        runpack = f"""# Runpack: {self.run_id}

**Version**: 1.0
**Date**: {datetime.now().strftime("%Y-%m-%d")}

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | {self.run_id} |
| **Method** | {r["run"]["method"]} |
| **Seed** | {r["run"]["seed"]} |
| **Research Question** | {r["run"]["research_question"]} |
| **Status** | {r["run"]["status"]} |

---

## Dataset Configuration

| Field | Value |
|-------|-------|
| **Domain A** | {inp["domain_a_name"]} |
| **Domain A Token Tier** | {inp["domain_a_token_tier"]} |
| **Domain A Tokens Used** | {inp["domain_a_tokens_used"]:,} |
| **Domain A Hash/ID** | {inp["domain_a_id"]} |
| **Domain B** | {inp["domain_b_name"]} |
| **Domain B Token Tier** | {inp["domain_b_token_tier"]} |
| **Domain B Tokens Used** | {inp["domain_b_tokens_used"]:,} |
| **Domain B Hash/ID** | {inp["domain_b_id"]} |

---

## Prompt Set Versions

| Set | Version |
|-----|---------|
| Drift Prompts | {inp["prompts_drift_version"]} |
| Quality Prompts | {inp["prompts_quality_version"]} |
| Toxicity Prompts | {inp["prompts_toxicity_version"] or "null"} |

---

## Primary Metrics

### Perplexity

| Checkpoint | PPL_A | PPL_B |
|------------|------:|------:|
| Init (pre-training) | {self._fmt(m.get("ppl_a_init"))} | {self._fmt(m.get("ppl_b_init"))} |
| After Domain A | {self._fmt(m.get("ppl_a_before"))} | — |
| After Domain B (final) | {self._fmt(m.get("ppl_a_after"))} | {self._fmt(m.get("ppl_b_after"))} |

### Forgetting

| Metric | Value |
|--------|------:|
| **Forget%** | {self._fmt(m.get("forgetting_pct"))}% |

---

## Generation Quality

| Metric | Before (after A) | After (final) |
|--------|----------------:|--------------:|
| **Rep-4** | {self._fmt(m.get("rep4_before"), ".4f")} | {self._fmt(m.get("rep4_after"), ".4f")} |
| Rep-8 | {self._fmt(m.get("rep8_before"), ".4f")} | {self._fmt(m.get("rep8_after"), ".4f")} |

---

## Drift Metrics

| Metric | Before | After |
|--------|-------:|------:|
| **Drift (JS divergence)** | {self._fmt(m.get("drift_value_before", 0), ".4f")} | {self._fmt(m.get("drift_value_after", 0), ".4f")} |
| Vocab Overlap | {self._fmt(m.get("vocab_overlap_before", 1), ".4f")} | {self._fmt(m.get("vocab_overlap_after", 1), ".4f")} |

---

## General Ability

| Checkpoint | LAMBADA Accuracy |
|------------|----------------:|
| Before CPT | {self._fmt(m.get("lambada_before"), ".4f")} |
| After CPT | {self._fmt(m.get("lambada_after"), ".4f")} |

---

## Resource Metrics

| Metric | Value |
|--------|------:|
| **Total Wall Time** | {self._fmt(res.get("total_hours", 0))} hours |
| Domain A Time | {self._fmt(res.get("domain_a_hours", 0))} hours |
| Domain B Time | {self._fmt(res.get("domain_b_hours", 0))} hours |
| **Peak VRAM** | {self._fmt(res.get("peak_vram_gb", 0))} GB |
| **Peak RAM** | {self._fmt(res.get("peak_ram_gb", 0))} GB |
| Avg Tokens/sec | {self._fmt(res.get("avg_tokens_per_sec", 0), ".0f")} |

---

## Method Parameters

{json.dumps(r.get("method_params", {}), indent=2) if r.get("method_params") else "None (baseline method)"}

---

## Anomalies

{chr(10).join(f"- {a}" for a in r.get("anomalies", [])) or "- None"}

---

## Notes

{r.get("notes", "No notes.")}

---

## Artifacts

| Artifact | Path |
|----------|------|
| Config | {r["run"]["config_file"]} |
| Metrics JSON | experiments/runs/{self.run_id}/metrics.json |
| Domain A Checkpoint | experiments/runs/{self.run_id}/checkpoints/theta_A.pt |
| Final Checkpoint | experiments/runs/{self.run_id}/checkpoints/theta_AB.pt |

---

**Generated**: {datetime.now().isoformat()}
"""

        runpack_path = self.output_dir / f"runpack_{self.run_id}.md"
        with open(runpack_path, 'w') as f:
            f.write(runpack)
        logger.info(f"Runpack saved: {runpack_path}")


def main():
    parser = argparse.ArgumentParser(description="Run continual pretraining experiment")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--project-root", type=str, default=None, help="Project root directory")
    parser.add_argument("--allow-cpu", action="store_true", default=False,
                        help="Allow CPU-only mode (not recommended, very slow)")
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else get_project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    # Determine seed
    seed = args.seed
    if seed is None:
        # Extract from run_id if present (e.g., pilot_baseline_s0 -> 0)
        if "_s" in args.run_id:
            try:
                seed = int(args.run_id.split("_s")[-1])
            except ValueError:
                seed = 0
        else:
            seed = 0

    runner = ExperimentRunner(
        run_id=args.run_id,
        config_path=config_path,
        project_root=project_root,
        seed=seed,
        allow_cpu=args.allow_cpu
    )

    runner.run()


if __name__ == "__main__":
    main()
