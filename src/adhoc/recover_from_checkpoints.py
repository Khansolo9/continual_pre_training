#!/usr/bin/env python3
"""
Recovery Tool: Eval-only mode from existing checkpoints.

This is a standalone recovery tool that loads theta_A.pt and theta_AB.pt
from an existing run and regenerates metrics.json and runpack without training.

IMPORTANT: This tool sets drift baseline at theta_A (post-A), not init.
Use this only for recovering partial runs, not for primary research runs.

Usage:
    python src/adhoc/recover_from_checkpoints.py --run-id pilot_baseline_s0_leanfix
"""

import os
import sys
import json
import yaml
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import torch
from transformers import GPT2Tokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer import CPTTrainer, create_model
from metrics import MetricsComputer, load_prompts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load dataset manifest."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def choose_device() -> str:
    """Choose compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def recover_run(run_id: str, project_root: Path, seed: int = 0):
    """
    Recover metrics.json and runpack from existing checkpoints.

    WARNING: This sets drift baseline at theta_A, not init.
    """
    output_dir = project_root / "experiments" / "runs" / run_id
    data_dir = project_root / "data"
    eval_dir = data_dir / "eval"

    # Load config from the run
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(config_path)

    # Check for checkpoints
    theta_a_path = output_dir / "checkpoints" / "theta_A.pt"
    theta_ab_path = output_dir / "checkpoints" / "theta_AB.pt"

    if not theta_a_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {theta_a_path}")
    if not theta_ab_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {theta_ab_path}")

    logger.info(f"Recovering run: {run_id}")
    logger.info(f"Found theta_A: {theta_a_path}")
    logger.info(f"Found theta_AB: {theta_ab_path}")

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = choose_device()
    logger.info(f"Using device: {device}")

    # Initialize model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = create_model(
        config.get("model", {}).get("name", "gpt2"),
        config.get("model", {}).get("gradient_checkpointing", True)
    ).to(device)

    # Initialize metrics computer
    metrics_computer = MetricsComputer(model=model, tokenizer=tokenizer, device=device)

    # Load manifests
    manifest_a = load_manifest(data_dir / "manifests" / "domain_a.json")
    manifest_b = load_manifest(data_dir / "manifests" / "domain_b.json")

    # Load validation data
    valid_a = torch.load(project_root / manifest_a["valid_path"])
    valid_b = torch.load(project_root / manifest_b["valid_path"])

    # Load prompts
    drift_prompts = load_prompts(eval_dir / "prompts_drift_v1.json")
    quality_prompts = load_prompts(eval_dir / "prompts_quality_v1.json")

    # Config extraction
    drift_cfg = config.get("evaluation", {}).get("drift", {})
    quality_cfg = config.get("evaluation", {}).get("quality", {})

    results = {
        "run": {},
        "inputs": {},
        "metrics": {},
        "resources": {},
        "method_params": {},
        "training_curves": None,
        "anomalies": [],
        "notes": ""
    }

    start_time = datetime.now(timezone.utc)
    ref_distributions = None

    # ===== LOAD THETA_A AND EVALUATE =====
    logger.info("\n[1/2] Loading theta_A and running post-A evaluation...")
    checkpoint_a = torch.load(theta_a_path, map_location=device)
    model.load_state_dict(checkpoint_a["model_state_dict"])

    # PPL on Domain A
    ppl_a = metrics_computer.compute_ppl(valid_a, batch_size=8)
    results["metrics"]["ppl_a_before"] = ppl_a["ppl_primary"]
    results["metrics"]["ppl_a_before_median_batch"] = ppl_a["ppl_median_batch"]

    # Rep-n (deterministic)
    rep_results = metrics_computer.compute_repetition(
        quality_prompts[:200],
        max_new_tokens=quality_cfg.get("max_new_tokens", 256),
        do_sample=False  # Deterministic
    )
    results["metrics"]["rep4_before"] = rep_results["rep4"]
    results["metrics"]["rep8_before"] = rep_results["rep8"]

    # Drift (sets baseline)
    drift_results, ref_distributions = metrics_computer.compute_drift_metrics(
        drift_prompts,
        reference_distributions=None,
        max_new_tokens=drift_cfg.get("max_new_tokens", 128),
        do_sample=False
    )
    # First eval: baseline
    drift_before = 0.0
    vocab_before = 1.0

    # LAMBADA
    lambada_path = eval_dir / "lambada_test.json"
    if lambada_path.exists():
        lambada_results = metrics_computer.compute_lambada_accuracy(lambada_path, max_examples=1000)
        results["metrics"]["lambada_before"] = lambada_results["accuracy"]

    # ===== LOAD THETA_AB AND EVALUATE =====
    logger.info("\n[2/2] Loading theta_AB and running final evaluation...")
    checkpoint_ab = torch.load(theta_ab_path, map_location=device)
    model.load_state_dict(checkpoint_ab["model_state_dict"])

    # PPL on both domains
    ppl_a_final = metrics_computer.compute_ppl(valid_a, batch_size=8)
    ppl_b_final = metrics_computer.compute_ppl(valid_b, batch_size=8)
    results["metrics"]["ppl_a_after"] = ppl_a_final["ppl_primary"]
    results["metrics"]["ppl_a_after_median_batch"] = ppl_a_final["ppl_median_batch"]
    results["metrics"]["ppl_b_after"] = ppl_b_final["ppl_primary"]
    results["metrics"]["ppl_b_after_median_batch"] = ppl_b_final["ppl_median_batch"]

    # Rep-n
    rep_results_final = metrics_computer.compute_repetition(
        quality_prompts[:200],
        max_new_tokens=quality_cfg.get("max_new_tokens", 256),
        do_sample=False
    )
    results["metrics"]["rep4_after"] = rep_results_final["rep4"]
    results["metrics"]["rep8_after"] = rep_results_final["rep8"]

    # Drift (vs theta_A baseline)
    drift_final, _ = metrics_computer.compute_drift_metrics(
        drift_prompts,
        reference_distributions=ref_distributions,
        max_new_tokens=drift_cfg.get("max_new_tokens", 128),
        do_sample=False
    )
    results["metrics"]["drift_metric_name"] = "js_divergence"
    results["metrics"]["drift_value_before"] = drift_before
    results["metrics"]["drift_value_after"] = drift_final.get("js_divergence", 0.0)
    results["metrics"]["vocab_overlap_before"] = vocab_before
    results["metrics"]["vocab_overlap_after"] = drift_final.get("vocab_overlap", 1.0)

    # LAMBADA
    if lambada_path.exists():
        lambada_final = metrics_computer.compute_lambada_accuracy(lambada_path, max_examples=1000)
        results["metrics"]["lambada_after"] = lambada_final["accuracy"]

    # Forgetting
    ppl_before = results["metrics"]["ppl_a_before"]
    ppl_after = results["metrics"]["ppl_a_after"]
    forgetting = ((ppl_after - ppl_before) / ppl_before) * 100 if ppl_before > 0 else float('nan')
    results["metrics"]["forgetting_pct"] = forgetting

    # Finalize
    end_time = datetime.now(timezone.utc)
    total_hours = (end_time - start_time).total_seconds() / 3600

    results["run"] = {
        "run_id": run_id,
        "research_question": "RQ0" if "pilot" in run_id else "RQ1",
        "method": config.get("method", "baseline"),
        "seed": seed,
        "timestamp_start": start_time.isoformat().replace("+00:00", "Z"),
        "timestamp_end": end_time.isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "config_file": f"configs/methods/{config.get('method', 'baseline')}.yaml",
        "runpack_file": f"runpack_{run_id}.md"
    }

    results["inputs"] = {
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
        "gen_drift_max_new_tokens": drift_cfg.get("max_new_tokens", 128),
        "gen_drift_decoding_mode": "deterministic",
        "gen_drift_do_sample": False,
        "gen_quality_max_new_tokens": quality_cfg.get("max_new_tokens", 256),
        "gen_quality_decoding_mode": "deterministic",
        "gen_quality_do_sample": False,
    }

    results["resources"] = {
        "total_hours": total_hours,
        "domain_a_hours": None,
        "domain_b_hours": None,
        "peak_vram_gb": None,
        "peak_ram_gb": None,
        "avg_tokens_per_sec": None
    }

    results["method_params"] = config.get("method_params", {})

    rep4_after = results["metrics"].get("rep4_after")
    if rep4_after is not None and rep4_after > 0.25:
        results["anomalies"].append("high_rep4")

    results["notes"] = "Recovered from checkpoints (eval-only recovery tool). Drift baseline set at theta_A, not init."

    # Save outputs
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Metrics saved: {metrics_path}")

    # Generate minimal runpack
    runpack_content = f"""# Runpack: {run_id} (Recovered)

**Version**: 1.0
**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Note**: This run was recovered from checkpoints using eval-only mode.

## Primary Metrics

| Metric | Before (theta_A) | After (theta_AB) |
|--------|----------------:|----------------:|
| PPL_A | {ppl_before:.2f} | {ppl_after:.2f} |
| PPL_B | N/A | {results["metrics"].get("ppl_b_after", "N/A")} |
| Rep-4 | {results["metrics"]["rep4_before"]:.4f} | {results["metrics"]["rep4_after"]:.4f} |
| Rep-8 | {results["metrics"]["rep8_before"]:.4f} | {results["metrics"]["rep8_after"]:.4f} |
| Drift (JS) | {drift_before:.4f} | {results["metrics"]["drift_value_after"]:.4f} |
| LAMBADA | {results["metrics"].get("lambada_before", "N/A")} | {results["metrics"].get("lambada_after", "N/A")} |

## Forgetting

**Forget%**: {forgetting:.2f}%

---

**Generated**: {datetime.now().isoformat()}
"""

    runpack_path = output_dir / f"runpack_{run_id}.md"
    with open(runpack_path, 'w') as f:
        f.write(runpack_content)
    logger.info(f"Runpack saved: {runpack_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Recovery complete for {run_id}")
    logger.info(f"Forgetting: {forgetting:.2f}%")
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Recover metrics from existing checkpoints")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier")
    parser.add_argument("--project-root", type=str, default=None, help="Project root directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else get_project_root()
    recover_run(args.run_id, project_root, args.seed)


if __name__ == "__main__":
    main()
