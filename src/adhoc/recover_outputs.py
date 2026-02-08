#!/usr/bin/env python3
"""
Recover outputs (metrics.json + runpack) for a run that crashed AFTER training finished.

This script:
- Loads the frozen config from experiments/runs/<run_id>/config.yaml
- Loads checkpoints:
    - experiments/runs/<run_id>/checkpoints/theta_A.pt
    - experiments/runs/<run_id>/checkpoints/theta_AB.pt
- Re-runs evaluations (init -> post_A -> final)
- Writes:
    - experiments/runs/<run_id>/metrics.json
    - experiments/runs/<run_id>/runpack_<run_id>.md
- Updates experiments/run_registry.csv best-effort

Run (from repo root):
    python src/Adhoc/recover_outputs.py --run-id pilot_baseline_s0_leanfix --seed 0

Notes:
- No retraining is performed.
- This assumes your checkpoints contain either:
    - {"model_state_dict": ...} or
    - directly a model state_dict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

import torch


def project_root_from_this_file() -> Path:
    # repo_root/src/Adhoc/recover_outputs.py -> repo_root
    return Path(__file__).resolve().parents[2]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_from_ckpt(runner, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location=runner.device)
    state = ckpt.get("model_state_dict", ckpt)  # support either format
    missing, unexpected = runner.model.load_state_dict(state, strict=False)

    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover metrics.json and runpack for a completed run.")
    parser.add_argument("--run-id", required=True, help="Run ID folder under experiments/runs/")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for runner construction (default: 0)")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        default=False,
        help="Allow CPU device if no CUDA/MPS. (If omitted and only CPU is available, we still proceed, but warn.)",
    )
    args = parser.parse_args()

    project_root = project_root_from_this_file()
    run_id = args.run_id

    run_dir = project_root / "experiments" / "runs" / run_id
    ckpt_dir = run_dir / "checkpoints"
    config_path = run_dir / "config.yaml"

    theta_A = ckpt_dir / "theta_A.pt"
    theta_AB = ckpt_dir / "theta_AB.pt"

    # preflight
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing frozen config.yaml: {config_path}")
    if not theta_A.exists():
        raise FileNotFoundError(f"Missing checkpoint: {theta_A}")
    if not theta_AB.exists():
        raise FileNotFoundError(f"Missing checkpoint: {theta_AB}")

    sys.path.insert(0, str(project_root / "src"))
    # Import AFTER sys.path injection
    from run_experiment import ExperimentRunner, update_run_registry  # type: ignore

    device = pick_device()
    if device == "cpu" and not args.allow_cpu:
        print("[warn] Only CPU detected. This recovery run will still proceed, but evaluation may be slow.")
        print("       If you want to silence this warning, re-run with --allow-cpu")

    print(f"[info] project_root: {project_root}")
    print(f"[info] run_id:       {run_id}")
    print(f"[info] device:       {device}")
    print(f"[info] run_dir:      {run_dir}")

    # Construct runner (no training occurs unless you call trainer.train_domain)
    runner = ExperimentRunner(
        run_id=run_id,
        config_path=config_path,
        project_root=project_root,
        seed=args.seed,
        allow_cpu=(device == "cpu" and args.allow_cpu) or (device != "cpu"),
    )

    start_time = datetime.now(timezone.utc)

    # Load data
    tokens_a, tokens_b, valid_a, valid_b, manifest_a, manifest_b = runner.load_data()

    # ---- init eval (sets drift reference distribution inside runner) ----
    print("\n[recovery] (1/3) eval init (sets drift reference)")
    init_results = runner.evaluate_checkpoint(valid_a, valid_b, "init")
    runner.results["metrics"]["ppl_a_init"] = init_results["ppl_a"]
    runner.results["metrics"]["ppl_b_init"] = init_results.get("ppl_b")
    runner.results["metrics"]["lambada_init"] = init_results.get("lambada_acc")

    # ---- load theta_A and eval post_A ----
    print("\n[recovery] (2/3) load theta_A -> eval post_A")
    load_model_from_ckpt(runner, theta_A)
    post_a_results = runner.evaluate_checkpoint(valid_a, None, "post_A")
    runner.results["metrics"]["ppl_a_before"] = post_a_results["ppl_a"]
    runner.results["metrics"]["ppl_a_before_median_batch"] = post_a_results["ppl_a_median_batch"]
    runner.results["metrics"]["rep4_before"] = post_a_results["rep4"]
    runner.results["metrics"]["rep8_before"] = post_a_results["rep8"]
    runner.results["metrics"]["lambada_before"] = post_a_results.get("lambada_acc")

    drift_before = post_a_results.get("drift_js", 0.0)
    vocab_before = post_a_results.get("vocab_overlap", 1.0)

    # ---- load theta_AB and eval final ----
    print("\n[recovery] (3/3) load theta_AB -> eval final")
    load_model_from_ckpt(runner, theta_AB)
    final_results = runner.evaluate_checkpoint(valid_a, valid_b, "final")
    runner.results["metrics"]["ppl_a_after"] = final_results["ppl_a"]
    runner.results["metrics"]["ppl_a_after_median_batch"] = final_results["ppl_a_median_batch"]
    runner.results["metrics"]["ppl_b_after"] = final_results.get("ppl_b")
    runner.results["metrics"]["ppl_b_after_median_batch"] = final_results.get("ppl_b_median_batch")
    runner.results["metrics"]["rep4_after"] = final_results["rep4"]
    runner.results["metrics"]["rep8_after"] = final_results["rep8"]
    runner.results["metrics"]["lambada_after"] = final_results.get("lambada_acc")

    # drift after
    runner.results["metrics"]["drift_metric_name"] = "js_divergence"
    runner.results["metrics"]["drift_value_before"] = drift_before
    runner.results["metrics"]["drift_value_after"] = final_results.get("drift_js", 0.0)
    runner.results["metrics"]["vocab_overlap_before"] = vocab_before
    runner.results["metrics"]["vocab_overlap_after"] = final_results.get("vocab_overlap", 1.0)

    # forgetting
    ppl_before = runner.results["metrics"]["ppl_a_before"]
    ppl_after = runner.results["metrics"]["ppl_a_after"]
    runner.results["metrics"]["forgetting_pct"] = runner.metrics_computer.compute_forgetting_pct(ppl_before, ppl_after)

    # resources (training already happened earlier; this pass is eval-only)
    resource_stats = runner.trainer.get_resource_stats()
    runner.results["resources"] = {
        "total_hours": None,
        "domain_a_hours": None,
        "domain_b_hours": None,
        "peak_vram_gb": resource_stats.get("peak_vram_gb"),
        "peak_ram_gb": resource_stats.get("peak_ram_gb"),
        "avg_tokens_per_sec": None,
    }

    # inputs (for runpack completeness)
    runner.results["inputs"] = {
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
    }

    # anomalies
    runner.results["anomalies"] = []
    if runner.results["metrics"]["rep4_after"] > 0.25:
        runner.results["anomalies"].append("high_rep4")
    peak_vram = resource_stats.get("peak_vram_gb")
    if peak_vram is not None and peak_vram > 7.5:
        runner.results["anomalies"].append("oom_recovery")

    # run metadata
    end_time = datetime.now(timezone.utc)
    runner.results["run"] = {
        "run_id": run_id,
        "research_question": "RQ0" if "pilot" in run_id else "RQ1",
        "method": runner.config.get("method", "baseline"),
        "seed": runner.seed,
        "timestamp_start": start_time.isoformat().replace("+00:00", "Z"),
        "timestamp_end": end_time.isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "config_file": str(config_path.relative_to(project_root)),
        "runpack_file": f"runpack_{run_id}.md",
    }
    runner.results["method_params"] = runner.config.get("method_params", {})
    runner.results["notes"] = "Recovered outputs after late crash (no retraining)."

    # write outputs (metrics.json + runpack)
    runner._save_outputs()

    # best-effort registry update
    registry_path = project_root / "experiments" / "run_registry.csv"
    if registry_path.exists():
        try:
            update_run_registry(
                registry_path,
                run_id,
                {
                    "status": "completed",
                    "timestamp_end": runner.results["run"]["timestamp_end"],
                    "metrics_file": f"experiments/runs/{run_id}/metrics.json",
                    "notes": "Recovered outputs after late crash",
                },
            )
        except Exception as e:
            print(f"[warn] could not update registry: {e}")

    print("\n[done] wrote:")
    print(" -", run_dir / "metrics.json")
    print(" -", run_dir / f"runpack_{run_id}.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
