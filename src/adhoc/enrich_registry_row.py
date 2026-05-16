#!/usr/bin/env python3
"""
Enrich a run_registry.csv row from the run's metrics.json.

The runner (`src/run_experiment.py`) only writes `status`, `timestamp_start`,
`timestamp_end`, `metrics_file` into the registry. Other columns
(`research_question`, `method`, `seed`, `model_id`, `model_family`,
`model_params_m`, `config_file`, `promptset_version`, `dataset_tier`,
`notes`) historically got filled in by hand or by a separate flow. This
script does it deterministically from the run's metrics.json so cloud-run
rows match the metadata richness of the original MPS rows.

Usage:
    python src/adhoc/enrich_registry_row.py <run_id>
    python src/adhoc/enrich_registry_row.py --all   # process every row with empty research_question

Idempotent: only fills cells that are currently empty. Never overwrites
existing non-empty values.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def _format_tier(metrics: dict) -> str:
    inputs = metrics.get("inputs", {})
    a_tier = inputs.get("domain_a_token_tier", "")
    a_tokens = inputs.get("domain_a_tokens_used", 0)
    b_tier = inputs.get("domain_b_token_tier", "")
    b_tokens = inputs.get("domain_b_tokens_used", 0)
    a_m = f"{a_tokens // 1_000_000}M" if a_tokens else "?"
    b_m = f"{b_tokens // 1_000_000}M" if b_tokens else "?"
    return f"A={a_tier}({a_m});B={b_tier}({b_m})"


def _format_promptset(metrics: dict) -> str:
    inputs = metrics.get("inputs", {})
    drift = inputs.get("prompts_drift_version", "")
    quality = inputs.get("prompts_quality_version", "")
    parts = []
    for label, val in (("drift", drift), ("quality", quality)):
        if val:
            ver = val.split("_v")[-1].replace(".json", "") if "_v" in val else val
            parts.append(f"{label}=v{ver}" if ver and not ver.startswith("v") else f"{label}={ver}")
    return ";".join(parts)


def derive_fields(run_id: str, metrics: dict) -> dict:
    run = metrics.get("run", {})
    fields = {
        "research_question": run.get("research_question", ""),
        "method": run.get("method", ""),
        "seed": str(run.get("seed", "")) if run.get("seed") is not None else "",
        "model_id": run.get("model_id", ""),
        "model_family": run.get("model_family", ""),
        "model_params_m": str(run.get("model_params_m", "")) if run.get("model_params_m") is not None else "",
        "config_file": run.get("config_file", ""),
        "promptset_version": _format_promptset(metrics),
        "dataset_tier": _format_tier(metrics),
    }
    notes = metrics.get("notes", "")
    if notes:
        fields["notes"] = notes
    return fields


def enrich(registry_path: Path, run_id: str, runs_dir: Path) -> bool:
    """Returns True if anything was changed."""
    metrics_path = runs_dir / run_id / "metrics.json"
    if not metrics_path.exists():
        print(f"[skip {run_id}] no metrics.json")
        return False

    with open(metrics_path) as f:
        metrics = json.load(f)
    new_fields = derive_fields(run_id, metrics)

    rows = []
    headers = None
    changed = False
    with open(registry_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            if row["run_id"] == run_id:
                for col, val in new_fields.items():
                    if col not in row:
                        continue
                    if not row[col] and val:
                        row[col] = val
                        changed = True
            rows.append(row)

    if changed:
        with open(registry_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        filled = [c for c, v in new_fields.items() if v]
        print(f"[enriched {run_id}] filled: {', '.join(filled)}")
    else:
        print(f"[ok {run_id}] already complete")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", help="Run ID to enrich; omit with --all")
    ap.add_argument("--all", action="store_true",
                    help="Enrich every row with empty research_question")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    registry_path = project_root / "experiments" / "run_registry.csv"
    runs_dir = project_root / "experiments" / "runs"

    if args.all:
        with open(registry_path, newline="") as f:
            rows = list(csv.DictReader(f))
        any_changed = False
        for row in rows:
            if row.get("research_question", "").strip():
                continue
            if (runs_dir / row["run_id"] / "metrics.json").exists():
                if enrich(registry_path, row["run_id"], runs_dir):
                    any_changed = True
        if not any_changed:
            print("All rows already enriched.")
        sys.exit(0)

    if not args.run_id:
        ap.error("Provide a run_id or --all")
    enrich(registry_path, args.run_id, runs_dir)


if __name__ == "__main__":
    main()
