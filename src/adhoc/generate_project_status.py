#!/usr/bin/env python3
"""
Project Status Inventory Generator

Reads experiments/run_registry.csv, inspects run directories for artifacts,
and updates the inventory section of docs/PROJECT_STATUS.md.

Usage:
    python src/adhoc/generate_project_status.py --write      # Update the file
    python src/adhoc/generate_project_status.py --dry-run    # Preview changes

The script only modifies content between markers:
    <!-- INVENTORY_START -->
    <!-- INVENTORY_END -->

All other content in PROJECT_STATUS.md is preserved.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def read_registry(registry_path: Path) -> List[Dict[str, str]]:
    """Read the run registry CSV file."""
    runs = []
    with open(registry_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row)
    return runs


def check_artifact(run_dir: Path, filename: str) -> str:
    """Check if an artifact exists. Returns 'YES', 'NO', or '-' for planned runs."""
    path = run_dir / filename
    if path.exists():
        return "YES"
    return "NO"


def check_checkpoint(run_dir: Path, checkpoint_name: str) -> str:
    """Check if a checkpoint exists."""
    path = run_dir / "checkpoints" / checkpoint_name
    if path.exists():
        return "YES"
    return "NO"


def format_timestamp(ts: Optional[str]) -> str:
    """Format timestamp for display, or return '-' if empty."""
    if not ts or ts.strip() == '':
        return "—"
    # Truncate to seconds for display
    if 'T' in ts:
        return ts.split('.')[0] + 'Z'
    return ts


def inspect_run(run_id: str, run_info: Dict[str, str], runs_dir: Path) -> Dict[str, Any]:
    """Inspect a run directory and gather artifact status."""
    run_dir = runs_dir / run_id
    status = run_info.get('status', 'unknown')

    result = {
        'run_id': run_id,
        'status': status,
        'method': run_info.get('method', '—'),
        'seed': run_info.get('seed', '—'),
        'start': format_timestamp(run_info.get('timestamp_start', '')),
        'end': format_timestamp(run_info.get('timestamp_end', '')),
        'notes': run_info.get('notes', ''),
    }

    # For planned runs, artifacts don't exist yet
    if status == 'planned':
        result['metrics_json'] = "—"
        result['runpack'] = "—"
        result['theta_A'] = "—"
        result['theta_AB'] = "—"
        result['training_log'] = "—"
    else:
        # Check actual artifacts
        result['metrics_json'] = check_artifact(run_dir, 'metrics.json')
        result['runpack'] = check_artifact(run_dir, f'runpack_{run_id}.md')
        result['theta_A'] = check_checkpoint(run_dir, 'theta_A.pt')
        result['theta_AB'] = check_checkpoint(run_dir, 'theta_AB.pt')
        result['training_log'] = check_artifact(run_dir, 'training_log.jsonl')

    return result


def find_unlisted_runs(runs_dir: Path, registered_ids: set) -> List[Dict[str, str]]:
    """Find run directories that are not in the registry."""
    unlisted = []

    if not runs_dir.exists():
        return unlisted

    for item in runs_dir.iterdir():
        if item.is_dir() and item.name not in registered_ids:
            # Determine status based on contents
            has_metrics = (item / 'metrics.json').exists()
            has_checkpoints = (item / 'checkpoints').exists()

            if has_metrics:
                status = "orphaned (has metrics)"
            elif has_checkpoints:
                status = "orphaned (has checkpoints)"
            else:
                status = "orphaned"

            # Check what files exist
            files = [f.name for f in item.iterdir() if f.is_file()]

            unlisted.append({
                'directory': item.name,
                'status': status,
                'notes': f"Contains: {', '.join(files[:3])}{'...' if len(files) > 3 else ''}" if files else "Empty"
            })

    return unlisted


def generate_inventory_section(runs: List[Dict[str, str]], runs_dir: Path) -> str:
    """Generate the inventory markdown section."""
    lines = []

    # Main table header
    lines.append("| Run ID | Status | Method | Seed | Start | End | metrics.json | runpack | theta_A.pt | theta_AB.pt | training_log.jsonl | Notes |")
    lines.append("|--------|--------|--------|------|-------|-----|--------------|---------|------------|-------------|-------------------|-------|")

    registered_ids = set()

    for run_info in runs:
        run_id = run_info.get('run_id', '')
        if not run_id:
            continue
        registered_ids.add(run_id)

        inspected = inspect_run(run_id, run_info, runs_dir)

        lines.append(
            f"| {inspected['run_id']} "
            f"| {inspected['status']} "
            f"| {inspected['method']} "
            f"| {inspected['seed']} "
            f"| {inspected['start']} "
            f"| {inspected['end']} "
            f"| {inspected['metrics_json']} "
            f"| {inspected['runpack']} "
            f"| {inspected['theta_A']} "
            f"| {inspected['theta_AB']} "
            f"| {inspected['training_log']} "
            f"| {inspected['notes']} |"
        )

    # Find unlisted runs
    unlisted = find_unlisted_runs(runs_dir, registered_ids)

    if unlisted:
        lines.append("")
        lines.append("**Unlisted runs in `experiments/runs/` (not in registry):**")
        lines.append("| Directory | Status | Notes |")
        lines.append("|-----------|--------|-------|")

        for item in unlisted:
            lines.append(f"| {item['directory']} | {item['status']} | {item['notes']} |")

    return "\n".join(lines)


def update_status_file(status_path: Path, new_inventory: str, dry_run: bool = False) -> bool:
    """
    Update the PROJECT_STATUS.md file with new inventory section.

    Returns True if the file was updated (or would be updated in dry-run mode).
    """
    start_marker = "<!-- INVENTORY_START -->"
    end_marker = "<!-- INVENTORY_END -->"

    if not status_path.exists():
        print(f"ERROR: Status file not found: {status_path}")
        return False

    content = status_path.read_text()

    # Find markers
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print("ERROR: Inventory markers not found in PROJECT_STATUS.md")
        print(f"  Looking for: {start_marker}")
        print(f"  And: {end_marker}")
        return False

    if start_idx >= end_idx:
        print("ERROR: Inventory markers are in wrong order")
        return False

    # Build new content
    before = content[:start_idx + len(start_marker)]
    after = content[end_idx:]

    new_content = before + "\n" + new_inventory + "\n" + after

    if dry_run:
        print("=" * 60)
        print("DRY RUN - Would update inventory section to:")
        print("=" * 60)
        print(new_inventory)
        print("=" * 60)
        print(f"File: {status_path}")
        print("No changes written (dry-run mode)")
        return True
    else:
        status_path.write_text(new_content)
        print(f"Updated: {status_path}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate/update project status inventory section"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--write",
        action="store_true",
        help="Write changes to PROJECT_STATUS.md"
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing"
    )

    args = parser.parse_args()

    project_root = get_project_root()
    registry_path = project_root / "experiments" / "run_registry.csv"
    runs_dir = project_root / "experiments" / "runs"
    status_path = project_root / "docs" / "PROJECT_STATUS.md"

    # Validate paths
    if not registry_path.exists():
        print(f"ERROR: Registry not found: {registry_path}")
        sys.exit(1)

    # Read registry
    print(f"Reading registry: {registry_path}")
    runs = read_registry(registry_path)
    print(f"Found {len(runs)} registered runs")

    # Generate inventory
    print(f"Inspecting run directories: {runs_dir}")
    inventory = generate_inventory_section(runs, runs_dir)

    # Update status file
    success = update_status_file(status_path, inventory, dry_run=args.dry_run)

    if success:
        if args.dry_run:
            print("\nInventory section preview generated successfully.")
        else:
            print("\nInventory section updated successfully.")
        sys.exit(0)
    else:
        print("\nFailed to update inventory section.")
        sys.exit(1)


if __name__ == "__main__":
    main()
