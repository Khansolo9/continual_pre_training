#!/usr/bin/env python3
"""
Track progress of running CPT experiments.

Reads run_registry.csv, watches log files, and shows live single-line
status updates with step progress, tok/s, phase, ETA, and throttle detection.

Usage:
    python src/adhoc/track_run.py                                # all running/pending
    python src/adhoc/track_run.py --run-id qwen3_rq1_baseline_s42
    python src/adhoc/track_run.py --interval 5                   # update every 5s
    python src/adhoc/track_run.py --once                         # single snapshot
    python src/adhoc/track_run.py --all                          # include completed

Writes progress.json per run to experiments/runs/{run_id}/progress.json.
Stdlib-only (no external dependencies).
"""

import sys
import csv
import json
import re
import os
import time
from pathlib import Path
from datetime import datetime


def get_project_root():
    """Resolve project root from this script's location."""
    return Path(__file__).resolve().parent.parent.parent


def read_registry(project_root):
    """Read run_registry.csv and return list of row dicts."""
    path = project_root / "experiments" / "run_registry.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def tail_lines(filepath, n_bytes=65536):
    """Read the last ~n_bytes of a file and return lines."""
    try:
        size = filepath.stat().st_size
        with open(filepath, "rb") as f:
            f.seek(max(0, size - n_bytes))
            content = f.read().decode("utf-8", errors="replace")
            return content.splitlines()
    except (FileNotFoundError, OSError):
        return []


def scan_phases(log_path):
    """Scan entire log file for phase markers (lightweight line scan)."""
    phase = "not_started"
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if "[1/5]" in line:
                    phase = "[1/5] Init eval"
                elif "[2/5]" in line and "[2.5/5]" not in line:
                    phase = "[2/5] Domain A"
                elif "[2.5/5]" in line:
                    phase = "[2.5/5] CL setup"
                elif "[3/5]" in line:
                    phase = "[3/5] Post-A eval"
                elif "[4/5]" in line:
                    phase = "[4/5] Domain B"
                elif "[5/5]" in line:
                    phase = "[5/5] Final eval"
                elif "completed!" in line:
                    phase = "completed"
                elif "Experiment failed" in line:
                    phase = "FAILED"
    except (FileNotFoundError, OSError):
        pass
    return phase


def parse_log(log_path):
    """Parse run.log for phase, step, total, tok/s, loss, elapsed."""
    result = {
        "phase": "not_started",
        "step": 0,
        "total_steps": 0,
        "tok_per_sec": 0.0,
        "last_loss": 0.0,
        "elapsed_sec": 0.0,
        "pct": 0.0,
    }

    # Phase detection: scan entire file (phase markers are sparse)
    result["phase"] = scan_phases(log_path)

    lines = tail_lines(log_path)
    if not lines:
        return result

    # --- Step/total from tqdm (search recent lines, last match wins) ---
    recent = lines[-100:]
    for line in reversed(recent):
        # tqdm: "Training ...: XX%|...| step/total [elapsed<remaining, ...]"
        m = re.search(r"(\d+)/(\d+)\s*\[", line)
        if m:
            result["step"] = int(m.group(1))
            result["total_steps"] = int(m.group(2))
            break

    # --- tok/s from tqdm postfix ---
    for line in reversed(recent):
        m = re.search(r"tok/s[=:]\s*([\d.]+)", line)
        if m:
            result["tok_per_sec"] = float(m.group(1))
            break

    # --- loss from tqdm postfix ---
    for line in reversed(recent):
        m = re.search(r"loss[=:]\s*([\d.]+)", line)
        if m:
            result["last_loss"] = float(m.group(1))
            break

    # --- elapsed from tqdm [MM:SS<...] or [HH:MM:SS<...] ---
    for line in reversed(recent):
        m = re.search(r"\[(\d+:\d+(?::\d+)?)<", line)
        if m:
            parts = m.group(1).split(":")
            if len(parts) == 2:
                result["elapsed_sec"] = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                result["elapsed_sec"] = (
                    int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                )
            break

    # --- Percentage ---
    if result["total_steps"] > 0:
        result["pct"] = round(100.0 * result["step"] / result["total_steps"], 1)

    return result


def format_duration(seconds):
    """Format seconds as human-readable duration."""
    if seconds <= 0:
        return "—"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def compute_eta(progress):
    """Estimate remaining time from step progress."""
    step = progress["step"]
    total = progress["total_steps"]
    elapsed = progress["elapsed_sec"]

    if step > 0 and total > 0 and elapsed > 0:
        remaining_steps = total - step
        sec_per_step = elapsed / step
        return remaining_steps * sec_per_step
    return 0


def detect_throttle(current_tok_s, history, threshold=0.30):
    """Detect thermal throttling: tok/s drop > threshold from rolling mean.

    Returns (flag_str, updated_history).
    history is a list of recent tok/s readings (max 10).
    """
    if current_tok_s <= 0:
        return "—", history

    updated = (history + [current_tok_s])[-10:]

    if len(updated) < 3:
        return "—", updated

    # Compare current to mean of previous readings (exclude current)
    prev = updated[:-1]
    mean_prev = sum(prev) / len(prev)

    if mean_prev > 0 and current_tok_s < mean_prev * (1.0 - threshold):
        drop_pct = round(100.0 * (1.0 - current_tok_s / mean_prev))
        return f"-{drop_pct}%", updated

    return "ok", updated


def load_progress_json(run_dir):
    """Load existing progress.json for throttle history."""
    path = run_dir / "progress.json"
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_progress_json(run_dir, progress, throttle_history):
    """Save progress.json with current state and throttle history."""
    if not run_dir.exists():
        return
    data = dict(progress)
    data["timestamp"] = datetime.now().isoformat()
    data["tok_per_sec_history"] = throttle_history
    with open(run_dir / "progress.json", "w") as f:
        json.dump(data, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Track CPT experiment progress"
    )
    parser.add_argument(
        "--run-id", nargs="*",
        help="Specific run IDs to track (default: all running/pending)"
    )
    parser.add_argument(
        "--interval", type=int, default=10,
        help="Update interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Print a single snapshot and exit"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show all runs, not just running/pending"
    )
    args = parser.parse_args()

    project_root = get_project_root()

    # Per-run throttle history (persisted in progress.json)
    throttle_histories = {}

    iteration = 0
    while True:
        rows = read_registry(project_root)

        # Filter rows
        if args.run_id:
            rows = [r for r in rows if r["run_id"] in args.run_id]
        elif not args.all:
            rows = [r for r in rows if r["status"] in ("running", "pending")]

        if not rows and iteration == 0:
            print("No matching runs found in registry.")
            break

        # Header
        now = datetime.now().strftime("%H:%M:%S")
        n_pending = sum(1 for r in rows if r["status"] == "pending")
        n_running = sum(1 for r in rows if r["status"] == "running")
        n_done = sum(1 for r in rows if r["status"] == "completed")

        if not args.once:
            print("\033[2J\033[H", end="")  # clear screen

        print(f"CPT Run Tracker | {now} | "
              f"running={n_running} pending={n_pending} completed={n_done}")
        print("=" * 120)
        print(
            f"{'Run ID':<35} {'Status':<10} {'Phase':<18} "
            f"{'Progress':>10} {'Tok/s':>7} {'Loss':>8} "
            f"{'Elapsed':>9} {'ETA':>9} {'Throttle':>8}"
        )
        print("-" * 120)

        for row in rows:
            run_id = row["run_id"]
            status = row["status"]
            run_dir = project_root / "experiments" / "runs" / run_id
            log_path = run_dir / "run.log"

            progress = parse_log(log_path)

            # Phase display
            if status == "running":
                phase_str = progress["phase"]
            elif status == "completed":
                phase_str = "done"
            elif status == "failed":
                phase_str = "FAILED"
            else:
                phase_str = "—"

            # Step progress
            if progress["total_steps"] > 0:
                step_str = f"{progress['step']}/{progress['total_steps']}"
            else:
                step_str = "—"

            # Tok/s
            tok_str = (
                f"{progress['tok_per_sec']:.0f}"
                if progress["tok_per_sec"] > 0 else "—"
            )

            # Loss
            loss_str = (
                f"{progress['last_loss']:.4f}"
                if progress["last_loss"] > 0 else "—"
            )

            # Elapsed
            elapsed_str = format_duration(progress["elapsed_sec"])

            # ETA
            eta_sec = compute_eta(progress)
            eta_str = format_duration(eta_sec)

            # Throttle detection
            hist = throttle_histories.get(run_id, [])
            throttle_str, hist = detect_throttle(
                progress["tok_per_sec"], hist
            )
            throttle_histories[run_id] = hist

            print(
                f"{run_id:<35} {status:<10} {phase_str:<18} "
                f"{step_str:>10} {tok_str:>7} {loss_str:>8} "
                f"{elapsed_str:>9} {eta_str:>9} {throttle_str:>8}"
            )

            # Write progress.json
            save_progress_json(run_dir, progress, hist)

        if args.once:
            break

        iteration += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
