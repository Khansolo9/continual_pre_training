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


def _parse_hms(s):
    """Parse 'MM:SS' or 'HH:MM:SS' to seconds. Returns 0 on '?:??' or junk."""
    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass
    return 0


# Match a tqdm bar tail: "step/total [elapsed<remaining, ...]"
# Captures all four fields from the SAME line so derived rates stay consistent.
_TQDM_PAT = re.compile(
    r"(?P<step>\d+)/(?P<total>\d+)\s+\["
    r"(?P<elapsed>\d+:\d+(?::\d+)?)<"
    r"(?P<remaining>\d+:\d+(?::\d+)?|\?+)"
)


def _windowed_step_duration_estimate(lines, window=20):
    """Estimate steady-state per-step duration from the last `window` Training
    bar lines, robust to pause/spike outliers.

    Method: parse (step, elapsed_sec) pairs from up to ~3*window recent lines,
    compute consecutive step durations as elapsed-time deltas, drop any duration
    > 5x the median (likely SIGSTOP/swap-thrash artifact), and return the
    trimmed median of what remains. None if insufficient data.

    This is more accurate than tqdm's smoothed remaining for runs that have
    pause events or memory-pressure spikes which would otherwise pollute the
    EWMA window for many subsequent steps.
    """
    samples = []  # list of (step, elapsed_sec)
    seen_steps = set()
    for line in reversed(lines[-(3 * window + 50):]):
        if "Training" not in line:
            continue
        m = _TQDM_PAT.search(line)
        if not m:
            continue
        step = int(m.group("step"))
        if step in seen_steps:
            continue
        seen_steps.add(step)
        samples.append((step, _parse_hms(m.group("elapsed"))))
        if len(samples) >= window + 5:
            break
    if len(samples) < 4:
        return None
    samples.sort(key=lambda s: s[0])  # ascending step
    durations = [
        b[1] - a[1] for a, b in zip(samples[:-1], samples[1:])
        if b[0] == a[0] + 1 and b[1] > a[1]
    ]
    if not durations:
        return None
    sorted_d = sorted(durations)
    median = sorted_d[len(sorted_d) // 2]
    # drop outliers > 5x median (SIGSTOP gaps, swap-thrash spikes)
    cleaned = [d for d in durations if d <= 5 * median]
    if not cleaned:
        return median
    cleaned_sorted = sorted(cleaned)
    return cleaned_sorted[len(cleaned_sorted) // 2]


def parse_log(log_path):
    """Parse run.log for phase, step, total, tok/s, loss, elapsed, remaining.

    All progress fields are pulled from a SINGLE tqdm training line so
    step/elapsed/remaining/tok_s remain mutually consistent. Lines without
    'Training' in them (e.g. eval bars, LAMBADA, drift) are skipped to avoid
    counter collisions.

    Adds a `windowed_sec_per_step` field: trimmed median of recent step
    durations, robust to pause events and memory-pressure spikes. Used by
    compute_eta() in preference to tqdm's smoothed remaining.
    """
    result = {
        "phase": "not_started",
        "step": 0,
        "total_steps": 0,
        "tok_per_sec": 0.0,
        "last_loss": 0.0,
        "elapsed_sec": 0.0,
        "remaining_sec": 0.0,  # tqdm's own EWMA-smoothed estimate
        "windowed_sec_per_step": 0.0,  # trimmed-median of recent step times
        "pct": 0.0,
    }

    # Phase detection: scan entire file (phase markers are sparse)
    result["phase"] = scan_phases(log_path)

    lines = tail_lines(log_path)
    if not lines:
        return result

    # Walk backward through recent lines; first training-bar match wins.
    matched_line = None
    for line in reversed(lines[-200:]):
        if "Training" not in line:
            continue
        m = _TQDM_PAT.search(line)
        if not m:
            continue
        result["step"] = int(m.group("step"))
        result["total_steps"] = int(m.group("total"))
        result["elapsed_sec"] = _parse_hms(m.group("elapsed"))
        rem = m.group("remaining")
        result["remaining_sec"] = _parse_hms(rem) if ":" in rem else 0.0
        matched_line = line
        break

    # tok/s and loss must come from the SAME line as step/elapsed.
    if matched_line is not None:
        m = re.search(r"tok/s[=:]\s*([\d.]+)", matched_line)
        if m:
            result["tok_per_sec"] = float(m.group(1))
        m = re.search(r"loss[=:]\s*([\d.]+)", matched_line)
        if m:
            result["last_loss"] = float(m.group(1))

    # Windowed per-step duration (robust to spikes and SIGSTOP gaps)
    spi = _windowed_step_duration_estimate(lines, window=20)
    if spi:
        result["windowed_sec_per_step"] = spi

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
    """Estimate remaining time, robust to pause/spike artifacts.

    Priority:
      1. windowed trimmed-median sec/step over the last ~20 steps
         (excludes SIGSTOP gaps and swap-thrash spikes)
      2. tqdm's own EWMA-smoothed remaining (lags after a spike but
         self-corrects within ~30 steps)
      3. cumulative average (fallback when only a few steps have happened)
    """
    step = progress["step"]
    total = progress["total_steps"]
    elapsed = progress["elapsed_sec"]
    remaining_steps = total - step

    spi = progress.get("windowed_sec_per_step", 0)
    if spi > 0 and remaining_steps > 0:
        return remaining_steps * spi

    rem = progress.get("remaining_sec", 0)
    if rem > 0:
        return rem

    if step > 0 and total > 0 and elapsed > 0 and remaining_steps > 0:
        return remaining_steps * (elapsed / step)
    return 0


def query_swap_free_mb():
    """Query macOS swap free in MB. Returns None on non-mac or query failure.

    Reads `sysctl vm.swapusage` and parses the human-readable line:
      vm.swapusage: total = X.XXM  used = Y.YYM  free = Z.ZZM  (encrypted)
    """
    import subprocess
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True, text=True, timeout=2
        )
        if out.returncode != 0:
            return None
        # Find "free = N.NN[MG]" token
        m = re.search(r"free\s*=\s*([\d.]+)([MG])", out.stdout)
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2)
        return val * 1024 if unit == "G" else val
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def format_pressure(swap_free_mb):
    """Format swap pressure as a short status string with severity hint."""
    if swap_free_mb is None:
        return "n/a"
    if swap_free_mb < 500:
        return f"!{swap_free_mb:.0f}M!"  # red zone — OOM risk imminent
    if swap_free_mb < 1024:
        return f"~{swap_free_mb:.0f}M"   # yellow zone — pressure rising
    if swap_free_mb < 4096:
        return f"{swap_free_mb / 1024:.1f}G"
    return f"{swap_free_mb / 1024:.0f}G"


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

        # Sample swap pressure once per tick — applies to whole machine,
        # not per-run, but displayed per-row for visibility.
        swap_free_mb = query_swap_free_mb()
        pressure_str = format_pressure(swap_free_mb)

        # Compact pressure indicator on the header
        pressure_header = (
            f" | swap-free={pressure_str}" if swap_free_mb is not None else ""
        )

        print(f"CPT Run Tracker | {now} | "
              f"running={n_running} pending={n_pending} completed={n_done}"
              f"{pressure_header}")
        print("=" * 130)
        print(
            f"{'Run ID':<35} {'Status':<10} {'Phase':<18} "
            f"{'Progress':>10} {'Tok/s':>7} {'Loss':>8} "
            f"{'Elapsed':>9} {'ETA':>9} {'Throttle':>8} {'Swap':>8}"
        )
        print("-" * 130)

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

            # Per-row swap is the same global value; shown here for at-a-glance
            # visibility next to throttle. Color/decor handled in format_pressure.
            row_pressure = pressure_str if status == "running" else "—"

            print(
                f"{run_id:<35} {status:<10} {phase_str:<18} "
                f"{step_str:>10} {tok_str:>7} {loss_str:>8} "
                f"{elapsed_str:>9} {eta_str:>9} {throttle_str:>8} "
                f"{row_pressure:>8}"
            )

            # Write progress.json
            save_progress_json(run_dir, progress, hist)

        if args.once:
            break

        iteration += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
