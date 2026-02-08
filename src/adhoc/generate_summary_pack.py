#!/usr/bin/env python3
"""
Summary Pack Generator

Generates a unified analysis pack for all completed experiment runs:
- experiments/summary_pack.md: Comprehensive markdown report
- experiments/summary_table.csv: Tabular data for all runs

Usage:
    python src/adhoc/generate_summary_pack.py --dry-run    # Preview outputs
    python src/adhoc/generate_summary_pack.py --write      # Write outputs
    python src/adhoc/generate_summary_pack.py --write --include-smoke  # Include smoke runs

By default, smoke runs are EXCLUDED from summaries:
- research_question == "SMOKE"
- run_id starts with "smoke_"

Inputs:
    - experiments/run_registry.csv
    - experiments/runs/<run_id>/metrics.json (for completed runs)
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


# Key metrics for analysis (order matters for tables)
KEY_METRICS = [
    "forgetting_pct",
    "ppl_a_after",
    "ppl_b_after",
    "lambada_after",
    "rep4_after",
    "drift_value_after",
    "total_hours",
    "avg_tokens_per_sec",
]

# Metrics where lower is better
LOWER_IS_BETTER = {
    "forgetting_pct",
    "ppl_a_after",
    "ppl_b_after",
    "rep4_after",
    "drift_value_after",
    "total_hours",
}

# Metric display names
METRIC_NAMES = {
    "forgetting_pct": "Forgetting %",
    "ppl_a_after": "PPL(A) After",
    "ppl_b_after": "PPL(B) After",
    "lambada_after": "LAMBADA After",
    "rep4_after": "Rep-4 After",
    "drift_value_after": "Drift (JS)",
    "total_hours": "Total Hours",
    "avg_tokens_per_sec": "Avg Tok/s",
}


def is_smoke_run(reg: Dict[str, str]) -> bool:
    """
    Check if a run is a smoke test run.

    Smoke runs are identified by:
    - research_question == "SMOKE"
    - run_id starts with "smoke_"
    """
    rq = reg.get("research_question", "").upper()
    run_id = reg.get("run_id", "")
    return rq == "SMOKE" or run_id.lower().startswith("smoke_")


def read_registry(registry_path: Path) -> List[Dict[str, str]]:
    """Read the run registry CSV file."""
    runs = []
    with open(registry_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row)
    return runs


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load and validate a metrics.json file."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    # Validate required sections
    required_sections = ["run", "metrics", "resources"]
    for section in required_sections:
        if section not in data:
            raise ValueError(f"Missing required section '{section}' in {metrics_path}")

    return data


def get_metric_value(data: Dict[str, Any], metric: str) -> Optional[float]:
    """Extract a metric value from metrics.json data."""
    # Check metrics section first
    if metric in data.get("metrics", {}):
        return data["metrics"][metric]
    # Then check resources section
    if metric in data.get("resources", {}):
        return data["resources"][metric]
    return None


def format_value(value: Optional[float], metric: str) -> str:
    """Format a metric value for display."""
    if value is None:
        return "—"

    if metric == "forgetting_pct":
        return f"{value:.2f}%"
    elif metric in ["ppl_a_after", "ppl_b_after"]:
        return f"{value:.2f}"
    elif metric == "lambada_after":
        return f"{value:.3f}"
    elif metric in ["rep4_after", "drift_value_after"]:
        return f"{value:.4f}"
    elif metric == "total_hours":
        return f"{value:.2f}h"
    elif metric == "avg_tokens_per_sec":
        return f"{value:.0f}"
    else:
        return f"{value:.4f}"


def compute_baseline_stats(baseline_runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, min, max for baseline runs."""
    stats = {}

    for metric in KEY_METRICS:
        values = []
        for run_data in baseline_runs:
            val = get_metric_value(run_data, metric)
            if val is not None:
                values.append(val)

        if len(values) >= 1:
            stats[metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }

    return stats


def compute_delta(method_value: Optional[float], baseline_mean: float, metric: str) -> Optional[float]:
    """Compute delta from baseline mean (negative = better for LOWER_IS_BETTER metrics)."""
    if method_value is None:
        return None
    return method_value - baseline_mean


def generate_executive_summary(
    runs_data: List[Tuple[Dict[str, str], Dict[str, Any]]],
    baseline_stats: Dict[str, Dict[str, float]]
) -> str:
    """Generate executive summary section."""
    lines = []
    lines.append("## Executive Summary")
    lines.append("")

    # Count runs by status
    total_runs = len(runs_data)
    methods = set()
    baseline_count = 0
    method_runs = []

    for reg, data in runs_data:
        method = reg.get("method", "unknown")
        methods.add(method)
        if method == "baseline":
            baseline_count += 1
        else:
            method_runs.append((reg, data))

    lines.append(f"**Completed Runs**: {total_runs}")
    lines.append(f"**Methods Tested**: {', '.join(sorted(methods))}")
    lines.append(f"**Baseline Seeds**: {baseline_count}")
    lines.append("")

    # Baseline summary
    if "forgetting_pct" in baseline_stats:
        fg = baseline_stats["forgetting_pct"]
        lines.append(f"**Baseline Forgetting**: {fg['mean']:.2f}% ± {fg['std']:.2f}% (n={fg['n']})")

    # Find best method for key metrics
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")

    if method_runs and baseline_stats:
        # Find best forgetting reduction
        best_forgetting = None
        best_forgetting_method = None
        baseline_forgetting_mean = baseline_stats.get("forgetting_pct", {}).get("mean")

        for reg, data in method_runs:
            fg = get_metric_value(data, "forgetting_pct")
            if fg is not None and baseline_forgetting_mean is not None:
                if best_forgetting is None or fg < best_forgetting:
                    best_forgetting = fg
                    best_forgetting_method = reg.get("method", "unknown")

        if best_forgetting is not None and baseline_forgetting_mean is not None:
            reduction = baseline_forgetting_mean - best_forgetting
            lines.append(f"- **Best Forgetting**: {best_forgetting_method} ({best_forgetting:.2f}%, "
                        f"Δ={reduction:+.2f}% vs baseline)")

        # Find fastest method
        fastest_time = None
        fastest_method = None
        for reg, data in method_runs:
            hours = get_metric_value(data, "total_hours")
            if hours is not None:
                if fastest_time is None or hours < fastest_time:
                    fastest_time = hours
                    fastest_method = reg.get("method", "unknown")

        if fastest_time is not None:
            lines.append(f"- **Fastest Method**: {fastest_method} ({fastest_time:.2f}h)")

        # Find highest throughput
        best_throughput = None
        best_throughput_method = None
        for reg, data in method_runs:
            tps = get_metric_value(data, "avg_tokens_per_sec")
            if tps is not None:
                if best_throughput is None or tps > best_throughput:
                    best_throughput = tps
                    best_throughput_method = reg.get("method", "unknown")

        if best_throughput is not None:
            lines.append(f"- **Highest Throughput**: {best_throughput_method} ({best_throughput:.0f} tok/s)")

    lines.append("")
    return "\n".join(lines)


def generate_comparison_table(runs_data: List[Tuple[Dict[str, str], Dict[str, Any]]]) -> str:
    """Generate full comparison table across all runs."""
    lines = []
    lines.append("## Full Comparison Table")
    lines.append("")

    # Header
    header = ["Run ID", "Method", "Seed"] + [METRIC_NAMES[m] for m in KEY_METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    # Rows
    for reg, data in runs_data:
        row = [
            reg.get("run_id", "—"),
            reg.get("method", "—"),
            reg.get("seed", "—"),
        ]
        for metric in KEY_METRICS:
            val = get_metric_value(data, metric)
            row.append(format_value(val, metric))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_baseline_variance(baseline_stats: Dict[str, Dict[str, float]]) -> str:
    """Generate baseline variance analysis section."""
    lines = []
    lines.append("## Baseline Variance Analysis")
    lines.append("")

    if not baseline_stats:
        lines.append("*No baseline runs available for variance analysis.*")
        lines.append("")
        return "\n".join(lines)

    # Header
    lines.append("| Metric | Mean | Std | Min | Max | N |")
    lines.append("|--------|------|-----|-----|-----|---|")

    for metric in KEY_METRICS:
        if metric not in baseline_stats:
            continue
        s = baseline_stats[metric]
        row = [
            METRIC_NAMES[metric],
            format_value(s["mean"], metric),
            format_value(s["std"], metric).replace("h", "").replace("%", ""),
            format_value(s["min"], metric),
            format_value(s["max"], metric),
            str(s["n"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_method_deltas(
    runs_data: List[Tuple[Dict[str, str], Dict[str, Any]]],
    baseline_stats: Dict[str, Dict[str, float]]
) -> str:
    """Generate method deltas vs baseline mean."""
    lines = []
    lines.append("## Method Deltas vs Baseline Mean")
    lines.append("")

    # Filter to non-baseline runs
    method_runs = [(reg, data) for reg, data in runs_data if reg.get("method") != "baseline"]

    if not method_runs or not baseline_stats:
        lines.append("*No method comparisons available.*")
        lines.append("")
        return "\n".join(lines)

    lines.append("*Negative delta = improvement for lower-is-better metrics (forgetting, PPL, rep, drift, time)*")
    lines.append("*Positive delta = improvement for higher-is-better metrics (LAMBADA, tok/s)*")
    lines.append("")

    # Header
    header = ["Method", "Seed"] + [f"Δ {METRIC_NAMES[m]}" for m in KEY_METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for reg, data in method_runs:
        row = [
            reg.get("method", "—"),
            reg.get("seed", "—"),
        ]
        for metric in KEY_METRICS:
            val = get_metric_value(data, metric)
            baseline_mean = baseline_stats.get(metric, {}).get("mean")

            if val is not None and baseline_mean is not None:
                delta = val - baseline_mean
                # Format delta
                if metric == "forgetting_pct":
                    row.append(f"{delta:+.2f}%")
                elif metric in ["ppl_a_after", "ppl_b_after"]:
                    row.append(f"{delta:+.2f}")
                elif metric == "lambada_after":
                    row.append(f"{delta:+.3f}")
                elif metric in ["rep4_after", "drift_value_after"]:
                    row.append(f"{delta:+.4f}")
                elif metric == "total_hours":
                    row.append(f"{delta:+.2f}h")
                elif metric == "avg_tokens_per_sec":
                    row.append(f"{delta:+.0f}")
                else:
                    row.append(f"{delta:+.4f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_pareto_ranking(
    runs_data: List[Tuple[Dict[str, str], Dict[str, Any]]],
    baseline_stats: Dict[str, Dict[str, float]]
) -> str:
    """Generate Pareto-style ranking notes."""
    lines = []
    lines.append("## Pareto-Style Rankings")
    lines.append("")

    method_runs = [(reg, data) for reg, data in runs_data if reg.get("method") != "baseline"]

    if not method_runs:
        lines.append("*No method runs available for ranking.*")
        lines.append("")
        return "\n".join(lines)

    baseline_forgetting = baseline_stats.get("forgetting_pct", {}).get("mean", 0)

    # Compute efficiency metrics
    rankings = []
    for reg, data in method_runs:
        method = reg.get("method", "unknown")
        forgetting = get_metric_value(data, "forgetting_pct")
        hours = get_metric_value(data, "total_hours")
        ppl_a = get_metric_value(data, "ppl_a_after")
        lambada = get_metric_value(data, "lambada_after")
        tps = get_metric_value(data, "avg_tokens_per_sec")

        if forgetting is not None and hours is not None and hours > 0:
            forgetting_reduction = baseline_forgetting - forgetting
            reduction_per_hour = forgetting_reduction / hours
            rankings.append({
                "method": method,
                "forgetting": forgetting,
                "hours": hours,
                "ppl_a": ppl_a,
                "lambada": lambada,
                "tps": tps,
                "reduction_per_hour": reduction_per_hour,
                "forgetting_reduction": forgetting_reduction,
            })

    if not rankings:
        lines.append("*Insufficient data for Pareto rankings.*")
        lines.append("")
        return "\n".join(lines)

    # Best forgetting reduction per hour
    lines.append("### Best Forgetting Reduction per Hour")
    sorted_by_efficiency = sorted(rankings, key=lambda x: x["reduction_per_hour"], reverse=True)
    for i, r in enumerate(sorted_by_efficiency, 1):
        lines.append(f"{i}. **{r['method']}**: {r['reduction_per_hour']:.3f}%/h "
                    f"(Δ={r['forgetting_reduction']:+.2f}% in {r['hours']:.2f}h)")
    lines.append("")

    # Best retention (lowest forgetting)
    lines.append("### Best Domain A Retention (Lowest Forgetting)")
    sorted_by_forgetting = sorted(rankings, key=lambda x: x["forgetting"])
    for i, r in enumerate(sorted_by_forgetting, 1):
        lines.append(f"{i}. **{r['method']}**: {r['forgetting']:.2f}%")
    lines.append("")

    # Best general ability (LAMBADA)
    lines.append("### Best General Ability (LAMBADA)")
    sorted_by_lambada = sorted(rankings, key=lambda x: x["lambada"] or 0, reverse=True)
    for i, r in enumerate(sorted_by_lambada, 1):
        if r["lambada"] is not None:
            lines.append(f"{i}. **{r['method']}**: {r['lambada']:.3f}")
    lines.append("")

    # Best throughput
    lines.append("### Best Throughput")
    sorted_by_tps = sorted(rankings, key=lambda x: x["tps"] or 0, reverse=True)
    for i, r in enumerate(sorted_by_tps, 1):
        if r["tps"] is not None:
            lines.append(f"{i}. **{r['method']}**: {r['tps']:.0f} tok/s")
    lines.append("")

    return "\n".join(lines)


def generate_anomalies_section(runs_data: List[Tuple[Dict[str, str], Dict[str, Any]]]) -> str:
    """Generate anomalies section summarizing anomalies from all runs."""
    lines = []
    lines.append("## Anomalies Summary")
    lines.append("")

    anomaly_counts: Dict[str, List[str]] = {}

    for reg, data in runs_data:
        run_id = reg.get("run_id", "unknown")
        anomalies = data.get("anomalies", [])
        for anomaly in anomalies:
            if anomaly not in anomaly_counts:
                anomaly_counts[anomaly] = []
            anomaly_counts[anomaly].append(run_id)

    if not anomaly_counts:
        lines.append("*No anomalies detected across all runs.*")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Anomaly | Count | Affected Runs |")
    lines.append("|---------|-------|---------------|")

    for anomaly, runs in sorted(anomaly_counts.items()):
        runs_str = ", ".join(runs[:5])
        if len(runs) > 5:
            runs_str += f" (+{len(runs)-5} more)"
        lines.append(f"| {anomaly} | {len(runs)} | {runs_str} |")

    lines.append("")
    return "\n".join(lines)


def generate_appendix(runs_data: List[Tuple[Dict[str, str], Dict[str, Any]]]) -> str:
    """Generate appendix with artifact paths per run."""
    lines = []
    lines.append("## Appendix: Artifact Paths")
    lines.append("")

    for reg, data in runs_data:
        run_id = reg.get("run_id", "unknown")
        lines.append(f"### {run_id}")
        lines.append("")
        lines.append(f"- **Metrics**: `experiments/runs/{run_id}/metrics.json`")
        lines.append(f"- **Runpack**: `experiments/runs/{run_id}/runpack_{run_id}.md`")
        lines.append(f"- **Config**: `experiments/runs/{run_id}/config.yaml`")
        lines.append(f"- **Checkpoints**:")
        lines.append(f"  - `experiments/runs/{run_id}/checkpoints/theta_A.pt`")
        lines.append(f"  - `experiments/runs/{run_id}/checkpoints/theta_AB.pt`")
        lines.append(f"- **Training Log**: `experiments/runs/{run_id}/training_log.jsonl`")
        lines.append("")

    return "\n".join(lines)


def generate_summary_pack(runs_data: List[Tuple[Dict[str, str], Dict[str, Any]]]) -> str:
    """Generate the complete summary_pack.md content."""
    lines = []

    # Header
    lines.append("# Experiment Summary Pack")
    lines.append("")
    lines.append(f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append(f"**Total Completed Runs**: {len(runs_data)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Compute baseline stats
    baseline_runs = [data for reg, data in runs_data if reg.get("method") == "baseline"]
    baseline_stats = compute_baseline_stats(baseline_runs)

    # Sections
    lines.append(generate_executive_summary(runs_data, baseline_stats))
    lines.append("---")
    lines.append("")
    lines.append(generate_comparison_table(runs_data))
    lines.append("---")
    lines.append("")
    lines.append(generate_baseline_variance(baseline_stats))
    lines.append("---")
    lines.append("")
    lines.append(generate_method_deltas(runs_data, baseline_stats))
    lines.append("---")
    lines.append("")
    lines.append(generate_pareto_ranking(runs_data, baseline_stats))
    lines.append("---")
    lines.append("")
    lines.append(generate_anomalies_section(runs_data))
    lines.append("---")
    lines.append("")
    lines.append(generate_appendix(runs_data))

    return "\n".join(lines)


def generate_summary_csv(runs_data: List[Tuple[Dict[str, str], Dict[str, Any]]]) -> str:
    """Generate the summary_table.csv content."""
    # Define all columns
    columns = [
        # Run metadata
        "run_id", "research_question", "method", "seed", "status",
        "timestamp_start", "timestamp_end",
        # Key metrics
        "forgetting_pct", "ppl_a_init", "ppl_b_init",
        "ppl_a_before", "ppl_a_after", "ppl_b_after",
        "lambada_before", "lambada_after",
        "rep4_before", "rep4_after", "rep8_before", "rep8_after",
        "drift_value_before", "drift_value_after",
        "vocab_overlap_before", "vocab_overlap_after",
        # Resources
        "total_hours", "domain_a_hours", "domain_b_hours",
        "peak_vram_gb", "peak_ram_gb", "avg_tokens_per_sec",
        # Inputs
        "domain_a_name", "domain_b_name",
        "domain_a_tokens_used", "domain_b_tokens_used",
        # Method params (as JSON string)
        "method_params",
        # Anomalies (as semicolon-separated string)
        "anomalies",
    ]

    rows = []
    for reg, data in runs_data:
        row = {}

        # Run metadata from registry
        row["run_id"] = reg.get("run_id", "")
        row["research_question"] = reg.get("research_question", "")
        row["method"] = reg.get("method", "")
        row["seed"] = reg.get("seed", "")
        row["status"] = reg.get("status", "")
        row["timestamp_start"] = reg.get("timestamp_start", "")
        row["timestamp_end"] = reg.get("timestamp_end", "")

        # Metrics
        metrics = data.get("metrics", {})
        for key in ["forgetting_pct", "ppl_a_init", "ppl_b_init",
                    "ppl_a_before", "ppl_a_after", "ppl_b_after",
                    "lambada_before", "lambada_after",
                    "rep4_before", "rep4_after", "rep8_before", "rep8_after",
                    "drift_value_before", "drift_value_after",
                    "vocab_overlap_before", "vocab_overlap_after"]:
            row[key] = metrics.get(key, "")

        # Resources
        resources = data.get("resources", {})
        for key in ["total_hours", "domain_a_hours", "domain_b_hours",
                    "peak_vram_gb", "peak_ram_gb", "avg_tokens_per_sec"]:
            row[key] = resources.get(key, "")

        # Inputs
        inputs = data.get("inputs", {})
        row["domain_a_name"] = inputs.get("domain_a_name", "")
        row["domain_b_name"] = inputs.get("domain_b_name", "")
        row["domain_a_tokens_used"] = inputs.get("domain_a_tokens_used", "")
        row["domain_b_tokens_used"] = inputs.get("domain_b_tokens_used", "")

        # Method params as JSON string
        method_params = data.get("method_params", {})
        row["method_params"] = json.dumps(method_params) if method_params else ""

        # Anomalies as semicolon-separated
        anomalies = data.get("anomalies", [])
        row["anomalies"] = ";".join(anomalies) if anomalies else ""

        rows.append(row)

    # Build CSV string
    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Generate unified summary pack for all completed experiment runs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--write",
        action="store_true",
        help="Write summary_pack.md and summary_table.csv"
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview outputs without writing"
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke test runs (excluded by default)"
    )

    args = parser.parse_args()

    project_root = get_project_root()
    registry_path = project_root / "experiments" / "run_registry.csv"
    runs_dir = project_root / "experiments" / "runs"
    output_md = project_root / "experiments" / "summary_pack.md"
    output_csv = project_root / "experiments" / "summary_table.csv"

    # Validate paths
    if not registry_path.exists():
        print(f"ERROR: Registry not found: {registry_path}")
        sys.exit(1)

    # Read registry
    print(f"Reading registry: {registry_path}")
    registry_runs = read_registry(registry_path)
    print(f"Found {len(registry_runs)} registered runs")

    # Smoke run filtering
    if args.include_smoke:
        print("Including smoke test runs")
    else:
        print("Excluding smoke test runs (use --include-smoke to include)")

    # Filter to completed runs and load metrics
    runs_data: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
    errors = []
    skipped_smoke = 0

    for reg in registry_runs:
        run_id = reg.get("run_id", "")
        status = reg.get("status", "")

        if status != "completed":
            print(f"  Skipping {run_id} (status: {status})")
            continue

        # Filter smoke runs unless --include-smoke is set
        if not args.include_smoke and is_smoke_run(reg):
            skipped_smoke += 1
            print(f"  Skipping {run_id} (smoke test)")
            continue

        metrics_path = runs_dir / run_id / "metrics.json"
        try:
            data = load_metrics(metrics_path)
            runs_data.append((reg, data))
            print(f"  Loaded: {run_id}")
        except FileNotFoundError as e:
            errors.append(f"{run_id}: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            errors.append(f"{run_id}: Invalid metrics.json - {e}")

    if errors:
        print("\nERRORS loading metrics:")
        for err in errors:
            print(f"  - {err}")
        print("\nAborting due to errors.")
        sys.exit(1)

    if not runs_data:
        print("\nNo completed runs found with valid metrics.")
        sys.exit(1)

    print(f"\nLoaded {len(runs_data)} completed runs with valid metrics")
    if skipped_smoke > 0:
        print(f"(Excluded {skipped_smoke} smoke test runs)")

    # Generate outputs
    print("\nGenerating summary pack...")
    md_content = generate_summary_pack(runs_data)
    csv_content = generate_summary_csv(runs_data)

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Would write to:")
        print("=" * 60)
        print(f"\n1. {output_md}")
        print("-" * 60)
        # Show first 100 lines of MD
        md_lines = md_content.split("\n")
        for line in md_lines[:100]:
            print(line)
        if len(md_lines) > 100:
            print(f"... ({len(md_lines) - 100} more lines)")

        print("\n" + "-" * 60)
        print(f"\n2. {output_csv}")
        print("-" * 60)
        # Show first 20 lines of CSV
        csv_lines = csv_content.split("\n")
        for line in csv_lines[:20]:
            print(line)
        if len(csv_lines) > 20:
            print(f"... ({len(csv_lines) - 20} more lines)")

        print("\n" + "=" * 60)
        print("No files written (dry-run mode)")
    else:
        output_md.write_text(md_content)
        print(f"Wrote: {output_md}")

        output_csv.write_text(csv_content)
        print(f"Wrote: {output_csv}")

        print("\nSummary pack generated successfully.")

    sys.exit(0)


if __name__ == "__main__":
    main()
