#!/usr/bin/env python3
"""
Multi-Model Analysis Assets Generator for Continual Pretraining Experiments

Loads metrics directly from experiments/runs/*/metrics.json, cross-references
with experiments/run_registry.csv, and generates publication-quality tables
and figures covering all model families and CL methods.

Usage:
    cd continual_pre_training
    source .cpt-env/bin/activate
    python experiments/analysis/scripts/make_analysis_assets.py

Outputs:
    experiments/analysis/tables/*.csv, *.md
    experiments/analysis/figures/*.png, *.pdf
    experiments/summary_table.csv  (regenerated from metrics.json)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ORDER = ["gpt2", "qwen3", "gemma3", "llama3"]
MODEL_LABELS = {
    "gpt2": "GPT-2 (124M)",
    "qwen3": "Qwen3 (0.6B)",
    "gemma3": "Gemma3 (1B)",
    "llama3": "Llama3 (1B)",
}
MODEL_PARAMS = {"gpt2": 124, "qwen3": 600, "gemma3": 1000, "llama3": 1000}

METHOD_ORDER = ["baseline", "replay25", "mer25", "ewc", "bandit_replay", "rmgs"]
METHOD_LABELS = {
    "baseline": "Baseline",
    "replay25": "Replay-25%",
    "mer25": "MER-lite",
    "ewc": "EWC",
    "bandit_replay": "Bandit Replay",
    "rmgs": "RMGS",
}
METHOD_COLORS = {
    "baseline": "#4C72B0",
    "replay25": "#55A868",
    "mer25": "#C44E52",
    "ewc": "#8172B3",
    "bandit_replay": "#DD8452",
    "rmgs": "#937860",
}
MODEL_MARKERS = {"gpt2": "o", "qwen3": "s", "gemma3": "D", "llama3": "^"}
MODEL_COLORS = {
    "gpt2": "#4C72B0",
    "qwen3": "#55A868",
    "gemma3": "#C44E52",
    "llama3": "#8172B3",
}

# Key metrics used across tables/figures
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


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def load_all_runs(project_root: Path) -> pd.DataFrame:
    """Build a dataframe from metrics.json files, enriched by registry metadata."""
    registry = pd.read_csv(project_root / "experiments" / "run_registry.csv")

    runs_dir = project_root / "experiments" / "runs"
    rows = []
    for mf in sorted(runs_dir.glob("*/metrics.json")):
        run_id = mf.parent.name
        with open(mf) as f:
            data = json.load(f)

        # Registry is source of truth for labels
        reg = registry[registry["run_id"] == run_id]
        if reg.empty:
            print(f"  SKIP {run_id}: not in registry")
            continue

        reg = reg.iloc[0]
        if reg["status"] != "completed":
            continue

        row = {
            "run_id": run_id,
            "research_question": reg["research_question"],
            "method": reg["method"],
            "seed": int(reg["seed"]),
            "model_family": reg["model_family"],
            "model_params_m": int(reg["model_params_m"]),
        }

        # Pull numeric metrics
        metrics = data.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[k] = v

        resources = data.get("resources", {})
        for k, v in resources.items():
            if isinstance(v, (int, float)):
                row[k] = v

        # RL-method stats (bandit_replay, rmgs) — pulled from rl_method_stats block
        rl_stats = data.get("rl_method_stats") or {}
        for k, v in rl_stats.items():
            if isinstance(v, (int, float)):
                row[f"rl_{k}"] = v

        # Filter out the retired high_rep4 flag (see docs/PROJECT_STATUS.md
        # 2026-05-09 retirement note). Frozen metrics.json files keep it for
        # audit but the analysis pipeline no longer surfaces it.
        retired_anomalies = {"high_rep4"}
        anomalies = [a for a in data.get("anomalies", []) if a not in retired_anomalies]
        row["anomalies"] = ",".join(anomalies)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} completed runs from metrics.json")
    return df


def filter_core(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude SMOKE runs."""
    return df[df["research_question"] != "SMOKE"].copy()


def save_figure(fig, path_stem: Path):
    fig.savefig(path_stem.with_suffix(".png"), dpi=300)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# TABLE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def table1_all_results(df: pd.DataFrame, out: Path) -> None:
    """Table 1: All core results — one row per run."""
    cols = [
        "run_id", "model_family", "method", "seed",
        "forgetting_pct", "ppl_a_after", "ppl_b_after",
        "lambada_after", "rep4_after", "drift_value_after",
        "total_hours", "avg_tokens_per_sec",
    ]
    t = df[cols].copy()
    t = t.sort_values(["model_family", "method", "seed"],
                      key=lambda s: s.map({v: i for i, v in enumerate(MODEL_ORDER)})
                      if s.name == "model_family"
                      else s.map({v: i for i, v in enumerate(METHOD_ORDER)})
                      if s.name == "method" else s)

    t.to_csv(out / "table1_all_results.csv", index=False)

    # Markdown
    disp = t.copy()
    disp["forgetting_pct"] = disp["forgetting_pct"].apply(lambda x: f"{x:.2f}%")
    disp["ppl_a_after"] = disp["ppl_a_after"].apply(lambda x: f"{x:.2f}")
    disp["ppl_b_after"] = disp["ppl_b_after"].apply(lambda x: f"{x:.2f}")
    disp["lambada_after"] = disp["lambada_after"].apply(lambda x: f"{x:.3f}")
    disp["rep4_after"] = disp["rep4_after"].apply(lambda x: f"{x:.4f}")
    disp["drift_value_after"] = disp["drift_value_after"].apply(lambda x: f"{x:.4f}")
    disp["total_hours"] = disp["total_hours"].apply(lambda x: f"{x:.1f}h")
    disp["avg_tokens_per_sec"] = disp["avg_tokens_per_sec"].apply(lambda x: f"{x:.0f}")
    disp.columns = [
        "Run ID", "Model", "Method", "Seed",
        "Forgetting %", "PPL_A After", "PPL_B After",
        "LAMBADA", "Rep-4", "Drift (JS)",
        "Hours", "Tok/s",
    ]

    md = "# Table 1: All Core Results\n\n"
    md += f"**Runs**: {len(t)} completed non-smoke runs across "
    md += f"{t['model_family'].nunique()} model families  \n"
    md += f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n"
    md += disp.to_markdown(index=False)
    md += "\n"
    (out / "table1_all_results.md").write_text(md)
    print(f"  -> table1_all_results ({len(t)} runs)")


def table2_baseline_by_model(df: pd.DataFrame, out: Path) -> None:
    """Table 2: Baseline forgetting per model family (mean +/- std)."""
    baselines = df[df["method"] == "baseline"]
    metrics = [
        ("forgetting_pct", "Forgetting %", ".2f"),
        ("ppl_a_after", "PPL_A After", ".2f"),
        ("ppl_b_after", "PPL_B After", ".2f"),
        ("lambada_after", "LAMBADA", ".3f"),
        ("rep4_after", "Rep-4 After", ".4f"),
        ("total_hours", "Hours", ".1f"),
        ("avg_tokens_per_sec", "Tok/s", ".0f"),
    ]

    rows = []
    for family in MODEL_ORDER:
        fam_data = baselines[baselines["model_family"] == family]
        if fam_data.empty:
            continue
        row = {"Model": MODEL_LABELS[family], "N": len(fam_data)}
        for col, label, fmt in metrics:
            vals = fam_data[col].dropna()
            mean = vals.mean()
            std = vals.std() if len(vals) > 1 else 0.0
            row[label] = f"{mean:{fmt}} ± {std:{fmt}}"
        rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv(out / "table2_baseline_by_model.csv", index=False)

    md = "# Table 2: Baseline Forgetting by Model (RQ1)\n\n"
    md += "Mean ± std across seeds for each model family.\n\n"
    md += t.to_markdown(index=False)
    md += "\n"
    (out / "table2_baseline_by_model.md").write_text(md)
    print(f"  -> table2_baseline_by_model ({len(t)} models)")


def table3_method_vs_baseline(df: pd.DataFrame, out: Path) -> None:
    """Table 3: Method effectiveness — delta from baseline per model."""
    baselines = df[df["method"] == "baseline"]
    methods = df[df["method"] != "baseline"]

    delta_metrics = [
        ("forgetting_pct", "Δ Forgetting %", "+.2f", True),   # lower is better
        ("ppl_a_after", "Δ PPL_A", "+.2f", True),
        ("ppl_b_after", "Δ PPL_B", "+.2f", True),
        ("lambada_after", "Δ LAMBADA", "+.3f", False),  # higher is better
    ]

    rows = []
    for family in MODEL_ORDER:
        fam_baselines = baselines[baselines["model_family"] == family]
        if fam_baselines.empty:
            continue
        baseline_means = {col: fam_baselines[col].mean() for col, _, _, _ in delta_metrics}

        fam_methods = methods[methods["model_family"] == family]
        for _, run in fam_methods.iterrows():
            row = {
                "Model": MODEL_LABELS[family],
                "Method": METHOD_LABELS.get(run["method"], run["method"]),
                "Forgetting %": f"{run['forgetting_pct']:.2f}%",
            }
            for col, label, fmt, _ in delta_metrics:
                delta = run[col] - baseline_means[col]
                row[label] = f"{delta:{fmt}}"
            rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv(out / "table3_method_vs_baseline.csv", index=False)

    md = "# Table 3: Method Effectiveness vs Baseline (RQ2 / RQ4)\n\n"
    md += "Delta from per-model baseline mean. "
    md += "Negative Δ Forgetting = better retention.\n\n"
    md += t.to_markdown(index=False)
    md += "\n"
    (out / "table3_method_vs_baseline.md").write_text(md)
    print(f"  -> table3_method_vs_baseline ({len(t)} rows)")


def table4_method_rankings(df: pd.DataFrame, out: Path) -> None:
    """Table 4: Method rankings within each model family."""
    methods = df[df["method"] != "baseline"]

    rows = []
    for family in MODEL_ORDER:
        fam = methods[methods["model_family"] == family]
        if fam.empty:
            continue

        # Rank by forgetting (lower is better)
        ranked = fam.sort_values("forgetting_pct")
        for rank, (_, run) in enumerate(ranked.iterrows(), 1):
            rows.append({
                "Model": MODEL_LABELS[family],
                "Rank": rank,
                "Method": METHOD_LABELS.get(run["method"], run["method"]),
                "Forgetting %": f"{run['forgetting_pct']:.2f}%",
                "PPL_B After": f"{run['ppl_b_after']:.2f}",
                "LAMBADA": f"{run['lambada_after']:.3f}",
            })

    t = pd.DataFrame(rows)
    t.to_csv(out / "table4_method_rankings.csv", index=False)

    md = "# Table 4: Method Rankings by Forgetting (per model)\n\n"
    md += "Ranked from lowest to highest forgetting within each model family.\n\n"
    md += t.to_markdown(index=False)
    md += "\n"
    (out / "table4_method_rankings.md").write_text(md)
    print(f"  -> table4_method_rankings ({len(t)} rows)")


def table5_compute(df: pd.DataFrame, out: Path) -> None:
    """Table 5: Compute efficiency per model × method."""
    rows = []
    for family in MODEL_ORDER:
        for method in METHOD_ORDER:
            subset = df[(df["model_family"] == family) & (df["method"] == method)]
            if subset.empty:
                continue
            rows.append({
                "Model": MODEL_LABELS[family],
                "Method": METHOD_LABELS[method],
                "N": len(subset),
                "Avg Hours": f"{subset['total_hours'].mean():.1f}",
                "Avg Tok/s": f"{subset['avg_tokens_per_sec'].mean():.0f}",
                "Avg RAM (GB)": f"{subset['peak_ram_gb'].mean():.1f}"
                if "peak_ram_gb" in subset.columns and subset["peak_ram_gb"].notna().any()
                else "N/A",
            })

    t = pd.DataFrame(rows)
    t.to_csv(out / "table5_compute.csv", index=False)

    md = "# Table 5: Compute Efficiency\n\n"
    md += t.to_markdown(index=False)
    md += "\n"
    (out / "table5_compute.md").write_text(md)
    print(f"  -> table5_compute ({len(t)} rows)")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def fig1_baseline_forgetting(df: pd.DataFrame, out: Path) -> None:
    """Fig 1: Baseline forgetting across model families (RQ1)."""
    baselines = df[df["method"] == "baseline"]
    fig, ax = plt.subplots(figsize=(8, 5))

    families = [f for f in MODEL_ORDER if f in baselines["model_family"].values]
    x = np.arange(len(families))
    means, stds = [], []
    for fam in families:
        vals = baselines[baselines["model_family"] == fam]["forgetting_pct"]
        means.append(vals.mean())
        stds.append(vals.std() if len(vals) > 1 else 0.0)

    colors = [MODEL_COLORS[f] for f in families]
    bars = ax.bar(x, means, 0.6, yerr=stds, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=5)

    for bar, m, s in zip(bars, means, stds):
        label = f"{m:.1f}%"
        if s > 0:
            label += f" ± {s:.1f}"
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[f] for f in families])
    ax.set_ylabel("Forgetting (%)", fontweight="bold")
    ax.set_title("Baseline Catastrophic Forgetting by Model Family (RQ1)", fontweight="bold")
    ax.set_ylim(0, max(means) * 1.2)

    save_figure(fig, out / "fig1_baseline_forgetting")
    print("  -> fig1_baseline_forgetting")


def fig2_forgetting_by_method(df: pd.DataFrame, out: Path) -> None:
    """Fig 2: Forgetting by method, grouped by model (GPT-2 + Qwen3 + any with methods)."""
    families_with_methods = sorted(
        df[df["method"] != "baseline"]["model_family"].unique(),
        key=lambda f: MODEL_ORDER.index(f) if f in MODEL_ORDER else 99,
    )
    if not families_with_methods:
        print("  SKIP fig2: no method runs available")
        return

    methods_present = [m for m in METHOD_ORDER
                       if m in df[df["model_family"].isin(families_with_methods)]["method"].unique()]

    fig, ax = plt.subplots(figsize=(max(10, len(methods_present) * 2), 6))

    n_fam = len(families_with_methods)
    n_meth = len(methods_present)
    bar_width = 0.8 / n_fam
    x = np.arange(n_meth)

    baselines = df[df["method"] == "baseline"]

    for i, family in enumerate(families_with_methods):
        vals = []
        errs = []
        for method in methods_present:
            if method == "baseline":
                fam_bl = baselines[baselines["model_family"] == family]["forgetting_pct"]
                vals.append(fam_bl.mean() if len(fam_bl) > 0 else 0)
                errs.append(fam_bl.std() if len(fam_bl) > 1 else 0)
            else:
                subset = df[(df["model_family"] == family) & (df["method"] == method)]
                vals.append(subset["forgetting_pct"].values[0] if len(subset) > 0 else np.nan)
                errs.append(0)

        offset = (i - (n_fam - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, yerr=errs,
                      label=MODEL_LABELS[family], color=MODEL_COLORS[family],
                      edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods_present], rotation=15, ha="right")
    ax.set_ylabel("Forgetting (%)", fontweight="bold")
    ax.set_title("Forgetting by Method × Model (RQ2 / RQ4)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)

    save_figure(fig, out / "fig2_forgetting_by_method")
    print("  -> fig2_forgetting_by_method")


def fig3_ppl_tradeoff(df: pd.DataFrame, out: Path) -> None:
    """Fig 3: PPL_A vs PPL_B scatter — shape=model, color=method."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for _, run in df.iterrows():
        family = run["model_family"]
        method = run["method"]
        ax.scatter(
            run["ppl_a_after"], run["ppl_b_after"],
            c=METHOD_COLORS.get(method, "gray"),
            marker=MODEL_MARKERS.get(family, "o"),
            s=120, edgecolors="black", linewidths=0.8, zorder=5,
        )

    # Legend: methods (colors)
    method_handles = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
                      for m in METHOD_ORDER if m in df["method"].values]
    # Legend: models (markers)
    model_handles = [plt.Line2D([0], [0], marker=MODEL_MARKERS[f], color="gray",
                                markerfacecolor="gray", markersize=8,
                                label=MODEL_LABELS[f], linestyle="None")
                     for f in MODEL_ORDER if f in df["model_family"].values]

    leg1 = ax.legend(handles=method_handles, title="Method", loc="upper left",
                     fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model", loc="lower right",
              fontsize=8, title_fontsize=9)

    ax.set_xlabel("PPL on Domain A After (↓ better retention)", fontweight="bold")
    ax.set_ylabel("PPL on Domain B After (↓ better adaptation)", fontweight="bold")
    ax.set_title("PPL Tradeoff: Retention vs Adaptation (all models)", fontweight="bold")

    # Ideal corner
    ax.text(0.02, 0.02, "Ideal\n(low A, low B)", transform=ax.transAxes,
            fontsize=9, alpha=0.5, color="green")

    save_figure(fig, out / "fig3_ppl_tradeoff")
    print("  -> fig3_ppl_tradeoff")


def fig4_forgetting_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Fig 4: Model × Method heatmap of forgetting %."""
    families = [f for f in MODEL_ORDER if f in df["model_family"].values]
    methods = [m for m in METHOD_ORDER if m in df["method"].values]

    matrix = np.full((len(families), len(methods)), np.nan)
    for i, fam in enumerate(families):
        for j, meth in enumerate(methods):
            subset = df[(df["model_family"] == fam) & (df["method"] == meth)]
            if not subset.empty:
                matrix[i, j] = subset["forgetting_pct"].mean()

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), max(4, len(families) * 1.2)))

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Forgetting %", fontweight="bold")

    # Annotate cells
    for i in range(len(families)):
        for j in range(len(methods)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="gray")
            else:
                color = "white" if val > 60 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels([MODEL_LABELS[f] for f in families])
    ax.set_title("Forgetting % by Model × Method", fontweight="bold")

    save_figure(fig, out / "fig4_forgetting_heatmap")
    print("  -> fig4_forgetting_heatmap")


def fig5_method_delta(df: pd.DataFrame, out: Path) -> None:
    """Fig 5: Delta forgetting (method − baseline mean) grouped by model."""
    baselines = df[df["method"] == "baseline"]
    methods_df = df[df["method"] != "baseline"]

    families_with_methods = [f for f in MODEL_ORDER
                             if f in methods_df["model_family"].values]
    if not families_with_methods:
        print("  SKIP fig5: no method runs")
        return

    methods_present = [m for m in METHOD_ORDER
                       if m != "baseline" and m in methods_df["method"].values]

    fig, ax = plt.subplots(figsize=(max(9, len(methods_present) * 2.5), 6))

    n_fam = len(families_with_methods)
    bar_width = 0.8 / n_fam
    x = np.arange(len(methods_present))

    for i, family in enumerate(families_with_methods):
        bl_mean = baselines[baselines["model_family"] == family]["forgetting_pct"].mean()
        deltas = []
        for method in methods_present:
            subset = methods_df[(methods_df["model_family"] == family) &
                                (methods_df["method"] == method)]
            if subset.empty:
                deltas.append(np.nan)
            else:
                deltas.append(subset["forgetting_pct"].values[0] - bl_mean)

        offset = (i - (n_fam - 1) / 2) * bar_width
        ax.bar(x + offset, deltas, bar_width, label=MODEL_LABELS[family],
               color=MODEL_COLORS[family], edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods_present], rotation=15, ha="right")
    ax.set_ylabel("Δ Forgetting vs Baseline (%)", fontweight="bold")
    ax.set_title("Forgetting Reduction by Method (negative = better)", fontweight="bold")
    ax.legend(loc="upper right")

    save_figure(fig, out / "fig5_method_delta")
    print("  -> fig5_method_delta")


def fig6_lambada_retention(df: pd.DataFrame, out: Path) -> None:
    """Fig 6: LAMBADA before vs after, grouped by model."""
    families = [f for f in MODEL_ORDER if f in df["model_family"].values]

    fig, ax = plt.subplots(figsize=(max(10, len(families) * 3), 6))

    # Collect unique (model, method) pairs
    groups = []
    for fam in families:
        for method in METHOD_ORDER:
            subset = df[(df["model_family"] == fam) & (df["method"] == method)]
            if not subset.empty:
                groups.append((fam, method, subset))

    n = len(groups)
    x = np.arange(n)
    bar_w = 0.35

    before_vals = [g[2]["lambada_before"].mean() for g in groups]
    after_vals = [g[2]["lambada_after"].mean() for g in groups]

    ax.bar(x - bar_w / 2, before_vals, bar_w, label="Before (θ_A)",
           color="#b2df8a", edgecolor="black", linewidth=0.5)
    ax.bar(x + bar_w / 2, after_vals, bar_w, label="After (θ_AB)",
           color="#33a02c", edgecolor="black", linewidth=0.5)

    labels = [f"{MODEL_LABELS[g[0]]}\n{METHOD_LABELS[g[1]]}" for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("LAMBADA Accuracy (↑ better)", fontweight="bold")
    ax.set_title("General Language Ability: Before vs After CPT", fontweight="bold")
    ax.legend()

    save_figure(fig, out / "fig6_lambada_retention")
    print("  -> fig6_lambada_retention")


def fig7_compute_scaling(df: pd.DataFrame, out: Path) -> None:
    """Fig 7: Throughput vs model size, colored by method."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, run in df.iterrows():
        family = run["model_family"]
        method = run["method"]
        ax.scatter(
            run["model_params_m"], run["avg_tokens_per_sec"],
            c=METHOD_COLORS.get(method, "gray"),
            marker=MODEL_MARKERS.get(family, "o"),
            s=100, edgecolors="black", linewidths=0.6, zorder=5,
        )

    # Trend line (unique model sizes → mean throughput)
    for method in df["method"].unique():
        subset = df[df["method"] == method]
        if subset["model_params_m"].nunique() >= 2:
            grouped = subset.groupby("model_params_m")["avg_tokens_per_sec"].mean()
            ax.plot(grouped.index, grouped.values, "--",
                    color=METHOD_COLORS.get(method, "gray"), alpha=0.4, linewidth=1)

    method_handles = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
                      for m in METHOD_ORDER if m in df["method"].values]
    ax.legend(handles=method_handles, loc="upper right", fontsize=8)

    ax.set_xlabel("Model Parameters (M)", fontweight="bold")
    ax.set_ylabel("Tokens / sec (↑ better)", fontweight="bold")
    ax.set_title("Compute Throughput vs Model Scale", fontweight="bold")

    save_figure(fig, out / "fig7_compute_scaling")
    print("  -> fig7_compute_scaling")


def fig8_dashboard(df: pd.DataFrame, out: Path) -> None:
    """Fig 8: Multi-panel dashboard — 2×3 grid of key metrics by model."""
    families = [f for f in MODEL_ORDER if f in df["model_family"].values]
    baselines = df[df["method"] == "baseline"]

    panels = [
        ("forgetting_pct", "Forgetting %", True),
        ("ppl_a_after", "PPL on Domain A", True),
        ("ppl_b_after", "PPL on Domain B", True),
        ("lambada_after", "LAMBADA Accuracy", False),
        ("avg_tokens_per_sec", "Throughput (tok/s)", False),
        ("total_hours", "Total Time (hours)", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (metric, title, lower_better) in zip(axes, panels):
        direction = "↓" if lower_better else "↑"
        ax.set_title(f"{title}\n({direction} better)", fontweight="bold", fontsize=10)

        x_pos = np.arange(len(families))

        for fam_idx, family in enumerate(families):
            fam_data = df[df["model_family"] == family]
            methods_in_fam = [m for m in METHOD_ORDER if m in fam_data["method"].values]

            n_m = len(methods_in_fam)
            bar_w = 0.8 / max(n_m, 1)

            for m_idx, method in enumerate(methods_in_fam):
                subset = fam_data[fam_data["method"] == method]
                val = subset[metric].mean()
                err = subset[metric].std() if len(subset) > 1 else 0

                offset = (m_idx - (n_m - 1) / 2) * bar_w
                ax.bar(fam_idx + offset, val, bar_w, yerr=err if err > 0 else None,
                       color=METHOD_COLORS.get(method, "gray"),
                       edgecolor="black", linewidth=0.3, capsize=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODEL_LABELS[f].split(" (")[0] for f in families],
                           fontsize=8, rotation=30, ha="right")

    # Shared legend
    method_handles = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
                      for m in METHOD_ORDER if m in df["method"].values]
    fig.legend(handles=method_handles, loc="lower center", ncol=len(method_handles),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Comprehensive Multi-Model Dashboard", fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()

    save_figure(fig, out / "fig8_dashboard")
    print("  -> fig8_dashboard")


# ═══════════════════════════════════════════════════════════════════════════
# EXTENDED ANALYSES — added 2026-05-09 to fill out the analysis dir
# ═══════════════════════════════════════════════════════════════════════════

def table6_method_consistency(df: pd.DataFrame, out: Path) -> None:
    """T6: Method consistency across models — std/CV of forgetting per method.

    Directly supports the "no method wins everywhere" framing: a method that
    works on every model has low std and low CV; a method that fails
    unpredictably has high values.
    """
    methods = [m for m in METHOD_ORDER if m != "baseline"]
    rows = []
    for method in methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        f = sub["forgetting_pct"]
        n = len(f)
        mean = f.mean()
        std = f.std() if n > 1 else 0.0
        cv = (std / mean * 100) if mean > 0 else 0.0
        rows.append({
            "Method": METHOD_LABELS[method],
            "N models": int(sub["model_family"].nunique()),
            "Mean forget %": f"{mean:.2f}",
            "Std forget %": f"{std:.2f}",
            "CV %": f"{cv:.1f}",
            "Min forget %": f"{f.min():.2f}",
            "Max forget %": f"{f.max():.2f}",
            "Range": f"{f.max() - f.min():.2f}",
        })
    rows.sort(key=lambda r: float(r["CV %"]))
    t = pd.DataFrame(rows)
    t.to_csv(out / "table6_method_consistency.csv", index=False)

    md = "# Table 6: Method Consistency Across Models\n\n"
    md += "Lower CV = method behaves consistently across architectures. "
    md += "Higher CV = effectiveness depends on the model.\n\n"
    md += t.to_markdown(index=False)
    md += "\n"
    (out / "table6_method_consistency.md").write_text(md)
    print(f"  -> table6_method_consistency ({len(t)} methods)")


def table7_generation_drift(df: pd.DataFrame, out: Path) -> None:
    """T7: Generation drift profile — Rep4/Rep8/JS/vocab before vs after."""
    rows = []
    for family in MODEL_ORDER:
        for method in METHOD_ORDER:
            sub = df[(df["model_family"] == family) & (df["method"] == method)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({
                "Model": MODEL_LABELS[family],
                "Method": METHOD_LABELS[method],
                "Rep4 before": f"{r.get('rep4_before', float('nan')):.3f}",
                "Rep4 after": f"{r.get('rep4_after', float('nan')):.3f}",
                "Δ Rep4": f"{r.get('rep4_after', 0) - r.get('rep4_before', 0):+.3f}",
                "Rep8 before": f"{r.get('rep8_before', float('nan')):.3f}",
                "Rep8 after": f"{r.get('rep8_after', float('nan')):.3f}",
                "JS before": f"{r.get('drift_value_before', float('nan')):.3f}",
                "JS after": f"{r.get('drift_value_after', float('nan')):.3f}",
                "Δ JS": f"{r.get('drift_value_after', 0) - r.get('drift_value_before', 0):+.3f}",
                "Vocab Δ": f"{r.get('vocab_overlap_after', 0) - r.get('vocab_overlap_before', 0):+.3f}",
            })
    t = pd.DataFrame(rows)
    t.to_csv(out / "table7_generation_drift.csv", index=False)

    md = "# Table 7: Generation Drift Profile (before vs after CPT)\n\n"
    md += "Δ Rep4 > 0 means generation became more repetitive; "
    md += "Δ JS > 0 means distribution drifted further from the reference set.\n\n"
    md += t.to_markdown(index=False)
    md += "\n"
    (out / "table7_generation_drift.md").write_text(md)
    print(f"  -> table7_generation_drift ({len(t)} rows)")


def table8_adaptation_ranking(df: pd.DataFrame, out: Path) -> None:
    """T8: Adaptation ranking by PPL_B (Domain B perplexity, lower=better).

    Complements the forgetting ranking: a method that prevents forgetting
    but fails to learn Domain B is not actually useful.
    """
    rows = []
    for family in MODEL_ORDER:
        sub = df[df["model_family"] == family]
        if sub.empty:
            continue
        sub = sub.sort_values("ppl_b_after")
        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            rows.append({
                "Model": MODEL_LABELS[family],
                "Rank (PPL_B)": rank,
                "Method": METHOD_LABELS.get(r["method"], r["method"]),
                "PPL_B after": f"{r['ppl_b_after']:.2f}",
                "PPL_A after": f"{r['ppl_a_after']:.2f}",
                "Forget %": f"{r['forgetting_pct']:.2f}",
            })
    t = pd.DataFrame(rows)
    t.to_csv(out / "table8_adaptation_ranking.csv", index=False)

    md = "# Table 8: Adaptation Ranking by PPL_B (lower is better)\n\n"
    md += "A method that minimises forgetting but doesn't learn Domain B "
    md += "is not actually useful. This table ranks by Domain B PPL after CPT.\n\n"
    md += t.to_markdown(index=False)
    md += "\n"
    (out / "table8_adaptation_ranking.md").write_text(md)
    print(f"  -> table8_adaptation_ranking ({len(t)} rows)")


def _is_pareto_optimal(points: np.ndarray) -> np.ndarray:
    """Return mask of non-dominated points where lower is better on all axes."""
    n = len(points)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # i is dominated if any other point j has all coords <= i and at least one <
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_efficient[i] = False
                break
    return is_efficient


def table9_pareto_frontier(df: pd.DataFrame, out: Path) -> None:
    """T9: Pareto frontier per model on (forgetting, wall-time)."""
    rows = []
    for family in MODEL_ORDER:
        sub = df[df["model_family"] == family]
        if sub.empty or len(sub) < 2:
            continue
        pts = sub[["forgetting_pct", "total_hours"]].to_numpy()
        mask = _is_pareto_optimal(pts)
        for (_, r), eff in zip(sub.iterrows(), mask):
            rows.append({
                "Model": MODEL_LABELS[family],
                "Method": METHOD_LABELS.get(r["method"], r["method"]),
                "Forget %": f"{r['forgetting_pct']:.2f}",
                "Hours": f"{r['total_hours']:.2f}",
                "PPL_B after": f"{r['ppl_b_after']:.2f}",
                "Pareto-optimal": "✓" if eff else "",
            })
    t = pd.DataFrame(rows)
    t.to_csv(out / "table9_pareto_frontier.csv", index=False)

    md = "# Table 9: Pareto Frontier (forgetting × wall-time)\n\n"
    md += "Pareto-optimal: no other run on the same model achieves both "
    md += "less forgetting AND less compute time. These are the rational "
    md += "method choices for a practitioner who cares about both.\n\n"
    md += t.to_markdown(index=False)
    md += "\n"
    (out / "table9_pareto_frontier.md").write_text(md)
    print(f"  -> table9_pareto_frontier ({len(t)} rows)")


def table10_rl_internals(df: pd.DataFrame, out: Path) -> None:
    """T10: RL method internals — bandit and RMGS internal state per run."""
    rl_methods = ["bandit_replay", "rmgs"]
    sub = df[df["method"].isin(rl_methods)].copy()
    if sub.empty:
        print("  SKIP table10: no RL method runs found")
        return

    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "Model": MODEL_LABELS.get(r["model_family"], r["model_family"]),
            "Method": METHOD_LABELS.get(r["method"], r["method"]),
            "Forget %": f"{r['forgetting_pct']:.2f}",
            "Mean replay rate": f"{r['rl_mean_replay_rate']:.3f}"
                if "rl_mean_replay_rate" in r and pd.notna(r.get("rl_mean_replay_rate")) else "—",
            "Replay rate std": f"{r['rl_replay_rate_std']:.3f}"
                if "rl_replay_rate_std" in r and pd.notna(r.get("rl_replay_rate_std")) else "—",
            "Mean grad scale": f"{r['rl_mean_gradient_scale']:.3f}"
                if "rl_mean_gradient_scale" in r and pd.notna(r.get("rl_mean_gradient_scale")) else "—",
            "Grad scale std": f"{r['rl_scale_std']:.3f}"
                if "rl_scale_std" in r and pd.notna(r.get("rl_scale_std")) else "—",
            "N evaluations": int(r["rl_n_evaluations"])
                if "rl_n_evaluations" in r and pd.notna(r.get("rl_n_evaluations")) else "—",
        })
    t = pd.DataFrame(rows)
    t.to_csv(out / "table10_rl_internals.csv", index=False)

    md = "# Table 10: RL Method Internal State\n\n"
    md += "Headline patterns to look for:\n"
    md += "- bandit_replay mean replay rate consistent across models?\n"
    md += "- RMGS mean grad scale far from 1.0 (active throttling) or near 1.0 (rare throttling)?\n\n"
    md += t.to_markdown(index=False)
    md += "\n"
    (out / "table10_rl_internals.md").write_text(md)
    print(f"  -> table10_rl_internals ({len(t)} rows)")


# T11 (anomaly catalog) removed 2026-05-09: after retiring high_rep4 the
# table had no rows. The anomalies column remains in summary_table.csv for
# any future genuine flag (oom_recovery, nan_loss, etc.).


# --------------------------------------------------------------------------
# New figures
# --------------------------------------------------------------------------

def fig9_method_consistency(df: pd.DataFrame, out: Path) -> None:
    """F9: Method consistency boxplot — distribution of forgetting per method."""
    methods = [m for m in METHOD_ORDER if m != "baseline" and m in df["method"].values]
    if not methods:
        print("  SKIP fig9: no method runs")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    data, labels, colors = [], [], []
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        data.append(sub["forgetting_pct"].values)
        labels.append(METHOD_LABELS[m])
        colors.append(METHOD_COLORS[m])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)

    # Overlay individual points colored by model
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        for _, r in sub.iterrows():
            ax.scatter(i + 1, r["forgetting_pct"],
                       color=MODEL_COLORS.get(r["model_family"], "gray"),
                       edgecolor="black", s=60, zorder=3)

    # Legend for model colors
    legend_handles = [mpatches.Patch(color=MODEL_COLORS[f], label=MODEL_LABELS[f])
                      for f in MODEL_ORDER if f in df["model_family"].values]
    ax.legend(handles=legend_handles, loc="upper right", title="Model")

    ax.set_ylabel("Forgetting (%)", fontweight="bold")
    ax.set_title("Forgetting Distribution per Method (across models)", fontweight="bold")
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=15, ha="right")

    save_figure(fig, out / "fig9_method_consistency")
    print("  -> fig9_method_consistency")


def fig10_pareto_frontier(df: pd.DataFrame, out: Path) -> None:
    """F10: Pareto frontier per model on (forgetting, wall-time)."""
    families = [f for f in MODEL_ORDER if f in df["model_family"].values]
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = df[df["model_family"] == family].copy()
        pts = sub[["forgetting_pct", "total_hours"]].to_numpy()
        mask = _is_pareto_optimal(pts)
        # Plot all points
        for (_, r), eff in zip(sub.iterrows(), mask):
            ax.scatter(r["total_hours"], r["forgetting_pct"],
                       color=METHOD_COLORS.get(r["method"], "gray"),
                       s=130 if eff else 60,
                       edgecolor="black",
                       linewidth=1.5 if eff else 0.5,
                       zorder=3 if eff else 2)
            ax.annotate(METHOD_LABELS.get(r["method"], r["method"]),
                        (r["total_hours"], r["forgetting_pct"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)

        # Connect Pareto-optimal points with a line
        if mask.any():
            efficient = sub[mask].sort_values("total_hours")
            ax.plot(efficient["total_hours"], efficient["forgetting_pct"],
                    "k--", alpha=0.4, linewidth=1.2, zorder=1)
        ax.set_title(MODEL_LABELS[family], fontweight="bold")
        ax.set_xlabel("Wall time (hours)")
        ax.set_ylabel("Forgetting (%)")
        ax.set_ylim(bottom=0)

    fig.suptitle("Pareto Frontier: Forgetting × Compute (per model)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, out / "fig10_pareto_frontier")
    print("  -> fig10_pareto_frontier")


def fig11_rep4_change_heatmap(df: pd.DataFrame, out: Path) -> None:
    """F11: Δ Rep4 (post-pre) heatmap — model × method."""
    methods = [m for m in METHOD_ORDER if m in df["method"].values]
    families = [f for f in MODEL_ORDER if f in df["model_family"].values]
    if not methods or not families:
        return
    grid = np.full((len(families), len(methods)), np.nan)
    for i, fam in enumerate(families):
        for j, m in enumerate(methods):
            sub = df[(df["model_family"] == fam) & (df["method"] == m)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            if pd.notna(r.get("rep4_before")) and pd.notna(r.get("rep4_after")):
                grid[i, j] = r["rep4_after"] - r["rep4_before"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    cmap = plt.get_cmap("RdYlGn_r")  # red = worse (more repetition)
    vlim = max(0.001, np.nanmax(np.abs(grid)))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([MODEL_LABELS[f] for f in families])
    for i in range(len(families)):
        for j in range(len(methods)):
            v = grid[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ Rep-4 (post − pre)\n(positive = more repetitive)")
    ax.set_title("Generation Repetition Change after CPT", fontweight="bold")
    plt.tight_layout()
    save_figure(fig, out / "fig11_rep4_change_heatmap")
    print("  -> fig11_rep4_change_heatmap")


def fig12_retention_adaptation_pareto(df: pd.DataFrame, out: Path) -> None:
    """F12: PPL_A (retention) vs PPL_B (adaptation) Pareto, per-model panels."""
    families = [f for f in MODEL_ORDER if f in df["model_family"].values]
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8))
    if n == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = df[df["model_family"] == family].copy()
        if sub.empty:
            continue
        pts = sub[["ppl_a_after", "ppl_b_after"]].to_numpy()
        mask = _is_pareto_optimal(pts)
        for (_, r), eff in zip(sub.iterrows(), mask):
            ax.scatter(r["ppl_a_after"], r["ppl_b_after"],
                       color=METHOD_COLORS.get(r["method"], "gray"),
                       s=130 if eff else 60,
                       edgecolor="black",
                       linewidth=1.5 if eff else 0.5)
            ax.annotate(METHOD_LABELS.get(r["method"], r["method"]),
                        (r["ppl_a_after"], r["ppl_b_after"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_title(MODEL_LABELS[family], fontweight="bold")
        ax.set_xlabel("PPL_A after (retention; lower = better)")
        ax.set_ylabel("PPL_B after (adaptation; lower = better)")
    fig.suptitle("Retention × Adaptation Pareto (per model)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, out / "fig12_retention_adaptation_pareto")
    print("  -> fig12_retention_adaptation_pareto")


def fig13_rl_internals(df: pd.DataFrame, out: Path) -> None:
    """F13: RL internals — bandit replay rate + RMGS grad scale per model."""
    bandit = df[df["method"] == "bandit_replay"]
    rmgs = df[df["method"] == "rmgs"]
    if bandit.empty and rmgs.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: bandit mean replay rate by model
    if not bandit.empty and "rl_mean_replay_rate" in bandit.columns:
        b = bandit.dropna(subset=["rl_mean_replay_rate"])
        x = np.arange(len(b))
        rates = b["rl_mean_replay_rate"].to_numpy()
        stds = b.get("rl_replay_rate_std", pd.Series([0] * len(b))).to_numpy()
        colors = [MODEL_COLORS.get(m, "gray") for m in b["model_family"]]
        ax1.bar(x, rates, yerr=stds, color=colors, edgecolor="black", capsize=4)
        for xi, r in zip(x, rates):
            ax1.text(xi, r + 0.01, f"{r:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax1.axhline(0.25, color="black", linestyle=":", linewidth=1, label="static replay25 default")
        ax1.set_xticks(x)
        ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in b["model_family"]],
                            rotation=15, ha="right")
        ax1.set_ylabel("Mean replay rate (with std)")
        ax1.set_title("bandit_replay: learned replay rate per model", fontweight="bold")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.set_ylim(0, max(0.6, rates.max() + 0.1))

    # Right: RMGS mean grad scale by model
    if not rmgs.empty and "rl_mean_gradient_scale" in rmgs.columns:
        r_ = rmgs.dropna(subset=["rl_mean_gradient_scale"])
        x = np.arange(len(r_))
        scales = r_["rl_mean_gradient_scale"].to_numpy()
        stds = r_.get("rl_scale_std", pd.Series([0] * len(r_))).to_numpy()
        colors = [MODEL_COLORS.get(m, "gray") for m in r_["model_family"]]
        ax2.bar(x, scales, yerr=stds, color=colors, edgecolor="black", capsize=4)
        for xi, s in zip(x, scales):
            ax2.text(xi, s + 0.005, f"{s:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax2.axhline(1.0, color="black", linestyle=":", linewidth=1, label="no throttling")
        ax2.set_xticks(x)
        ax2.set_xticklabels([MODEL_LABELS.get(m, m) for m in r_["model_family"]],
                            rotation=15, ha="right")
        ax2.set_ylabel("Mean gradient scale (with std)")
        ax2.set_title("RMGS: mean gradient scale per model", fontweight="bold")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.set_ylim(0.5, 1.05)

    plt.tight_layout()
    save_figure(fig, out / "fig13_rl_internals")
    print("  -> fig13_rl_internals")


def fig14_method_radar(df: pd.DataFrame, out: Path) -> None:
    """F14: Per-model radar of method signatures across 5 dimensions.

    Axes: forgetting↓, ppl_b↓, lambada↑, rep4↓, throughput↑.
    All normalized to [0, 1] within model. Each method = one polygon.
    """
    families = [f for f in MODEL_ORDER if f in df["model_family"].values]
    metrics = [
        ("forgetting_pct", "Forget↓", True),
        ("ppl_b_after", "PPL_B↓", True),
        ("lambada_after", "LAMBADA↑", False),
        ("rep4_after", "Rep4↓", True),
        ("avg_tokens_per_sec", "Tok/s↑", False),
    ]
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5),
                             subplot_kw=dict(polar=True))
    if n == 1:
        axes = [axes]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for ax, family in zip(axes, families):
        sub = df[df["model_family"] == family]
        if sub.empty:
            continue
        # Normalize each metric within model to [0, 1] where 1 = good
        normed = {}
        for col, _label, lower_is_better in metrics:
            vals = sub[col].to_numpy(dtype=float)
            if np.isnan(vals).all():
                normed[col] = np.zeros_like(vals)
                continue
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax - vmin < 1e-9:
                normed[col] = np.ones_like(vals) * 0.5
            else:
                z = (vals - vmin) / (vmax - vmin)
                normed[col] = 1.0 - z if lower_is_better else z

        for i, (_, r) in enumerate(sub.iterrows()):
            vals = [normed[col][i] for col, _, _ in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, color=METHOD_COLORS.get(r["method"], "gray"),
                    linewidth=2, label=METHOD_LABELS.get(r["method"], r["method"]))
            ax.fill(angles, vals, color=METHOD_COLORS.get(r["method"], "gray"),
                    alpha=0.10)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
        ax.set_yticklabels([])
        ax.set_title(MODEL_LABELS[family], fontweight="bold", pad=18)

    # One shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)),
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle("Method Signatures per Model (radar; outer = better)",
                 fontweight="bold", y=1.04)
    plt.tight_layout()
    save_figure(fig, out / "fig14_method_radar")
    print("  -> fig14_method_radar")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MULTI-MODEL ANALYSIS ASSETS GENERATOR")
    print("=" * 70)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Generated: {ts}\n")

    root = get_project_root()
    tables_dir = root / "experiments" / "analysis" / "tables"
    figures_dir = root / "experiments" / "analysis" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading runs from metrics.json files...")
    all_df = load_all_runs(root)
    if all_df.empty:
        print("ERROR: No completed runs found.")
        sys.exit(1)

    core = filter_core(all_df)
    print(f"  Core runs (non-smoke): {len(core)}")
    print(f"  Models: {sorted(core['model_family'].unique())}")
    print(f"  Methods: {sorted(core['method'].unique())}")

    # Regenerate summary_table.csv
    summary_path = root / "experiments" / "summary_table.csv"
    core.to_csv(summary_path, index=False)
    print(f"\n  Wrote {summary_path} ({len(core)} rows)")

    # Tables
    print("\nGenerating tables...")
    table1_all_results(core, tables_dir)
    table2_baseline_by_model(core, tables_dir)
    table3_method_vs_baseline(core, tables_dir)
    table4_method_rankings(core, tables_dir)
    table5_compute(core, tables_dir)
    table6_method_consistency(core, tables_dir)
    table7_generation_drift(core, tables_dir)
    table8_adaptation_ranking(core, tables_dir)
    table9_pareto_frontier(core, tables_dir)
    table10_rl_internals(core, tables_dir)

    # Figures
    print("\nGenerating figures...")
    fig1_baseline_forgetting(core, figures_dir)
    fig2_forgetting_by_method(core, figures_dir)
    fig3_ppl_tradeoff(core, figures_dir)
    fig4_forgetting_heatmap(core, figures_dir)
    fig5_method_delta(core, figures_dir)
    fig6_lambada_retention(core, figures_dir)
    fig7_compute_scaling(core, figures_dir)
    fig8_dashboard(core, figures_dir)
    fig9_method_consistency(core, figures_dir)
    fig10_pareto_frontier(core, figures_dir)
    fig11_rep4_change_heatmap(core, figures_dir)
    fig12_retention_adaptation_pareto(core, figures_dir)
    fig13_rl_internals(core, figures_dir)
    fig14_method_radar(core, figures_dir)

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    n_tables = len(list(tables_dir.glob("*.csv")))
    n_figs = len(list(figures_dir.glob("*.png")))
    print(f"  Tables: {n_tables} (CSV + Markdown)")
    print(f"  Figures: {n_figs} (PNG + PDF)")
    print(f"  Summary table: {summary_path}")


if __name__ == "__main__":
    main()
