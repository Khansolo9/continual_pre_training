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

        row["anomalies"] = ",".join(data.get("anomalies", []))
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
