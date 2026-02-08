#!/usr/bin/env python3
"""
Analysis Assets Generator for Continual Pretraining Experiments

Generates publication-quality tables and figures from completed experiment runs.

Usage:
    cd continual_pre_training
    source .cpt-env/bin/activate
    python experiments/analysis/scripts/make_analysis_assets.py

Outputs:
    experiments/analysis/tables/*.csv, *.md
    experiments/analysis/figures/*.png, *.pdf

Author: Claude Code (automated analysis)
Generated: 2026-02-07
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Color palette for methods (consistent across all figures)
METHOD_COLORS = {
    'baseline': '#4C72B0',  # Steel blue
    'replay25': '#55A868',  # Green
    'mer25': '#C44E52',     # Red
    'ewc': '#8172B3',       # Purple
}

METHOD_LABELS = {
    'baseline': 'Baseline (no mitigation)',
    'replay25': 'Replay-25%',
    'mer25': 'MER-lite',
    'ewc': 'EWC',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def get_project_root() -> Path:
    """Get project root directory."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent.parent


def load_summary_table(project_root: Path) -> pd.DataFrame:
    """Load the summary table CSV."""
    path = project_root / "experiments" / "summary_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Summary table not found: {path}")

    df = pd.read_csv(path)
    return df


def load_run_registry(project_root: Path) -> pd.DataFrame:
    """Load the run registry CSV."""
    path = project_root / "experiments" / "run_registry.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run registry not found: {path}")

    df = pd.read_csv(path)
    return df


def filter_mpp_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to MPP core runs (exclude smoke runs)."""
    # Exclude runs where research_question is SMOKE or run_id starts with smoke_
    mask = ~(
        (df['research_question'].str.upper() == 'SMOKE') |
        (df['run_id'].str.lower().str.startswith('smoke_'))
    )
    return df[mask].copy()


def get_baseline_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Get baseline runs only."""
    return df[df['method'] == 'baseline'].copy()


def get_method_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Get non-baseline method runs."""
    return df[df['method'] != 'baseline'].copy()


def compute_baseline_stats(baseline_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute mean and std for baseline runs."""
    stats = {}
    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        values = baseline_df[col].dropna()
        if len(values) > 0:
            stats[col] = {
                'mean': values.mean(),
                'std': values.std() if len(values) > 1 else 0.0,
                'min': values.min(),
                'max': values.max(),
                'n': len(values),
            }

    return stats


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_primary_results_table(df: pd.DataFrame, baseline_stats: Dict) -> Tuple[pd.DataFrame, str]:
    """Generate the primary results table."""

    # Select columns for the table
    cols = [
        'run_id', 'method', 'seed', 'forgetting_pct',
        'ppl_a_init', 'ppl_a_before', 'ppl_a_after',
        'ppl_b_init', 'ppl_b_after',
        'lambada_before', 'lambada_after',
        'rep4_before', 'rep4_after',
        'rep8_before', 'rep8_after',
        'drift_value_before', 'drift_value_after',
        'vocab_overlap_before', 'vocab_overlap_after',
        'total_hours', 'avg_tokens_per_sec'
    ]

    table_df = df[cols].copy()

    # Format for display
    display_df = table_df.copy()
    display_df['forgetting_pct'] = display_df['forgetting_pct'].apply(lambda x: f"{x:.2f}%")
    display_df['ppl_a_init'] = display_df['ppl_a_init'].apply(lambda x: f"{x:.2f}")
    display_df['ppl_a_before'] = display_df['ppl_a_before'].apply(lambda x: f"{x:.2f}")
    display_df['ppl_a_after'] = display_df['ppl_a_after'].apply(lambda x: f"{x:.2f}")
    display_df['ppl_b_init'] = display_df['ppl_b_init'].apply(lambda x: f"{x:.2f}")
    display_df['ppl_b_after'] = display_df['ppl_b_after'].apply(lambda x: f"{x:.2f}")
    display_df['lambada_before'] = display_df['lambada_before'].apply(lambda x: f"{x:.3f}")
    display_df['lambada_after'] = display_df['lambada_after'].apply(lambda x: f"{x:.3f}")
    display_df['rep4_before'] = display_df['rep4_before'].apply(lambda x: f"{x:.4f}")
    display_df['rep4_after'] = display_df['rep4_after'].apply(lambda x: f"{x:.4f}")
    display_df['rep8_before'] = display_df['rep8_before'].apply(lambda x: f"{x:.4f}")
    display_df['rep8_after'] = display_df['rep8_after'].apply(lambda x: f"{x:.4f}")
    display_df['drift_value_before'] = display_df['drift_value_before'].apply(lambda x: f"{x:.4f}")
    display_df['drift_value_after'] = display_df['drift_value_after'].apply(lambda x: f"{x:.4f}")
    display_df['vocab_overlap_before'] = display_df['vocab_overlap_before'].apply(lambda x: f"{x:.4f}")
    display_df['vocab_overlap_after'] = display_df['vocab_overlap_after'].apply(lambda x: f"{x:.4f}")
    display_df['total_hours'] = display_df['total_hours'].apply(lambda x: f"{x:.2f}h")
    display_df['avg_tokens_per_sec'] = display_df['avg_tokens_per_sec'].apply(lambda x: f"{x:.0f}")

    # Rename columns for display
    display_df.columns = [
        'Run ID', 'Method', 'Seed', 'Forgetting %',
        'PPL_A Init', 'PPL_A Before', 'PPL_A After',
        'PPL_B Init', 'PPL_B After',
        'LAMBADA Before', 'LAMBADA After',
        'Rep-4 Before', 'Rep-4 After',
        'Rep-8 Before', 'Rep-8 After',
        'Drift Before', 'Drift After',
        'Vocab Overlap Before', 'Vocab Overlap After',
        'Total Hours', 'Tokens/sec'
    ]

    # Generate markdown
    md = "# Primary Results Table\n\n"
    md += "**Domain A**: wikitext-103 (10M tokens)  \n"
    md += "**Domain B**: arxiv_abstracts (10M tokens)  \n"
    md += f"**Runs included**: {len(df)} MPP core runs (smoke runs excluded)  \n"
    md += f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n"
    md += display_df.to_markdown(index=False)
    md += "\n\n**Notes**:\n"
    md += "- `high_rep4` anomaly flagged on all runs (expected GPT-2 behavior)\n"
    md += "- `peak_vram_gb` is N/A on MPS (Mac); RAM tracking used instead\n"
    md += "- Rep-n metrics computed with sampling (temperature=0.7, top_p=0.9)\n"

    return table_df, md


def generate_delta_table(df: pd.DataFrame, baseline_stats: Dict) -> Tuple[pd.DataFrame, str]:
    """Generate delta table vs baseline mean."""

    method_runs = get_method_runs(df)

    metrics = [
        ('forgetting_pct', 'Forgetting %', '{:+.2f}%'),
        ('ppl_a_after', 'PPL_A After', '{:+.2f}'),
        ('ppl_b_after', 'PPL_B After', '{:+.2f}'),
        ('lambada_after', 'LAMBADA After', '{:+.3f}'),
        ('rep4_after', 'Rep-4 After', '{:+.4f}'),
        ('rep8_after', 'Rep-8 After', '{:+.4f}'),
        ('drift_value_after', 'Drift After', '{:+.4f}'),
        ('total_hours', 'Total Hours', '{:+.2f}h'),
        ('avg_tokens_per_sec', 'Tokens/sec', '{:+.0f}'),
    ]

    rows = []
    for _, run in method_runs.iterrows():
        row = {
            'Method': run['method'],
            'Seed': run['seed'],
        }
        for metric, display_name, fmt in metrics:
            if metric in baseline_stats:
                baseline_mean = baseline_stats[metric]['mean']
                value = run[metric]
                delta = value - baseline_mean
                row[f'Δ {display_name}'] = fmt.format(delta)
        rows.append(row)

    delta_df = pd.DataFrame(rows)

    # Generate markdown
    md = "# Delta Table vs Baseline Mean\n\n"
    md += f"**Baseline mean computed from**: pilot_baseline_s0, rq1_baseline_s42, rq1_baseline_s123 (n=3)\n"
    md += f"**Baseline Forgetting**: {baseline_stats['forgetting_pct']['mean']:.2f}% ± {baseline_stats['forgetting_pct']['std']:.2f}%\n\n"
    md += "*Negative delta = improvement for lower-is-better metrics (forgetting, PPL, rep, drift)*  \n"
    md += "*Positive delta = improvement for higher-is-better metrics (LAMBADA, tokens/sec)*\n\n"
    md += delta_df.to_markdown(index=False)

    return delta_df, md


def generate_ranking_table(df: pd.DataFrame, baseline_stats: Dict) -> Tuple[pd.DataFrame, str]:
    """Generate ranking table."""

    method_runs = get_method_runs(df)
    baseline_mean_forgetting = baseline_stats['forgetting_pct']['mean']

    rankings = []

    # Best retention (lowest forgetting)
    sorted_by_forgetting = method_runs.sort_values('forgetting_pct')
    for rank, (_, run) in enumerate(sorted_by_forgetting.iterrows(), 1):
        rankings.append({
            'Category': 'Best Retention (Lowest Forgetting)',
            'Rank': rank,
            'Method': run['method'],
            'Value': f"{run['forgetting_pct']:.2f}%",
            'vs Baseline': f"{run['forgetting_pct'] - baseline_mean_forgetting:+.2f}%",
        })

    # Best B adaptation (lowest PPL_B after)
    sorted_by_ppl_b = method_runs.sort_values('ppl_b_after')
    for rank, (_, run) in enumerate(sorted_by_ppl_b.iterrows(), 1):
        rankings.append({
            'Category': 'Best B Adaptation (Lowest PPL_B)',
            'Rank': rank,
            'Method': run['method'],
            'Value': f"{run['ppl_b_after']:.2f}",
            'vs Baseline': f"{run['ppl_b_after'] - baseline_stats['ppl_b_after']['mean']:+.2f}",
        })

    # Best general ability (highest LAMBADA)
    sorted_by_lambada = method_runs.sort_values('lambada_after', ascending=False)
    for rank, (_, run) in enumerate(sorted_by_lambada.iterrows(), 1):
        rankings.append({
            'Category': 'Best General Ability (Highest LAMBADA)',
            'Rank': rank,
            'Method': run['method'],
            'Value': f"{run['lambada_after']:.3f}",
            'vs Baseline': f"{run['lambada_after'] - baseline_stats['lambada_after']['mean']:+.3f}",
        })

    # Best compute efficiency (highest tokens/sec)
    sorted_by_tps = method_runs.sort_values('avg_tokens_per_sec', ascending=False)
    for rank, (_, run) in enumerate(sorted_by_tps.iterrows(), 1):
        rankings.append({
            'Category': 'Best Compute Efficiency (Highest Tokens/sec)',
            'Rank': rank,
            'Method': run['method'],
            'Value': f"{run['avg_tokens_per_sec']:.0f}",
            'vs Baseline': f"{run['avg_tokens_per_sec'] - baseline_stats['avg_tokens_per_sec']['mean']:+.0f}",
        })

    ranking_df = pd.DataFrame(rankings)

    # Generate markdown
    md = "# Method Rankings\n\n"
    md += "Rankings across key dimensions for RQ2 method comparison runs.\n\n"
    md += ranking_df.to_markdown(index=False)

    return ranking_df, md


def generate_baseline_variance_table(baseline_df: pd.DataFrame, baseline_stats: Dict) -> Tuple[pd.DataFrame, str]:
    """Generate baseline variance analysis table."""

    metrics = [
        ('forgetting_pct', 'Forgetting %', '{:.2f}%'),
        ('ppl_a_after', 'PPL_A After', '{:.2f}'),
        ('ppl_b_after', 'PPL_B After', '{:.2f}'),
        ('lambada_after', 'LAMBADA After', '{:.3f}'),
        ('rep4_after', 'Rep-4 After', '{:.4f}'),
        ('drift_value_after', 'Drift After', '{:.4f}'),
        ('total_hours', 'Total Hours', '{:.2f}h'),
        ('avg_tokens_per_sec', 'Tokens/sec', '{:.0f}'),
    ]

    rows = []
    for metric, display_name, fmt in metrics:
        if metric in baseline_stats:
            s = baseline_stats[metric]
            rows.append({
                'Metric': display_name,
                'Mean': fmt.format(s['mean']),
                'Std': f"{s['std']:.4f}".rstrip('0').rstrip('.') if s['std'] != 0 else '0',
                'Min': fmt.format(s['min']),
                'Max': fmt.format(s['max']),
                'N': int(s['n']),
            })

    variance_df = pd.DataFrame(rows)

    # Generate markdown
    md = "# Baseline Variance Analysis\n\n"
    md += "**Baseline runs**: pilot_baseline_s0, rq1_baseline_s42, rq1_baseline_s123\n\n"
    md += variance_df.to_markdown(index=False)

    return variance_df, md


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def create_forgetting_bar_chart(df: pd.DataFrame, baseline_stats: Dict, output_dir: Path):
    """Create bar chart of forgetting by method."""

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['baseline', 'replay25', 'mer25', 'ewc']
    x_positions = np.arange(len(methods))
    bar_width = 0.6

    values = []
    errors = []
    colors = []

    for method in methods:
        if method == 'baseline':
            values.append(baseline_stats['forgetting_pct']['mean'])
            errors.append(baseline_stats['forgetting_pct']['std'])
        else:
            method_data = df[df['method'] == method]['forgetting_pct'].values
            values.append(method_data[0] if len(method_data) > 0 else 0)
            errors.append(0)  # Single run, no error bar
        colors.append(METHOD_COLORS[method])

    bars = ax.bar(x_positions, values, bar_width, yerr=errors,
                  color=colors, edgecolor='black', linewidth=0.5,
                  capsize=5, error_kw={'linewidth': 1.5})

    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        label = f'{val:.1f}%'
        if err > 0:
            label += f' ± {err:.1f}'
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Forgetting (%)', fontweight='bold')
    ax.set_title('Catastrophic Forgetting by Method\n(Domain A: wikitext-103 → Domain B: arxiv_abstracts)',
                fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha='right')
    ax.set_ylim(0, 100)

    # Add horizontal line at baseline mean
    ax.axhline(y=baseline_stats['forgetting_pct']['mean'], color='gray',
               linestyle='--', linewidth=1, alpha=0.7, label='Baseline mean')

    ax.legend(loc='upper right')

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'fig1_forgetting_by_method.png', dpi=300)
    fig.savefig(output_dir / 'fig1_forgetting_by_method.pdf')
    plt.close(fig)

    return 'fig1_forgetting_by_method'


def create_ppl_tradeoff_scatter(df: pd.DataFrame, baseline_stats: Dict, output_dir: Path):
    """Create PPL tradeoff scatter plot."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot baseline as a point with error region
    baseline_ppl_a = baseline_stats['ppl_a_after']['mean']
    baseline_ppl_b = baseline_stats['ppl_b_after']['mean']
    baseline_ppl_a_std = baseline_stats['ppl_a_after']['std']
    baseline_ppl_b_std = baseline_stats['ppl_b_after']['std']

    # Plot baseline with error ellipse
    ax.scatter(baseline_ppl_a, baseline_ppl_b,
              c=METHOD_COLORS['baseline'], s=200, marker='s',
              edgecolors='black', linewidths=1.5, zorder=5,
              label=f"Baseline (n=3)")

    # Error bars for baseline
    ax.errorbar(baseline_ppl_a, baseline_ppl_b,
               xerr=baseline_ppl_a_std, yerr=baseline_ppl_b_std,
               fmt='none', ecolor=METHOD_COLORS['baseline'],
               capsize=5, capthick=2, elinewidth=2, zorder=4)

    # Plot method runs
    method_runs = get_method_runs(df)
    for _, run in method_runs.iterrows():
        method = run['method']
        ax.scatter(run['ppl_a_after'], run['ppl_b_after'],
                  c=METHOD_COLORS[method], s=200, marker='o',
                  edgecolors='black', linewidths=1.5, zorder=5,
                  label=METHOD_LABELS[method])

        # Annotate with method name
        ax.annotate(METHOD_LABELS[method],
                   xy=(run['ppl_a_after'], run['ppl_b_after']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                  color='gray', alpha=0.7))

    # Add reference lines
    ax.axvline(x=baseline_ppl_a, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=baseline_ppl_b, color='gray', linestyle=':', alpha=0.5)

    # Labels and title
    ax.set_xlabel('PPL on Domain A After Training (↓ better retention)', fontweight='bold')
    ax.set_ylabel('PPL on Domain B After Training (↓ better adaptation)', fontweight='bold')
    ax.set_title('PPL Tradeoff: Domain A Retention vs Domain B Adaptation\n'
                '(wikitext-103 → arxiv_abstracts)', fontweight='bold')

    # Add quadrant annotations
    ax.text(0.02, 0.98, 'Better A retention,\nWorse B adaptation',
           transform=ax.transAxes, fontsize=8, alpha=0.6, va='top')
    ax.text(0.98, 0.02, 'Worse A retention,\nBetter B adaptation',
           transform=ax.transAxes, fontsize=8, alpha=0.6, ha='right')
    ax.text(0.02, 0.02, 'Better both\n(ideal)',
           transform=ax.transAxes, fontsize=8, alpha=0.6, color='green')

    ax.legend(loc='upper right')

    plt.tight_layout()

    fig.savefig(output_dir / 'fig2_ppl_tradeoff.png', dpi=300)
    fig.savefig(output_dir / 'fig2_ppl_tradeoff.pdf')
    plt.close(fig)

    return 'fig2_ppl_tradeoff'


def create_rep4_paired_plot(df: pd.DataFrame, baseline_stats: Dict, output_dir: Path):
    """Create Rep-4 before vs after paired plot."""

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['baseline', 'replay25', 'mer25', 'ewc']
    x_positions = np.arange(len(methods))
    bar_width = 0.35

    before_values = []
    after_values = []

    for method in methods:
        if method == 'baseline':
            before_values.append(baseline_stats['rep4_before']['mean'])
            after_values.append(baseline_stats['rep4_after']['mean'])
        else:
            method_data = df[df['method'] == method]
            before_values.append(method_data['rep4_before'].values[0])
            after_values.append(method_data['rep4_after'].values[0])

    bars1 = ax.bar(x_positions - bar_width/2, before_values, bar_width,
                   label='Before (θ_A)', color='#a6cee3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_positions + bar_width/2, after_values, bar_width,
                   label='After (θ_AB)', color='#1f78b4', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Rep-4 Score (↓ better)', fontweight='bold')
    ax.set_title('Repetition (4-gram) Before vs After Domain B Training\n'
                '(wikitext-103 → arxiv_abstracts)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha='right')
    ax.legend(loc='upper left')

    # Add threshold line
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label='Anomaly threshold (0.25)')

    plt.tight_layout()

    fig.savefig(output_dir / 'fig3_rep4_comparison.png', dpi=300)
    fig.savefig(output_dir / 'fig3_rep4_comparison.pdf')
    plt.close(fig)

    return 'fig3_rep4_comparison'


def create_lambada_paired_plot(df: pd.DataFrame, baseline_stats: Dict, output_dir: Path):
    """Create LAMBADA before vs after paired plot."""

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['baseline', 'replay25', 'mer25', 'ewc']
    x_positions = np.arange(len(methods))
    bar_width = 0.35

    before_values = []
    after_values = []

    for method in methods:
        if method == 'baseline':
            before_values.append(baseline_stats['lambada_before']['mean'])
            after_values.append(baseline_stats['lambada_after']['mean'])
        else:
            method_data = df[df['method'] == method]
            before_values.append(method_data['lambada_before'].values[0])
            after_values.append(method_data['lambada_after'].values[0])

    bars1 = ax.bar(x_positions - bar_width/2, before_values, bar_width,
                   label='Before (θ_A)', color='#b2df8a', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_positions + bar_width/2, after_values, bar_width,
                   label='After (θ_AB)', color='#33a02c', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('LAMBADA Accuracy (↑ better)', fontweight='bold')
    ax.set_title('LAMBADA General Ability Before vs After Domain B Training\n'
                '(wikitext-103 → arxiv_abstracts)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.35)

    plt.tight_layout()

    fig.savefig(output_dir / 'fig4_lambada_comparison.png', dpi=300)
    fig.savefig(output_dir / 'fig4_lambada_comparison.pdf')
    plt.close(fig)

    return 'fig4_lambada_comparison'


def create_efficiency_scatter(df: pd.DataFrame, baseline_stats: Dict, output_dir: Path):
    """Create tokens/sec vs forgetting scatter plot."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baseline with error
    baseline_tps = baseline_stats['avg_tokens_per_sec']['mean']
    baseline_forgetting = baseline_stats['forgetting_pct']['mean']

    ax.scatter(baseline_tps, baseline_forgetting,
              c=METHOD_COLORS['baseline'], s=200, marker='s',
              edgecolors='black', linewidths=1.5, zorder=5,
              label=f"Baseline (n=3)")

    # Error bars
    ax.errorbar(baseline_tps, baseline_forgetting,
               xerr=baseline_stats['avg_tokens_per_sec']['std'],
               yerr=baseline_stats['forgetting_pct']['std'],
               fmt='none', ecolor=METHOD_COLORS['baseline'],
               capsize=5, capthick=2, elinewidth=2, zorder=4)

    # Plot method runs
    method_runs = get_method_runs(df)
    for _, run in method_runs.iterrows():
        method = run['method']
        ax.scatter(run['avg_tokens_per_sec'], run['forgetting_pct'],
                  c=METHOD_COLORS[method], s=200, marker='o',
                  edgecolors='black', linewidths=1.5, zorder=5,
                  label=METHOD_LABELS[method])

    ax.set_xlabel('Tokens/sec (↑ better efficiency)', fontweight='bold')
    ax.set_ylabel('Forgetting % (↓ better retention)', fontweight='bold')
    ax.set_title('Compute Efficiency vs Retention Tradeoff\n'
                '(wikitext-103 → arxiv_abstracts)', fontweight='bold')
    ax.legend(loc='upper right')

    # Ideal corner annotation
    ax.annotate('Ideal\n(high efficiency,\nlow forgetting)',
               xy=(3100, 0), fontsize=9, alpha=0.6, color='green',
               ha='center')

    plt.tight_layout()

    fig.savefig(output_dir / 'fig5_efficiency_tradeoff.png', dpi=300)
    fig.savefig(output_dir / 'fig5_efficiency_tradeoff.pdf')
    plt.close(fig)

    return 'fig5_efficiency_tradeoff'


def create_drift_paired_plot(df: pd.DataFrame, baseline_stats: Dict, output_dir: Path):
    """Create drift (JS divergence) before vs after paired plot."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = ['baseline', 'replay25', 'mer25', 'ewc']
    x_positions = np.arange(len(methods))
    bar_width = 0.35

    # JS Divergence
    before_values = []
    after_values = []

    for method in methods:
        if method == 'baseline':
            before_values.append(baseline_stats['drift_value_before']['mean'])
            after_values.append(baseline_stats['drift_value_after']['mean'])
        else:
            method_data = df[df['method'] == method]
            before_values.append(method_data['drift_value_before'].values[0])
            after_values.append(method_data['drift_value_after'].values[0])

    ax1.bar(x_positions - bar_width/2, before_values, bar_width,
            label='Before (θ_A)', color='#fdbf6f', edgecolor='black', linewidth=0.5)
    ax1.bar(x_positions + bar_width/2, after_values, bar_width,
            label='After (θ_AB)', color='#ff7f00', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Method', fontweight='bold')
    ax1.set_ylabel('JS Divergence', fontweight='bold')
    ax1.set_title('Distribution Drift (JS Divergence)', fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha='right')
    ax1.legend(loc='upper right')

    # Vocab Overlap
    before_values = []
    after_values = []

    for method in methods:
        if method == 'baseline':
            before_values.append(baseline_stats['vocab_overlap_before']['mean'])
            after_values.append(baseline_stats['vocab_overlap_after']['mean'])
        else:
            method_data = df[df['method'] == method]
            before_values.append(method_data['vocab_overlap_before'].values[0])
            after_values.append(method_data['vocab_overlap_after'].values[0])

    ax2.bar(x_positions - bar_width/2, before_values, bar_width,
            label='Before (θ_A)', color='#cab2d6', edgecolor='black', linewidth=0.5)
    ax2.bar(x_positions + bar_width/2, after_values, bar_width,
            label='After (θ_AB)', color='#6a3d9a', edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Method', fontweight='bold')
    ax2.set_ylabel('Vocabulary Overlap', fontweight='bold')
    ax2.set_title('Vocabulary Overlap with Base Model', fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha='right')
    ax2.legend(loc='upper right')

    fig.suptitle('Distribution Drift Metrics (wikitext-103 → arxiv_abstracts)',
                fontweight='bold', fontsize=12, y=1.02)

    plt.tight_layout()

    fig.savefig(output_dir / 'fig6_drift_metrics.png', dpi=300)
    fig.savefig(output_dir / 'fig6_drift_metrics.pdf')
    plt.close(fig)

    return 'fig6_drift_metrics'


def create_summary_comparison_chart(df: pd.DataFrame, baseline_stats: Dict, output_dir: Path):
    """Create a comprehensive summary comparison chart."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    methods = ['baseline', 'replay25', 'mer25', 'ewc']
    x_positions = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]

    def get_values(metric):
        values = []
        errors = []
        for method in methods:
            if method == 'baseline':
                values.append(baseline_stats[metric]['mean'])
                errors.append(baseline_stats[metric]['std'])
            else:
                method_data = df[df['method'] == method][metric].values
                values.append(method_data[0] if len(method_data) > 0 else 0)
                errors.append(0)
        return values, errors

    # Forgetting
    ax = axes[0, 0]
    values, errors = get_values('forgetting_pct')
    ax.bar(x_positions, values, color=colors, edgecolor='black', linewidth=0.5, yerr=errors, capsize=3)
    ax.set_title('Forgetting %\n(↓ better)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # PPL_A After
    ax = axes[0, 1]
    values, errors = get_values('ppl_a_after')
    ax.bar(x_positions, values, color=colors, edgecolor='black', linewidth=0.5, yerr=errors, capsize=3)
    ax.set_title('PPL on Domain A\n(↓ better retention)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # PPL_B After
    ax = axes[0, 2]
    values, errors = get_values('ppl_b_after')
    ax.bar(x_positions, values, color=colors, edgecolor='black', linewidth=0.5, yerr=errors, capsize=3)
    ax.set_title('PPL on Domain B\n(↓ better adaptation)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # LAMBADA
    ax = axes[1, 0]
    values, errors = get_values('lambada_after')
    ax.bar(x_positions, values, color=colors, edgecolor='black', linewidth=0.5, yerr=errors, capsize=3)
    ax.set_title('LAMBADA Accuracy\n(↑ better)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # Tokens/sec
    ax = axes[1, 1]
    values, errors = get_values('avg_tokens_per_sec')
    ax.bar(x_positions, values, color=colors, edgecolor='black', linewidth=0.5, yerr=errors, capsize=3)
    ax.set_title('Throughput (tokens/sec)\n(↑ better)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # Total Hours
    ax = axes[1, 2]
    values, errors = get_values('total_hours')
    ax.bar(x_positions, values, color=colors, edgecolor='black', linewidth=0.5, yerr=errors, capsize=3)
    ax.set_title('Total Time (hours)\n(↓ better)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    fig.suptitle('Comprehensive Method Comparison\n(Domain A: wikitext-103 → Domain B: arxiv_abstracts)',
                fontweight='bold', fontsize=14, y=1.02)

    plt.tight_layout()

    fig.savefig(output_dir / 'fig7_summary_comparison.png', dpi=300)
    fig.savefig(output_dir / 'fig7_summary_comparison.pdf')
    plt.close(fig)

    return 'fig7_summary_comparison'


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ANALYSIS ASSETS GENERATOR")
    print("=" * 70)
    print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print()

    # Setup paths
    project_root = get_project_root()
    tables_dir = project_root / "experiments" / "analysis" / "tables"
    figures_dir = project_root / "experiments" / "analysis" / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data sources...")
    summary_df = load_summary_table(project_root)
    registry_df = load_run_registry(project_root)

    print(f"  Summary table: {len(summary_df)} rows")
    print(f"  Run registry: {len(registry_df)} rows")

    # Filter to MPP runs
    mpp_df = filter_mpp_runs(summary_df)
    print(f"  MPP core runs: {len(mpp_df)} (smoke runs excluded)")

    baseline_df = get_baseline_runs(mpp_df)
    method_df = get_method_runs(mpp_df)
    print(f"  Baseline runs: {len(baseline_df)}")
    print(f"  Method runs: {len(method_df)}")

    # Validate labels
    print("\nValidating labels...")
    for _, row in mpp_df.iterrows():
        run_id = row['run_id']
        method = row['method']
        seed = row['seed']
        rq = row['research_question']

        # Check registry match
        reg_row = registry_df[registry_df['run_id'] == run_id]
        if len(reg_row) > 0:
            reg_method = reg_row['method'].values[0]
            reg_seed = reg_row['seed'].values[0]
            reg_rq = reg_row['research_question'].values[0]

            if method != reg_method or seed != reg_seed or rq != reg_rq:
                print(f"  WARNING: Label mismatch for {run_id}")
                print(f"    Summary: method={method}, seed={seed}, rq={rq}")
                print(f"    Registry: method={reg_method}, seed={reg_seed}, rq={reg_rq}")
            else:
                print(f"  ✓ {run_id}: method={method}, seed={seed}, rq={rq}")
        else:
            print(f"  WARNING: {run_id} not found in registry")

    # Compute baseline stats
    print("\nComputing baseline statistics...")
    baseline_stats = compute_baseline_stats(baseline_df)
    print(f"  Baseline forgetting: {baseline_stats['forgetting_pct']['mean']:.2f}% ± {baseline_stats['forgetting_pct']['std']:.2f}%")

    # Generate tables
    print("\nGenerating tables...")

    primary_df, primary_md = generate_primary_results_table(mpp_df, baseline_stats)
    primary_df.to_csv(tables_dir / 'table1_primary_results.csv', index=False)
    (tables_dir / 'table1_primary_results.md').write_text(primary_md)
    print(f"  ✓ table1_primary_results.csv/md")

    delta_df, delta_md = generate_delta_table(mpp_df, baseline_stats)
    delta_df.to_csv(tables_dir / 'table2_delta_vs_baseline.csv', index=False)
    (tables_dir / 'table2_delta_vs_baseline.md').write_text(delta_md)
    print(f"  ✓ table2_delta_vs_baseline.csv/md")

    ranking_df, ranking_md = generate_ranking_table(mpp_df, baseline_stats)
    ranking_df.to_csv(tables_dir / 'table3_rankings.csv', index=False)
    (tables_dir / 'table3_rankings.md').write_text(ranking_md)
    print(f"  ✓ table3_rankings.csv/md")

    variance_df, variance_md = generate_baseline_variance_table(baseline_df, baseline_stats)
    variance_df.to_csv(tables_dir / 'table4_baseline_variance.csv', index=False)
    (tables_dir / 'table4_baseline_variance.md').write_text(variance_md)
    print(f"  ✓ table4_baseline_variance.csv/md")

    # Generate figures
    print("\nGenerating figures...")

    fig1 = create_forgetting_bar_chart(mpp_df, baseline_stats, figures_dir)
    print(f"  ✓ {fig1}.png/pdf")

    fig2 = create_ppl_tradeoff_scatter(mpp_df, baseline_stats, figures_dir)
    print(f"  ✓ {fig2}.png/pdf")

    fig3 = create_rep4_paired_plot(mpp_df, baseline_stats, figures_dir)
    print(f"  ✓ {fig3}.png/pdf")

    fig4 = create_lambada_paired_plot(mpp_df, baseline_stats, figures_dir)
    print(f"  ✓ {fig4}.png/pdf")

    fig5 = create_efficiency_scatter(mpp_df, baseline_stats, figures_dir)
    print(f"  ✓ {fig5}.png/pdf")

    fig6 = create_drift_paired_plot(mpp_df, baseline_stats, figures_dir)
    print(f"  ✓ {fig6}.png/pdf")

    fig7 = create_summary_comparison_chart(mpp_df, baseline_stats, figures_dir)
    print(f"  ✓ {fig7}.png/pdf")

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nTables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print(f"\nFiles generated:")
    print(f"  - 4 tables (CSV + Markdown)")
    print(f"  - 7 figures (PNG + PDF)")

    # Return summary for validation
    return {
        'tables_dir': str(tables_dir),
        'figures_dir': str(figures_dir),
        'n_runs': len(mpp_df),
        'n_baseline': len(baseline_df),
        'n_methods': len(method_df),
        'baseline_forgetting_mean': baseline_stats['forgetting_pct']['mean'],
        'baseline_forgetting_std': baseline_stats['forgetting_pct']['std'],
    }


if __name__ == "__main__":
    main()
