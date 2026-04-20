"""
Utility functions extracted from the HMS analysis notebook.

Contains helper functions for model renaming, size parsing,
sensitivity result aggregation, and ablation correlation plotting.
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path


def rename_models(smoothed_data: dict, rename_map: dict) -> dict:
    """
    Rename model keys in a smoothed_data dict.

    Parameters:
        smoothed_data: Dict mapping model_name -> {seed -> data}
        rename_map: Dict mapping old_name -> new_name

    Returns:
        New dict with renamed keys (unmapped keys kept as-is).
    """
    renamed_data = {}
    for model_name, seeds in smoothed_data.items():
        new_model_name = rename_map.get(model_name, model_name)
        renamed_data[new_model_name] = seeds
    return renamed_data


def model_size_to_float(name: str) -> float:
    """
    Parse model size from a name string (e.g. '7B' -> 7.0, '0.5B' -> 0.5).

    Returns inf if no size pattern is found.
    """
    m = re.search(r'(\d+(?:\.\d+)?)\s*B', name)
    return float(m.group(1)) if m else float('inf')


def aggregate_sensitivity_results(
    boot_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate bootstrap CI and LOO results into a summary table.

    Parameters:
        boot_df: Output of bootstrap_seed_sensitivity / bootstrap_from_smoothed_data.
            Must have columns: benchmark_metric, ci_low, ci_high
        loo_df: Output of loo_sensitivity.
            Must have columns: benchmark_metric, loo_min, loo_max
        metrics: Metrics to include as rows.
            Defaults to ['num_params', 'capability_score', 'sycophancy'].

    Returns:
        DataFrame with columns: metric, Bootstrap CI, LOO range
    """
    if metrics is None:
        metrics = ['num_params', 'capability_score', 'sycophancy']

    boot_sub = boot_df.set_index('benchmark_metric')
    loo_sub = loo_df.set_index('benchmark_metric')

    rows = []
    for m in metrics:
        ci_low = boot_sub.loc[m, 'ci_low']
        ci_high = boot_sub.loc[m, 'ci_high']
        loo_min = loo_sub.loc[m, 'loo_min']
        loo_max = loo_sub.loc[m, 'loo_max']
        rows.append({
            'metric':       m,
            'Bootstrap CI': f'[{ci_low:.3f}, {ci_high:.3f}]',
            'LOO range':    f'[{loo_min:.3f}, {loo_max:.3f}]',
        })

    return pd.DataFrame(rows).set_index('metric')


def plot_ablation_correlations(
    ablation_results: Dict[str, list],
    benchmark_metrics: List[str],
    corr_metric: str,
    figsize=(10, 6),
    group_spacing: float = 0.15,
    output_path: Optional[Path] = None,
):
    """
    Bar chart comparing correlations across ablation experiments.

    Bars show mean Spearman rho over seeds; error bars show +/- 1 std.

    Parameters:
        ablation_results: Dict mapping experiment labels to list of per-seed corr DataFrames
        benchmark_metrics: List of benchmark metric names to plot
        corr_metric: HMS metric name (e.g., 'max_er_gap', 'hms_score')
        figsize: Figure size (width, height)
        group_spacing: Width of individual bars
        output_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(ablation_results.keys())
    x = np.arange(len(benchmark_metrics))
    n_experiments = len(labels)

    colors = ["#630062", "#970b54", "#db3657", "#e74c7d", "#f6af35"]

    for e_idx, label in enumerate(labels):
        corr_dfs = ablation_results[label]
        rho_means = []
        rho_stds = []
        for bm in benchmark_metrics:
            rho_per_seed = []
            for corr_df in corr_dfs:
                row = corr_df[corr_df["benchmark_metric"] == bm]
                rho = float(row["spearman_r"].values[0]) if len(row) > 0 else 0.0
                rho_per_seed.append(rho)
            rho_means.append(np.mean(rho_per_seed))
            rho_stds.append(np.std(rho_per_seed))

        offset = (e_idx - n_experiments / 2 + 0.5) * group_spacing
        ax.bar(
            x + offset, rho_means, group_spacing,
            label=label,
            color=colors[e_idx % len(colors)], alpha=0.85,
            edgecolor="black", linewidth=0.5,
            yerr=rho_stds, capsize=3,
            error_kw={"linewidth": 1, "ecolor": "black"},
        )

    ax.set_xlabel("Benchmark Metric", fontsize=14)
    ax.set_ylabel("Spearman \u03c1", fontsize=14)
    ax.set_title(
        "Misalignment Correlation: Environment Ablation",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [bm.replace("_", " ").title() for bm in benchmark_metrics], fontsize=12
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.67, 1.0))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()
    return fig, ax
