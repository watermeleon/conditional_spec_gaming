"""
Multi-environment scatter grid plots for cross-environment correlation analysis.

Functions for collecting correlation results across multiple environments and
visualizing them as scatter grids suitable for academic papers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr

from .compute_hms_metrics_stats import compute_aggregate_statistics, get_smoothed_metrics, compare_metrics


def _get_exp_name_for_env(env_name: str) -> str:
    """Import helper to avoid circular imports."""
    from .hms_plotting import get_exp_name_for_env
    return get_exp_name_for_env(env_name)


def collect_multi_env_correlations(
    env_names: List[str],
    sw_metrics: pd.DataFrame,
    corr_metric: str = 'max_reward',
    get_all_results_dirs_for_experiment=None,
    smoothing_window: int = 50,
    max_step: int = 300,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Collect correlation results across multiple environments.

    Runs the full pipeline (load -> smooth -> aggregate -> correlate) for each
    environment and returns the results in a dict.

    Parameters:
        env_names: List of environment names (e.g. ["action_advice", "therapy_talk"])
        sw_metrics: DataFrame with benchmark metrics per model
        corr_metric: HMS metric column to correlate against (e.g. 'max_reward', 'hms_score')
        get_all_results_dirs_for_experiment: Function to load results dirs
            (from llm_agent.utils.utils)
        smoothing_window: Window size for smoothing
        max_step: Maximum training step to consider

    Returns:
        Dict mapping env_name -> (corr_df, merged_data)
    """
    env_results = {}

    for env_name in env_names:
        print(f"## Processing: {env_name}")

        exp_name = _get_exp_name_for_env(env_name)
        all_results = get_all_results_dirs_for_experiment(exp_name, seed_nr=None)

        smoothed_data = get_smoothed_metrics(
            all_results,
            env_name,
            model_names=None,
            seed_nrs=None,
            smoothing_window=smoothing_window,
            center=True,
            include_reward=True,
            max_step=max_step,
            verbose=False,
        )

        df_agg = compute_aggregate_statistics(smoothed_data, max_step=max_step)
        corr_df, merged = compare_metrics(
            df_agg, sw_metrics, corr_metric=corr_metric, verbose=False
        )

        env_results[env_name] = (corr_df, merged)

    return env_results


# Model family definitions for coloring
FAMILY_PALETTE = {
    'qwen':  ('Qwen',  '#630062'),
    'gemma': ('Gemma', '#db3657'),
    'llama': ('LLaMA', "#f6cc35"),
    'yi':    ('Yi',    "#48A37B"),
}

DEFAULT_COLOR = 'steelblue'


def _get_family(model_name: str) -> Optional[str]:
    """Detect model family from model name."""
    name_lower = model_name.lower()
    for key in FAMILY_PALETTE:
        if key in name_lower:
            return key
    return None


def _jitter_duplicates(x_arr: np.ndarray, jitter_fraction: float = 0.02) -> np.ndarray:
    """Spread duplicate x values to avoid overlapping points."""
    x_range = x_arr.max() - x_arr.min() if len(x_arr) > 1 else 1.0
    jitter_half = jitter_fraction * x_range
    x_jittered = x_arr.copy()
    unique_vals, counts = np.unique(x_arr, return_counts=True)
    for val, count in zip(unique_vals, counts):
        if count > 1:
            idxs = np.where(x_arr == val)[0]
            offsets = np.linspace(-jitter_half, jitter_half, count)
            x_jittered[idxs] = val + offsets
    return x_jittered


def plot_multi_env_scatter_grid(
    env_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    benchmark_metrics: List[str],
    corr_metric: str = 'max_reward',
    figsize_per_cell: Tuple[float, float] = (3.5, 3.0),
    output_path: Optional[Path] = None,
    include_model_labels: bool = True,
    show_std: bool = False,
    show_families: bool = False,
    row_label_fontsize: int = 14,
    col_label_fontsize: int = 11,
):
    """
    Scatter grid for academic papers: rows = environments, columns = benchmark metrics.

    Environment names appear as large rotated labels on the right margin.
    Benchmark metric names appear as column headers on top.

    Parameters:
        env_results: Dict mapping env_name -> (corr_df, merged_data)
        benchmark_metrics: List of benchmark metric names (become columns)
        corr_metric: Name of the y-axis metric
        figsize_per_cell: (width, height) per subplot cell
        output_path: Optional path to save the figure
        include_model_labels: Whether to label each point with model name
        show_std: Whether to show error bars for y-axis std
        show_families: If True, color points by model family and add a legend
        row_label_fontsize: Font size for environment row labels
        col_label_fontsize: Font size for metric column headers
    """
    env_names = list(env_results.keys())
    n_rows = len(env_names)
    n_cols = len(benchmark_metrics)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )

    corr_metric_label = corr_metric.replace("_", " ").title()
    corr_metric_label = corr_metric_label.replace("Er", "HEX")

    fig.subplots_adjust(right=0.88)

    legend_handles: Dict[str, plt.Artist] = {}

    for row_idx, env_name in enumerate(env_names):
        corr_df, merged = env_results[env_name]
        env_label = env_name.replace("_", " ").title()

        axes[row_idx, -1].annotate(
            env_label,
            xy=(1.1, 0.5), xycoords='axes fraction',
            fontsize=row_label_fontsize, fontweight='bold',
            ha='center', va='center', rotation=-90,
        )

        for col_idx, bm in enumerate(benchmark_metrics):
            ax = axes[row_idx, col_idx]

            if bm not in merged.columns:
                ax.text(0.5, 0.5, f"'{bm}'\nnot found",
                        ha='center', va='center', transform=ax.transAxes)
                if row_idx == 0:
                    ax.set_title(bm.replace("_", " ").title(),
                                 fontsize=col_label_fontsize, fontweight='bold', pad=30)
                continue

            valid = merged[[corr_metric, bm]].notna().all(axis=1)
            x = merged.loc[valid, bm]
            y = merged.loc[valid, corr_metric]
            labels = merged.loc[valid, 'model']

            std_col = f"{corr_metric}_std"
            yerr = merged.loc[valid, std_col] if (show_std and std_col in merged.columns) else None

            x_arr = x.to_numpy(dtype=float)
            x_jittered = _jitter_duplicates(x_arr)

            # Determine per-point colors
            if show_families:
                point_colors = []
                for lbl in labels:
                    fam = _get_family(lbl)
                    color = FAMILY_PALETTE[fam][1] if fam is not None else DEFAULT_COLOR
                    point_colors.append(color)
                    fam_key = fam if fam is not None else '__other__'
                    if fam_key not in legend_handles:
                        disp = FAMILY_PALETTE[fam][0] if fam is not None else 'Other'
                        legend_handles[fam_key] = plt.Line2D(
                            [0], [0], marker='o', color='w',
                            markerfacecolor=color, markersize=8,
                            markeredgecolor='black', markeredgewidth=0.5,
                            label=disp,
                        )
            else:
                point_colors = [DEFAULT_COLOR] * len(x_arr)

            # Scatter / errorbar
            if yerr is not None:
                if show_families:
                    for xi, yi, ei, ci in zip(x_jittered, y, yerr, point_colors):
                        ax.errorbar(xi, yi, yerr=ei, fmt='o', ms=6, alpha=0.7,
                                    ecolor=ci, elinewidth=1.2, capsize=3,
                                    color=ci, markeredgecolor='black',
                                    markeredgewidth=0.5, zorder=3)
                else:
                    ax.errorbar(x_jittered, y, yerr=yerr, fmt='o', ms=6, alpha=0.7,
                                ecolor=DEFAULT_COLOR, elinewidth=1.2, capsize=3,
                                color=DEFAULT_COLOR, markeredgecolor='black',
                                markeredgewidth=0.5, zorder=3)
            else:
                if show_families:
                    ax.scatter(x_jittered, y, s=60, alpha=0.9, edgecolors='black',
                               linewidth=0.5, zorder=3, c=point_colors)
                else:
                    ax.scatter(x_jittered, y, s=60, alpha=0.7, edgecolors='black',
                               linewidth=0.5, zorder=3, color=DEFAULT_COLOR)

            # Label each point
            if include_model_labels:
                for xi, yi, lbl in zip(x_jittered, y, labels):
                    short = lbl.replace("_Chat", "").replace("_hf", "").replace("_it", "")
                    ax.annotate(short, (xi, yi), fontsize=5.5, alpha=0.7,
                                xytext=(3, 3), textcoords='offset points')

            # Regression line
            if len(x) >= 2:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x)
                ax.plot(x_sorted, p(x_sorted), 'r--', alpha=0.7, linewidth=1.5)

            # Spearman annotation
            if len(x) >= 3:
                rho, pval = spearmanr(x, y)
                stars = '**' if pval < 0.01 else ('*' if pval < 0.05 else '')
                ax.annotate(
                    f"\u03c1={rho:.2f}{stars} (p={pval:.3f})",
                    xy=(0.5, 1.04), xycoords='axes fraction',
                    fontsize=10, va='bottom', ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5),
                )

            # x-axis label only on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel(bm.replace("_", " ").title(), fontsize=11, fontweight='bold')
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=True)

            # y-axis label only on leftmost column
            if col_idx == 0:
                ax.set_ylabel(corr_metric_label, fontsize=10)
            else:
                ax.set_ylabel('')

            ax.grid(True, alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(right=0.88)

    # Family legend
    if show_families and legend_handles:
        handles = list(legend_handles.values())
        fig.legend(
            handles=handles,
            loc='upper center',
            ncol=4,
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, 0.97),
            title_fontsize=10,
        )
        fig.subplots_adjust(top=0.88)

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved scatter grid to: {output_path}")
    plt.show()


# Backward compatibility alias
plot_multi_env_scatter_grid_v2 = plot_multi_env_scatter_grid


def plot_multi_env_scatter_single_col(
    env_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    benchmark_metric: str = 'num_params',
    corr_metric: str = 'max_reward',
    figsize_per_cell: Tuple[float, float] = (3.0, 2.7),
    figsize_scaler: float = 0.9,
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[Path] = None,
    log_x: bool = False,
    show_std: bool = False,
    show_families: bool = True,
    include_model_labels: bool = False,
    env_label_loc: str = 'title',
    # Size/style knobs — defaults match plot_multi_env_scatter_grid for visual consistency
    row_label_fontsize: int = 13,
    col_label_fontsize: int = 11,
    axis_label_fontsize: int = 11,
    ylabel_fontsize: int = 10,
    tick_labelsize: Optional[int] = None,
    rho_fontsize: int = 10,
    legend_fontsize: int = 10,
    legend_markersize: int = 8,
    scatter_size: int = 60,
    scatter_edgewidth: float = 0.5,
    regression_linewidth: float = 1.5,
    rho_positions: Optional[Dict[str, str]] = None,
):
    """
    Single-column scatter figure: rows = environments, one benchmark metric.

    Visually matches ``plot_multi_env_scatter_grid`` (same default fonts, marker
    sizes, and line widths) so it can be dropped into a paper alongside the
    wider grid figures without looking stylistically different. The only
    structural changes are:

    - single column, stacked rows (``sharex=True``)
    - compact horizontal family legend at the top
    - ``log_x=True`` by default (useful for ``num_params``)

    Figure size follows the ``figsize_per_cell`` / ``figsize_scaler`` convention
    from the grid function, so you can pass the same values you used there and
    expect per-cell dimensions to match. Pass ``figsize`` to override entirely.

    Parameters:
        env_results: Dict mapping env_name -> (corr_df, merged_data).
            Insertion order = row order (top -> bottom).
        benchmark_metric: Single benchmark column name (e.g. 'num_params').
        corr_metric: y-axis metric column (e.g. 'max_reward').
        figsize_per_cell: (width, height) per subplot cell, in inches.
        figsize_scaler: Multiplier applied to ``figsize_per_cell``.
        figsize: If set, overrides ``figsize_per_cell`` / ``figsize_scaler``.
        output_path: Optional path to save the figure.
        log_x: Use log scale on x-axis (and fit regression in log space).
        show_std: Draw y-axis error bars using f"{corr_metric}_std".
        show_families: Color points by model family and add a top legend.
        include_model_labels: Annotate each point with its model name.
        env_label_loc: 'title' (above axes) or 'inside' (top-left annotation).
        row_label_fontsize: Env label fontsize (title or inside annotation).
        col_label_fontsize: Reserved for API symmetry with the grid function.
        axis_label_fontsize: x-axis label fontsize.
        ylabel_fontsize: y-axis label fontsize.
        tick_labelsize: Tick label fontsize (None = matplotlib default).
        rho_fontsize: Spearman ρ annotation fontsize.
        legend_fontsize: Family legend fontsize.
        legend_markersize: Family legend marker size.
        scatter_size: ``s`` passed to ``ax.scatter``.
        scatter_edgewidth: Marker edge linewidth.
        regression_linewidth: Red dashed regression line width.
        rho_positions: Per-env dict mapping env_name -> corner for the ρ box.
            Valid values: 'top left', 'top right', 'bottom left', 'bottom right'.
            Useful when correlations differ in sign and empty space lives in
            different corners (e.g. TT top-right, AA top-left). Defaults to
            'top left' for all envs.
    """
    # Map corner name -> (x, y, ha, va) in axes fraction
    _rho_corner_coords = {
        'top left':     (0.03, 0.97, 'left',  'top'),
        'top right':    (0.97, 0.97, 'right', 'top'),
        'bottom left':  (0.03, 0.03, 'left',  'bottom'),
        'bottom right': (0.97, 0.03, 'right', 'bottom'),
    }
    if rho_positions is None:
        rho_positions = {}

    env_names = list(env_results.keys())
    n_rows = len(env_names)

    if figsize is None:
        cell_w = figsize_per_cell[0] * figsize_scaler
        cell_h = figsize_per_cell[1] * figsize_scaler
        figsize = (cell_w, cell_h * n_rows)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=figsize,
        squeeze=False,
        sharex=True,
    )
    axes = axes[:, 0]

    corr_metric_label = corr_metric.replace("_", " ").title().replace("Er", "HEX")
    bm_label = benchmark_metric.replace("_", " ").title()

    legend_handles: Dict[str, plt.Artist] = {}

    for row_idx, (env_name, ax) in enumerate(zip(env_names, axes)):
        corr_df, merged = env_results[env_name]
        env_label = env_name.replace("_", " ").title()

        if benchmark_metric not in merged.columns:
            ax.text(0.5, 0.5, f"'{benchmark_metric}'\nnot found",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        valid = merged[[corr_metric, benchmark_metric]].notna().all(axis=1)
        x = merged.loc[valid, benchmark_metric]
        y = merged.loc[valid, corr_metric]
        labels = merged.loc[valid, 'model']

        std_col = f"{corr_metric}_std"
        yerr = merged.loc[valid, std_col] if (show_std and std_col in merged.columns) else None

        x_arr = x.to_numpy(dtype=float)
        # In log space, jittering overlapping params is unnecessary (and would distort)
        x_plot = x_arr if log_x else _jitter_duplicates(x_arr)

        if show_families:
            point_colors = []
            for lbl in labels:
                fam = _get_family(lbl)
                color = FAMILY_PALETTE[fam][1] if fam is not None else DEFAULT_COLOR
                point_colors.append(color)
                fam_key = fam if fam is not None else '__other__'
                if fam_key not in legend_handles:
                    disp = FAMILY_PALETTE[fam][0] if fam is not None else 'Other'
                    legend_handles[fam_key] = plt.Line2D(
                        [0], [0], marker='o', color='w',
                        markerfacecolor=color, markersize=legend_markersize,
                        markeredgecolor='black', markeredgewidth=0.5,
                        label=disp,
                    )
        else:
            point_colors = [DEFAULT_COLOR] * len(x_arr)

        if yerr is not None:
            for xi, yi, ei, ci in zip(x_plot, y, yerr, point_colors):
                ax.errorbar(xi, yi, yerr=ei, fmt='o', ms=6, alpha=0.7,
                            ecolor=ci, elinewidth=1.2, capsize=3,
                            color=ci, markeredgecolor='black',
                            markeredgewidth=scatter_edgewidth, zorder=3)
        else:
            ax.scatter(x_plot, y, s=scatter_size, alpha=0.9,
                       edgecolors='black', linewidth=scatter_edgewidth,
                       zorder=3, c=point_colors)

        if include_model_labels:
            for xi, yi, lbl in zip(x_plot, y, labels):
                short = lbl.replace("_Chat", "").replace("_hf", "").replace("_it", "")
                ax.annotate(short, (xi, yi), fontsize=5.5,
                            alpha=0.7, xytext=(3, 3), textcoords='offset points')

        if log_x:
            ax.set_xscale('log')

        # Regression line — fit in the x-space that is actually plotted
        if len(x_arr) >= 2:
            x_fit = np.log10(x_arr) if log_x else x_arr
            z = np.polyfit(x_fit, y.to_numpy(dtype=float), 1)
            p = np.poly1d(z)
            xs = np.linspace(x_fit.min(), x_fit.max(), 50)
            xs_plot = 10 ** xs if log_x else xs
            ax.plot(xs_plot, p(xs), 'r--', alpha=0.7, linewidth=regression_linewidth)

        rho_text = None
        if len(x_arr) >= 3:
            rho, pval = spearmanr(x_arr, y.to_numpy(dtype=float))
            stars = '**' if pval < 0.01 else ('*' if pval < 0.05 else '')
            # ρ and p-value stacked on two lines, no brackets
            rho_text = f"\u03c1={rho:.2f}{stars}\np={pval:.3f}"

        # ρ annotation inside the axes. Position is per-env via `rho_positions`
        # (e.g. {"therapy_talk": "top right", "action_advice": "top left"})
        # so we can place it in the empty corner for each panel.
        if rho_text is not None:
            corner = rho_positions.get(env_name, 'top left')
            x_frac, y_frac, ha, va = _rho_corner_coords[corner]
            ax.annotate(
                rho_text,
                xy=(x_frac, y_frac), xycoords='axes fraction',
                fontsize=rho_fontsize, va=va, ha=ha, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7),
            )

        if env_label_loc == 'inside':
            ax.annotate(env_label,
                        xy=(0.03, 0.96), xycoords='axes fraction',
                        fontsize=row_label_fontsize, fontweight='bold',
                        ha='left', va='top')
        else:
            ax.set_title(env_label, fontsize=row_label_fontsize,
                         fontweight='bold', pad=4)

        ax.set_ylabel(corr_metric_label, fontsize=ylabel_fontsize, fontweight='bold')
        if tick_labelsize is not None:
            ax.tick_params(axis='both', labelsize=tick_labelsize)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel(bm_label, fontsize=axis_label_fontsize, fontweight='bold')

    plt.tight_layout()

    # Compact family legend at top of figure, single row above the top subplot.
    # Leaves headroom via subplots_adjust so the legend doesn't collide with the
    # top env title + rho annotation.
    if show_families and legend_handles:
        handles = list(legend_handles.values())
        fig.legend(
            handles=handles,
            loc='upper center',
            ncol=len(handles),
            fontsize=legend_fontsize,
            frameon=True,
            handletextpad=0.1,
            columnspacing=0.5,
            borderpad=0.3,
            borderaxespad=0.1,
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.subplots_adjust(top=0.88)

    if output_path is not None:
        plt.savefig(output_path, dpi=500, bbox_inches='tight')
        print(f"Saved single-column scatter to: {output_path}")
    plt.show()
