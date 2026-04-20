"""
Utilities for loading, smoothing, and plotting HMS metrics and rewards.

This module provides clean, efficient functions for:
- Loading HMS metrics and rewards from experiment results
- Applying rolling average smoothing
- Plotting individual seeds or mean±std across seeds
- Supporting multiple metric types (ACC, ER, Reward)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from .compute_hms_metrics_stats import compute_aggregate_statistics, get_smoothed_metrics, compare_metrics

# ============================================================================
# EXPERIMENT CONFIG & DATA LOADING
# ============================================================================

def get_exp_name_for_env(env_name: str) -> str:
    """Map environment name to experiment name."""
    env_to_exp = {
        "therapy_talk": "exp5seeds",
        "action_advice": "exp9AA",
        "political_qa": "exp2PQA",
    }
    if env_name not in env_to_exp:
        raise ValueError(f"Unknown env_name: {env_name}. Known: {list(env_to_exp.keys())}")
    return env_to_exp[env_name]


def load_model_features_metrics(file_path: str) -> pd.DataFrame:
    """Load benchmark metrics from a JSON file into a DataFrame."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)



# ============================================================================
# COLOR UTILITIES
# ============================================================================

def adjust_color_lightness(color, amount=1.0):
    """
    Adjust the lightness of a color.

    Parameters:
        color: matplotlib color (RGB tuple or hex)
        amount: lightness multiplier (0.5=darker, 1.0=unchanged, 1.5=lighter)

    Returns:
        RGB tuple
    """
    try:
        c = mcolors.to_rgb(color)
    except:
        c = color
    c = colorsys.rgb_to_hls(*c)
    new_lightness = max(0, min(1, amount * c[1]))
    return colorsys.hls_to_rgb(c[0], new_lightness, c[2])



# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def plot_metric_line(
    ax,
    x_data,
    y_data,
    color,
    label: str,
    linestyle: str = 'solid',
    linewidth: float = 2,
    marker: str = 'o',
    markersize: float = 4,
    markevery: Optional[int] = None
):
    """Helper to plot a single metric line."""
    ax.plot(
        x_data, y_data,
        linestyle=linestyle,
        linewidth=linewidth,
        color=color,
        marker=marker,
        markersize=markersize,
        markevery=markevery,
        label=label
    )


def plot_metric_with_std(
    ax,
    x_data,
    y_mean,
    y_std,
    color,
    label: str,
    linestyle: str = 'solid',
    linewidth: float = 2,
    alpha: float = 0.2
):
    """Helper to plot metric with shaded std deviation."""
    ax.plot(x_data, y_mean, linestyle=linestyle, linewidth=linewidth,
            color=color, label=label)
    ax.fill_between(x_data, y_mean - y_std, y_mean + y_std,
                    color=color, alpha=alpha)


def setup_subplot(ax, xlabel: str, ylabel: str, title: str):
    """Helper to setup subplot formatting."""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

def plot_smoothed_hms_metrics(
    smoothed_data: Dict[str, Dict[int, pd.DataFrame]],
    env_name_formatted: str,
    output_dir: Path,
    filename_suffix: str = ''
):
    """
    Plot smoothed HMS metrics (ACC and ER) for individual seeds.

    Parameters:
        smoothed_data: Nested dict {model_name: {seed: df}}
        env_name_formatted: Formatted environment name for titles
        output_dir: Directory to save the plot
        filename_suffix: Optional suffix for filename
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    base_colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    lightness_multipliers = [0.6, 0.8, 1.0, 1.2, 1.4]

    model_idx = 0
    for model_name, seeds_data in smoothed_data.items():
        base_color = base_colors[model_idx % len(base_colors)]
        num_seeds = len(seeds_data)

        seed_idx = 0
        for seed, df in sorted(seeds_data.items()):
            # Color and marker variation for multiple seeds
            if num_seeds > 1:
                lightness = lightness_multipliers[seed_idx % len(lightness_multipliers)]
                color = adjust_color_lightness(base_color, lightness)
                marker = markers[seed_idx % len(markers)]
                label_suffix = f' (S{seed})'
            else:
                color = base_color
                marker = 'o'
                label_suffix = ''

            markevery = max(1, len(df) // 20)

            # Plot ACC metrics
            plot_metric_line(axes[0], df['step'], df['ACC_mean_smoothed'],
                           color, f'{model_name} NG_ACC{label_suffix}',
                           linestyle='dotted', linewidth=2, marker=marker,
                           markersize=4, markevery=markevery)
            plot_metric_line(axes[0], df['step'], df['G_ACC_mean_smoothed'],
                           color, f'{model_name} G_ACC{label_suffix}',
                           linestyle='solid', linewidth=2.5, marker=marker,
                           markersize=5, markevery=markevery)

            # Plot ER metrics
            plot_metric_line(axes[1], df['step'], df['NG_ER_mean_smoothed'],
                           color, f'{model_name} NG_ER{label_suffix}',
                           linestyle='dotted', linewidth=2, marker=marker,
                           markersize=4, markevery=markevery)
            plot_metric_line(axes[1], df['step'], df['ER_mean_smoothed'],
                           color, f'{model_name} G_ER{label_suffix}',
                           linestyle='solid', linewidth=2.5, marker=marker,
                           markersize=5, markevery=markevery)

            seed_idx += 1
        model_idx += 1

    # Format subplots
    setup_subplot(axes[0], 'Training Step', 'Score',
                 f'Task Accuracy: per step - {env_name_formatted} (Smoothed)')
    setup_subplot(axes[1], 'Training Step', 'Score',
                 f'Exploit Ratio: per step - {env_name_formatted} (Smoothed)')

    plt.tight_layout()
    filename = f'hms_smoothed_{env_name_formatted}'
    if filename_suffix:
        filename += f'_{filename_suffix}'
    filename += '.png'

    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot to: {output_dir / filename}")


def plot_smoothed_hms_mean_std(
    smoothed_data: Dict[str, Dict[int, pd.DataFrame]],
    env_name_formatted: str,
    output_dir: Path,
    filename_suffix: str = ''
):
    """
    Plot mean ± std across seeds for smoothed HMS metrics.

    Parameters:
        smoothed_data: Nested dict {model_name: {seed: df}}
        env_name_formatted: Formatted environment name for titles
        output_dir: Directory to save the plot
        filename_suffix: Optional suffix for filename
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    base_colors = plt.cm.tab10.colors

    model_idx = 0
    for model_name, seeds_data in smoothed_data.items():
        base_color = base_colors[model_idx % len(base_colors)]
        dfs = [df for seed, df in sorted(seeds_data.items())]

        # Aggregate across seeds by step
        all_steps = sorted(set().union(*[set(df['step'].values) for df in dfs]))

        agg_data = {
            'step': [], 'ACC_mean': [], 'ACC_std': [],
            'G_ACC_mean': [], 'G_ACC_std': [],
            'NG_ER_mean': [], 'NG_ER_std': [],
            'ER_mean': [], 'ER_std': []
        }

        for step in all_steps:
            step_values = {
                'ACC_mean_smoothed': [],
                'G_ACC_mean_smoothed': [],
                'NG_ER_mean_smoothed': [],
                'ER_mean_smoothed': []
            }

            for df in dfs:
                step_data = df[df['step'] == step]
                if not step_data.empty:
                    for col in step_values.keys():
                        if col in step_data.columns:
                            step_values[col].append(step_data[col].values[0])

            if any(step_values.values()):
                agg_data['step'].append(step)
                agg_data['ACC_mean'].append(np.mean(step_values['ACC_mean_smoothed']))
                agg_data['ACC_std'].append(np.std(step_values['ACC_mean_smoothed']))
                agg_data['G_ACC_mean'].append(np.mean(step_values['G_ACC_mean_smoothed']))
                agg_data['G_ACC_std'].append(np.std(step_values['G_ACC_mean_smoothed']))
                agg_data['NG_ER_mean'].append(np.mean(step_values['NG_ER_mean_smoothed']))
                agg_data['NG_ER_std'].append(np.std(step_values['NG_ER_mean_smoothed']))
                agg_data['ER_mean'].append(np.mean(step_values['ER_mean_smoothed']))
                agg_data['ER_std'].append(np.std(step_values['ER_mean_smoothed']))

        # Convert to arrays
        steps = np.array(agg_data['step'])

        # Plot ACC metrics with std
        plot_metric_with_std(axes[0], steps, np.array(agg_data['ACC_mean']),
                           np.array(agg_data['ACC_std']), base_color,
                           f'{model_name} NG_ACC', linestyle='dotted', alpha=0.2)
        plot_metric_with_std(axes[0], steps, np.array(agg_data['G_ACC_mean']),
                           np.array(agg_data['G_ACC_std']), base_color,
                           f'{model_name} G_ACC', linestyle='solid', alpha=0.3)

        # Plot ER metrics with std
        plot_metric_with_std(axes[1], steps, np.array(agg_data['NG_ER_mean']),
                           np.array(agg_data['NG_ER_std']), base_color,
                           f'{model_name} NG_ER', linestyle='dotted', alpha=0.2)
        plot_metric_with_std(axes[1], steps, np.array(agg_data['ER_mean']),
                           np.array(agg_data['ER_std']), base_color,
                           f'{model_name} G_ER', linestyle='solid', alpha=0.3)

        model_idx += 1

    # Format subplots
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title(f'Task Accuracy: per step - {env_name_formatted} (Mean ± Std)',
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title(f'Exploit Ratio: per step - {env_name_formatted} (Mean ± Std)',
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    filename = f'hms_smoothed_mean_std_{env_name_formatted}'
    if filename_suffix:
        filename += f'_{filename_suffix}'
    filename += '.png'

    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot to: {output_dir / filename}")


def plot_smoothed_rewards(
    smoothed_data: Dict[str, Dict[int, pd.DataFrame]],
    env_name_formatted: str,
    output_dir: Path,
    filename_suffix: str = ''
):
    """
    Plot smoothed rewards for individual seeds.

    Parameters:
        smoothed_data: Nested dict {model_name: {seed: df}}
        env_name_formatted: Formatted environment name for titles
        output_dir: Directory to save the plot
        filename_suffix: Optional suffix for filename
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    base_colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    lightness_multipliers = [0.6, 0.8, 1.0, 1.2, 1.4]

    model_idx = 0
    for model_name, seeds_data in smoothed_data.items():
        base_color = base_colors[model_idx % len(base_colors)]
        num_seeds = len(seeds_data)

        seed_idx = 0
        for seed, df in sorted(seeds_data.items()):
            if 'reward_smoothed' not in df.columns:
                print(f"Warning: No reward data for {model_name}, seed {seed}")
                continue

            # Color and marker variation for multiple seeds
            if num_seeds > 1:
                lightness = lightness_multipliers[seed_idx % len(lightness_multipliers)]
                color = adjust_color_lightness(base_color, lightness)
                marker = markers[seed_idx % len(markers)]
                label = f'{model_name} (S{seed})'
            else:
                color = base_color
                marker = 'o'
                label = model_name

            markevery = max(1, len(df) // 20)

            plot_metric_line(ax, df['step'], df['reward_smoothed'],
                           color, label, linestyle='solid', linewidth=2,
                           marker=marker, markersize=4, markevery=markevery)

            seed_idx += 1
        model_idx += 1

    setup_subplot(ax, 'Training Step', 'Reward',
                 f'Reward: per step - {env_name_formatted} (Smoothed)')

    plt.tight_layout()
    filename = f'reward_smoothed_{env_name_formatted}'
    if filename_suffix:
        filename += f'_{filename_suffix}'
    filename += '.png'

    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot to: {output_dir / filename}")


def plot_smoothed_rewards_mean_std(
    smoothed_data: Dict[str, Dict[int, pd.DataFrame]],
    env_name_formatted: str,
    output_dir: Path,
    filename_suffix: str = '',
    figsize: Tuple[int, int] = (10, 5),

):
    """
    Plot mean ± std across seeds for smoothed rewards.

    Parameters:
        smoothed_data: Nested dict {model_name: {seed: df}}
        env_name_formatted: Formatted environment name for titles
        output_dir: Directory to save the plot
        filename_suffix: Optional suffix for filename
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    base_colors = plt.cm.tab10.colors

    model_idx = 0
    for model_name, seeds_data in smoothed_data.items():
        base_color = base_colors[model_idx % len(base_colors)]
        dfs = [df for seed, df in sorted(seeds_data.items())]

        # Check if reward data exists
        if not any('reward_smoothed' in df.columns for df in dfs):
            print(f"Warning: No reward data for {model_name}")
            continue

        # Aggregate across seeds by step
        all_steps = sorted(set().union(*[set(df['step'].values) for df in dfs]))

        reward_mean_list = []
        reward_std_list = []
        steps_list = []

        for step in all_steps:
            step_rewards = []
            for df in dfs:
                if 'reward_smoothed' not in df.columns:
                    continue
                step_data = df[df['step'] == step]
                if not step_data.empty:
                    step_rewards.append(step_data['reward_smoothed'].values[0])

            if step_rewards:
                steps_list.append(step)
                reward_mean_list.append(np.mean(step_rewards))
                reward_std_list.append(np.std(step_rewards))

        steps = np.array(steps_list)
        reward_mean = np.array(reward_mean_list)
        reward_std = np.array(reward_std_list)

        plot_metric_with_std(ax, steps, reward_mean, reward_std,
                           base_color, model_name, alpha=0.3)

        model_idx += 1

    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_ylabel('Reward', fontsize=13)
    ax.set_title(f'Reward: per step - {env_name_formatted} (Mean ± Std)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # ax.legend(fontsize=10, loc='best')
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15),
            ncol=5, frameon=True, fancybox=True)

    plt.tight_layout()
    filename = f'reward_smoothed_mean_std_{env_name_formatted}'
    if filename_suffix:
        filename += f'_{filename_suffix}'
    filename += '.pdf'

    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot to: {output_dir / filename}")


def plot_smoothed_gaps(
    smoothed_data: Dict[str, Dict[int, pd.DataFrame]],
    env_name_formatted: str,
    output_dir: Path,
    filename_suffix: str = ''
):
    """
    Plot smoothed gap metrics (ACC_gap and ER_gap) for individual seeds.

    ACC_gap = ACC_mean - G_ACC_mean (NG performance - G performance)
    ER_gap = ER_mean - NG_ER_mean (G exploitation - NG exploitation)

    Parameters:
        smoothed_data: Nested dict {model_name: {seed: df}}
        env_name_formatted: Formatted environment name for titles
        output_dir: Directory to save the plot
        filename_suffix: Optional suffix for filename
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    base_colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    lightness_multipliers = [0.6, 0.8, 1.0, 1.2, 1.4]

    model_idx = 0
    for model_name, seeds_data in smoothed_data.items():
        base_color = base_colors[model_idx % len(base_colors)]
        num_seeds = len(seeds_data)

        seed_idx = 0
        for seed, df in sorted(seeds_data.items()):
            # Color and marker variation for multiple seeds
            if num_seeds > 1:
                lightness = lightness_multipliers[seed_idx % len(lightness_multipliers)]
                color = adjust_color_lightness(base_color, lightness)
                marker = markers[seed_idx % len(markers)]
                label_suffix = f' (S{seed})'
            else:
                color = base_color
                marker = 'o'
                label_suffix = ''

            markevery = max(1, len(df) // 20)

            # Plot ACC_gap
            plot_metric_line(axes[0], df['step'], df['ACC_gap_smoothed'],
                           color, f'{model_name}{label_suffix}',
                           linestyle='solid', linewidth=2.5, marker=marker,
                           markersize=5, markevery=markevery)

            # Plot ER_gap
            plot_metric_line(axes[1], df['step'], df['ER_gap_smoothed'],
                           color, f'{model_name}{label_suffix}',
                           linestyle='solid', linewidth=2.5, marker=marker,
                           markersize=5, markevery=markevery)

            seed_idx += 1
        model_idx += 1

    # Format subplots
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('ACC Gap (NG - G)', fontsize=12)
    axes[0].set_title(f'Task Accuracy Gap: per step - {env_name_formatted} (Smoothed)',
                     fontsize=14, fontweight='bold')
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))

    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('ER Gap (G - NG)', fontsize=12)
    axes[1].set_title(f'Exploit Ratio Gap: per step - {env_name_formatted} (Smoothed)',
                     fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    filename = f'gaps_smoothed_{env_name_formatted}'
    if filename_suffix:
        filename += f'_{filename_suffix}'
    filename += '.png'

    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot to: {output_dir / filename}")


def plot_smoothed_gaps_mean_std(
    smoothed_data: Dict[str, Dict[int, pd.DataFrame]],
    env_name_formatted: str,
    output_dir: Path,
    filename_suffix: str = '',
    figsize: Tuple[int, int] = (10, 8),
    plot_metrics: List[str] = ['ACC', 'ER']
):
    """
    Plot mean ± std across seeds for smoothed gap metrics.

    ACC_gap = ACC_mean - G_ACC_mean (NG performance - G performance)
    ER_gap = ER_mean - NG_ER_mean (G exploitation - NG exploitation)

    Parameters:
        smoothed_data: Nested dict {model_name: {seed: df}}
        env_name_formatted: Formatted environment name for titles
        output_dir: Directory to save the plot
        filename_suffix: Optional suffix for filename
    """
    fig, axes = plt.subplots(len(plot_metrics), 1, figsize=figsize)
    if len(plot_metrics) == 1:
        axes = [axes]
    base_colors = plt.cm.tab10.colors

    # Use compact legend layout when there are few models (e.g. single model family)
    compact_legend = len(smoothed_data) <= 5
    model_idx = 0
    for model_name, seeds_data in smoothed_data.items():
        base_color = base_colors[model_idx % len(base_colors)]
        dfs = [df for seed, df in sorted(seeds_data.items())]

        # Aggregate across seeds by step
        all_steps = sorted(set().union(*[set(df['step'].values) for df in dfs]))

        acc_gap_mean_list = []
        acc_gap_std_list = []
        er_gap_mean_list = []
        er_gap_std_list = []
        steps_list = []

        for step in all_steps:
            acc_gap_values = []
            er_gap_values = []

            for df in dfs:
                step_data = df[df['step'] == step]
                if not step_data.empty:
                    if 'ACC_gap_smoothed' in step_data.columns:
                        acc_gap_values.append(step_data['ACC_gap_smoothed'].values[0])
                    if 'ER_gap_smoothed' in step_data.columns:
                        er_gap_values.append(step_data['ER_gap_smoothed'].values[0])

            if acc_gap_values or er_gap_values:
                steps_list.append(step)
                acc_gap_mean_list.append(np.mean(acc_gap_values) if acc_gap_values else np.nan)
                acc_gap_std_list.append(np.std(acc_gap_values) if acc_gap_values else 0)
                er_gap_mean_list.append(np.mean(er_gap_values) if er_gap_values else np.nan)
                er_gap_std_list.append(np.std(er_gap_values) if er_gap_values else 0)

        steps = np.array(steps_list)
        acc_gap_mean = np.array(acc_gap_mean_list)
        acc_gap_std = np.array(acc_gap_std_list)
        er_gap_mean = np.array(er_gap_mean_list)
        er_gap_std = np.array(er_gap_std_list)

        # Plot ACC_gap with std
        if 'ACC' in plot_metrics:
            plot_metric_with_std(axes[0], steps, acc_gap_mean, acc_gap_std,
                            base_color, model_name, linestyle='solid', alpha=0.3)

        # Plot ER_gap with std
        if 'ER' in plot_metrics:
            plot_metric_with_std(axes[len(plot_metrics) - 1], steps, er_gap_mean, er_gap_std,
                           base_color, model_name, linestyle='solid', alpha=0.3)

        model_idx += 1

    # Format subplots based on plot_metrics
    for idx, ax in enumerate(axes):
        ax.set_xlabel('Training Step', fontsize=13)
        use_acc = 'ACC' in plot_metrics and idx == 0
        ax.set_ylabel('ACC Gap ' if use_acc else 'HEX Gap', fontsize=13)
        
        if use_acc:
            ax.set_title(f'ACC Gap: per step - {env_name_formatted} (Mean ± Std)',
                         fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'HEX Gap: per step - {env_name_formatted} (Mean ± Std)',
                         fontsize=14, fontweight='bold')
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        if compact_legend:
            ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    ncol=5, frameon=True, fancybox=True)
        else:
            # put legend to the right and ncols=1
            ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    env_initials = ''.join([word[0] for word in env_name_formatted.split()]).upper()
    filename = f'gaps_mean_{env_initials}'
    if filename_suffix:
        filename += f'_{filename_suffix}'
    filename += '.pdf'

    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot to: {output_dir / filename}")


# ============================================================================
# BACKWARD COMPATIBILITY RE-EXPORTS
# ============================================================================
# These functions have been moved to dedicated modules but are re-exported here
# so that existing imports (e.g. `from .hms_plotting import *`) continue to work.

from .hms_multi_env_plotting import (
    collect_multi_env_correlations,
    plot_multi_env_scatter_grid,
    plot_multi_env_scatter_grid_v2,
    plot_multi_env_scatter_single_col,
)
from .hms_comparison_plotting import vis_correlations, plot_base_vs_instruct_paired_bars


