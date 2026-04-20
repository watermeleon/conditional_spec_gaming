"""
Comparison and specialized plots for HMS analysis.

Contains:
- vis_correlations: Scatter plots of top N benchmark correlations
- plot_base_vs_instruct_paired_bars: Paired bar chart comparing base vs instruct models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


def vis_correlations(
    corr_results: pd.DataFrame,
    merged_data: pd.DataFrame,
    max_n: int = 6,
    corr_metric: str = 'hms_score',
    sort_by: str = 'spearman_r',
    output_dir: Optional[Path] = None,
    subset_benchmark_metrics: Optional[List[str]] = None,
):
    """
    Visualize the top N correlations as scatter plots.

    Parameters:
        corr_results: DataFrame with correlation results
        merged_data: Merged DataFrame with model scores
        max_n: Maximum number of scatter plots to show
        corr_metric: The HMS metric on the y-axis
        sort_by: Which correlation column to sort/display by
        output_dir: Optional directory to save the figure
    """
    if subset_benchmark_metrics is not None:
        corr_results = corr_results[corr_results['benchmark_metric'].isin(subset_benchmark_metrics)]

    sortby_metric = sort_by.split("_")[0]
    top_n = min(max_n, len(corr_results))
    top_metrics = corr_results.head(top_n)
    corr_metric_format = corr_metric.replace("_", " ").title()

    if top_n > 3 and top_n <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    elif top_n <= 3:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_metrics.iterrows()):
        if idx >= top_n:
            break

        ax = axes[idx]
        metric = row['benchmark_metric']

        valid_mask = merged_data[[corr_metric, metric]].notna().all(axis=1)
        x = merged_data.loc[valid_mask, metric]
        y = merged_data.loc[valid_mask, corr_metric]

        ax.scatter(x, y, s=100, alpha=0.6, edgecolors='black', linewidth=1)

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(x)
        ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel(metric, fontsize=10)
        ax.set_ylabel(corr_metric_format, fontsize=10)
        ax.set_title(
            f"{metric}\n {sortby_metric}: \u03c1={row[f'{sortby_metric}_r']:.3f} "
            f"(p={row[f'{sortby_metric}_p']:.3f})",
            fontsize=10, fontweight='bold',
        )
        ax.grid(alpha=0.3)

    for idx in range(top_n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / f'{corr_metric}_benchmark_correlations.png',
                    dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_dir / f'{corr_metric}_benchmark_correlations.png'}")
    plt.show()


def plot_base_vs_instruct_paired_bars(
    df_chat,
    df_base,
    sw_metrics,
    metric_name='max_er_gap',
    figsize=(7, 4.5),
    output_path=None,
    base_to_chat=None,
    y_label=None,
    title=None,
    ylim=None,
    annotate_delta=True,
    show_values=True,
    show_std=False,
    sort_by='num_params',
    group_gap=0.55,
    font_scale=1.0,
):
    """
    Paired bar chart comparing base model vs instruction-tuned variant.

    Groups models by size (Small/Medium/Large) with colored background spans.
    Shows delta annotations between paired bars.

    Parameters:
        df_chat: DataFrame with instruct-tuned model metrics (must have 'model' column)
        df_base: DataFrame with base model metrics
        sw_metrics: DataFrame with benchmark metrics (needs 'num_params' column)
        metric_name: Column name for the metric to plot
        base_to_chat: Dict mapping base model names to their instruct counterparts
        y_label: Custom y-axis label
        title: Custom plot title
        ylim: (min, max) y-axis limits
        annotate_delta: Show delta annotations between bar pairs
        show_values: Show value labels above bars
        show_std: Show error bars
        sort_by: Sort pairs by 'num_params' or 'label'
        group_gap: Extra spacing between size groups
        font_scale: Scale factor for all font sizes
    """
    font_sizes = {
        'ylabel':     11,
        'xticks':      9,
        'title':      12,
        'bar_values':  9,
        'delta':       8.5,
        'legend':     10,
        'group_label': 8.5,
    }
    fs = {k: v * font_scale for k, v in font_sizes.items()}

    if base_to_chat is None:
        base_to_chat = {
            'Qwen1_5_7B': 'Qwen1_5_7B_Chat',
            'Qwen1_5_14B': 'Qwen1_5_14B_Chat',
            'Qwen1_5_1_8B': 'Qwen1_5_1_8B_Chat',
            'Llama_2_7b_hf': 'Llama_2_7b_chat_hf',
            'Llama_2_13b_hf': 'Llama_2_13b_chat_hf',
            'gemma_2b': 'gemma_1_1_2b_it',
        }

    model_name_recast = {
        'Qwen1_5': 'Qwen1.5',
        '_1_8B': '-1.8B',
        '_0_5B': '-0.5B',
        'gemma_1_1': 'gemma1.1',
        'Meta_Llama_3': 'Llama3',
        'Llama_2': 'Llama2',
        '_hf': '',
        '_': '-',
        '-13b': '\n13B',
        '-14B': '\n14B',
        '-7b': '\n7B',
        '-7B': '\n7B',
        '-1.8B': '\n1.8B',
        '-2b': '\n2B',
    }

    def recast_model_name(name):
        for old, new in model_name_recast.items():
            name = name.replace(old, new)
        return name

    def size_group(n):
        if n < 5:
            return 'Small'
        elif n < 10:
            return 'Medium'
        return 'Large'

    std_col = f'{metric_name}_std'
    params_map = sw_metrics.set_index('model')['num_params'].to_dict()
    df_chat_lookup = df_chat.set_index('model')
    df_base_lookup = df_base.set_index('model')

    paired_rows = []
    for base_model, chat_model in base_to_chat.items():
        if base_model not in df_base_lookup.index or chat_model not in df_chat_lookup.index:
            continue

        chat_value = df_chat_lookup.at[chat_model, metric_name]
        base_value = df_base_lookup.at[base_model, metric_name]
        num_params = params_map.get(chat_model, np.nan)

        if pd.isna(chat_value) or pd.isna(base_value) or pd.isna(num_params):
            continue

        chat_std = df_chat_lookup.at[chat_model, std_col] if std_col in df_chat_lookup.columns else np.nan
        base_std = df_base_lookup.at[base_model, std_col] if std_col in df_base_lookup.columns else np.nan

        paired_rows.append(
            {
                'label': recast_model_name(base_model),
                'num_params': float(num_params),
                'chat_model': chat_model,
                'base_model': base_model,
                'chat_value': float(chat_value),
                'base_value': float(base_value),
                'chat_std': float(chat_std) if not pd.isna(chat_std) else 0.0,
                'base_std': float(base_std) if not pd.isna(base_std) else 0.0,
            }
        )

    if not paired_rows:
        raise ValueError(
            f'No matched base/instruct pairs found for metric {metric_name!r}. '
            'Check the mapping and dataframe contents.'
        )

    df_pairs = pd.DataFrame(paired_rows)
    if sort_by == 'label':
        df_pairs = df_pairs.sort_values(['label', 'num_params']).reset_index(drop=True)
    else:
        df_pairs = df_pairs.sort_values(['num_params', 'label']).reset_index(drop=True)

    labels = df_pairs['label'].tolist()
    chat_vals = df_pairs['chat_value'].tolist()
    base_vals = df_pairs['base_value'].tolist()
    chat_stds = df_pairs['chat_std'].tolist()
    base_stds = df_pairs['base_std'].tolist()
    groups = [size_group(p) for p in df_pairs['num_params']]

    # Build x positions with extra gap between size groups
    x_positions = []
    group_spans = {}
    current_x = 0.0
    prev_group = None
    for i, g in enumerate(groups):
        if prev_group is not None and g != prev_group:
            current_x += group_gap
        x_positions.append(current_x)
        if g not in group_spans:
            group_spans[g] = [current_x, current_x]
        else:
            group_spans[g][1] = current_x
        current_x += 1.0
        prev_group = g
    x = np.array(x_positions)

    width = 0.36

    fig, ax = plt.subplots(figsize=figsize)

    instruct_color = '#630062'
    base_color = "#ee9906"
    err_kw = dict(elinewidth=1.2, capsize=3, capthick=1.2, ecolor='#444444')

    bars_chat = ax.bar(
        x - width / 2,
        chat_vals,
        width,
        yerr=chat_stds if show_std else None,
        error_kw=err_kw,
        label='Instruct-tuned',
        color=instruct_color,
        alpha=0.9,
        edgecolor='white',
        linewidth=0.6,
    )
    bars_base = ax.bar(
        x + width / 2,
        base_vals,
        width,
        yerr=base_stds if show_std else None,
        error_kw=err_kw,
        label='Base',
        color=base_color,
        alpha=0.9,
        edgecolor='white',
        linewidth=0.6,
    )

    all_values = chat_vals + base_vals
    value_range = max(all_values) - min(all_values) if len(all_values) > 1 else max(all_values[0], 1.0)
    text_offset = max(0.03 * value_range, 0.03)

    if show_values:
        for bar in bars_chat:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + text_offset,
                f"{bar.get_height():.2f}",
                ha='center',
                va='bottom',
                fontsize=fs['bar_values'],
                color=instruct_color,
            )
        for bar in bars_base:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + text_offset,
                f"{bar.get_height():.2f}",
                ha='center',
                va='bottom',
                fontsize=fs['bar_values'],
                color=base_color,
            )

    if annotate_delta:
        for idx, row in df_pairs.iterrows():
            y_mid = (row['chat_value'] + row['base_value']) / 2
            delta = row['chat_value'] - row['base_value']
            ax.text(
                x[idx],
                y_mid,
                f"\u0394={delta:+.2f}",
                ha='center',
                va='center',
                fontsize=fs['delta'],
                color='dimgray',
                bbox={'boxstyle': 'round,pad=0.18', 'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
            )

    metric_label = y_label or metric_name.replace('_', ' ').title()
    ax.set_ylabel(metric_label, fontsize=fs['ylabel'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs['xticks'])
    ax.set_title(
        title or f'{metric_label}: Base vs Instruct Matched Pairs',
        fontsize=fs['title'],
        fontweight='bold',
    )

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        upper = max(all_values) + 3 * text_offset
        lower = min(0, min(all_values) - text_offset)
        ax.set_ylim(lower, upper)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.45)
    ax.legend(frameon=False, fontsize=fs['legend'], loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Shade size groups and add group labels
    y_lo, y_hi = ax.get_ylim()
    half_bar = 0.5 + width / 2 + 0.05
    for g, (xl, xr) in group_spans.items():
        ax.axvspan(xl - half_bar, xr + half_bar, alpha=0.05, color='steelblue', zorder=0)
        ax.text(
            (xl + xr) / 2, y_hi,
            g,
            ha='center', va='top',
            fontsize=fs['group_label'], color='steelblue', style='italic',
        )

    # Vertical separators between groups
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            sep_x = (x[i - 1] + x[i]) / 2
            ax.axvline(sep_x, color='lightgray', linewidth=1.0, linestyle='--', alpha=0.8, zorder=1)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved to: {output_path}')

    plt.show()
    return fig, ax, df_pairs
