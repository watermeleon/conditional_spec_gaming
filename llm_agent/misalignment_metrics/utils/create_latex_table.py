
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from scipy.stats import spearmanr, pearsonr

BENCHMARK_CATEGORY_MAP = {
    "machiavelli": "Ethics",
    "sycophancy": "Ethics",
    "human_jailbreak": "Jailbreaks",
    "tap": "Jailbreaks",
    "gcc": "Jailbreaks",
    "bbq": "Bias",
    "crows_pair": "Bias",
    "discrim_eval": "Bias",
    "rmsce_mmlu": "Calibration",
    "capability_score": "Model Properties",
    "num_params": "Model Properties",
}


def generate_latex_correlation_table(
    env_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    benchmark_metrics: List[str],
    corr_metric: str = 'max_reward',
    caption: Optional[str] = None,
    label: Optional[str] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    category_map: Optional[Dict[str, str]] = None,
    ordered_categories: Optional[List[str]] = None,
) -> str:
    """
    Generate a colored LaTeX correlation table.

    Rows = benchmark metrics, columns = environments.
    Cells show Spearman rho with significance stars (* p<0.05, ** p<0.01).
    Cells are shaded using a diverging colormap (blue-white-red), capped at 70%
    intensity so text remains readable. Significant entries are bold.
    Metrics are grouped by category with alternating light-grey row shading.

    Parameters:
        env_results: Dict from collect_multi_env_correlations,
                     mapping env_name -> (corr_df, merged_data)
        benchmark_metrics: List of benchmark metric names for the rows
        corr_metric: The target metric that was correlated against (for caption)
        caption: Optional LaTeX caption (auto-generated if None)
        label: Optional LaTeX label (auto-generated if None)
        vmin, vmax: Range for the colormap normalization
        category_map: Optional dict mapping metric -> category name.
                      Defaults to BENCHMARK_CATEGORY_MAP.
        ordered_categories: Optional list specifying the order of categories.
                      e.g. ["Model Properties", "Ethics", "Jailbreaks", "Bias", "Calibration"]
                      If None, categories appear in the order they are first encountered
                      in benchmark_metrics.

    Returns:
        LaTeX string for the table (also printed to stdout)
    """
    if category_map is None:
        category_map = BENCHMARK_CATEGORY_MAP

    env_names = list(env_results.keys())

    # Build a lookup: (env, benchmark_metric) -> (rho, pval)
    corr_lookup = {}
    for env_name in env_names:
        corr_df, _ = env_results[env_name]
        for _, row in corr_df.iterrows():
            corr_lookup[(env_name, row['benchmark_metric'])] = (
                row['spearman_r'], row['spearman_p']
            )

    # Diverging colormap capped at 70% intensity for readability
    cmap = plt.cm.RdBu_r
    norm_fn = plt.Normalize(vmin=vmin, vmax=vmax)

    def rho_to_latex_color(rho):
        """Convert rho to an RGB LaTeX \\cellcolor string, capped at 70% intensity."""
        rgba = cmap(norm_fn(rho))
        # Scale distance from white: extremes reach 70% of full saturation
        max_intensity = 0.70
        r = int((1.0 - max_intensity * (1.0 - rgba[0])) * 255)
        g = int((1.0 - max_intensity * (1.0 - rgba[1])) * 255)
        b = int((1.0 - max_intensity * (1.0 - rgba[2])) * 255)
        return f"\\cellcolor[RGB]{{{r},{g},{b}}}"

    # Group benchmark metrics by category
    # First, collect which metrics belong to which category
    cat_to_metrics = {}
    for bm in benchmark_metrics:
        cat = category_map.get(bm, "Other")
        if cat not in cat_to_metrics:
            cat_to_metrics[cat] = []
        cat_to_metrics[cat].append(bm)

    # Determine group ordering
    if ordered_categories is not None:
        # Use the specified order; append any categories not listed at the end
        group_order = list(ordered_categories)
        for cat in cat_to_metrics:
            if cat not in group_order:
                group_order.append(cat)
    else:
        # Preserve first-encountered order from benchmark_metrics
        group_order = list(cat_to_metrics.keys())

    ordered_groups = [(cat, cat_to_metrics[cat]) for cat in group_order if cat in cat_to_metrics]

    # Build LaTeX
    env_headers = [e.replace("_", " ").title() for e in env_names]
    n_envs = len(env_names)

    if caption is None:
        caption = (
            f"Spearman correlations between benchmark metrics and "
            f"\\texttt{{{corr_metric.replace('_', chr(92) + '_')}}} across environments."
        )
    if label is None:
        label = f"tab:corr_{corr_metric}"

    lines = []
    lines.append("\\begin{table*}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * n_envs + "}")
    lines.append("\\toprule")
    header = "Benchmark Metric & " + " & ".join(env_headers) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for group_idx, (cat_name, group_metrics) in enumerate(ordered_groups):
        # Alternating group shading: even groups get lightgray background
        use_row_bg = (group_idx % 2 == 0)
        row_bg = "\\rowcolor[gray]{0.92}" if use_row_bg else ""

        for bm in group_metrics:
            bm_display = bm.replace("_", " ").title()
            cells = [bm_display]

            for env_name in env_names:
                key = (env_name, bm)
                if key in corr_lookup:
                    rho, pval = corr_lookup[key]
                    stars = '**' if pval < 0.01 else ('*' if pval < 0.05 else '')
                    color_cmd = rho_to_latex_color(rho)
                    text = f"{rho:+.2f}{stars}"
                    if pval < 0.05:
                        text = f"\\textbf{{{text}}}"
                    cells.append(f"{color_cmd} {text}")
                else:
                    cells.append("--")

            row_line = " & ".join(cells) + " \\\\"
            if use_row_bg:
                row_line = row_bg + " " + row_line
            lines.append(row_line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")    
    lines.append("\\end{table*}")

    latex_str = "\n".join(lines)
    print(latex_str)
    return latex_str

def generate_latex_combined_correlation_table(
    multi_corr_results,
    benchmark_metrics,
    caption=None,
    label=None,
    vmin=-1.0,
    vmax=1.0,
    category_map=None,
    ordered_categories=None,
    corr_metric_display_names=None,
):
    if category_map is None:
        category_map = BENCHMARK_CATEGORY_MAP

    env_names_short = {
        'therapy_talk': 'TT',
        'action_advice': 'AA',
        'political_qa': 'PQA',
    }

    corr_metrics = list(multi_corr_results.keys())

    first_env_results = multi_corr_results[corr_metrics[0]]
    env_names = list(first_env_results.keys())
    # env_headers = [e.replace("_", " ").title() for e in env_names]
    env_headers = [env_names_short.get(e, e.replace("_", " ").title()) for e in env_names]
    n_envs = len(env_names)

    # Build lookups: (corr_metric, env, bm) -> (rho, pval)
    corr_lookup = {}
    for cm, env_results in multi_corr_results.items():
        for env_name in env_names:
            if env_name not in env_results:
                continue
            corr_df, _ = env_results[env_name]
            for _, row in corr_df.iterrows():
                corr_lookup[(cm, env_name, row['benchmark_metric'])] = (
                    row['spearman_r'], row['spearman_p']
                )

    cmap = plt.cm.RdBu_r
    norm_fn = plt.Normalize(vmin=vmin, vmax=vmax)

    def rho_to_latex_color(rho):
        rgba = cmap(norm_fn(rho))
        max_intensity = 0.70
        r = int((1.0 - max_intensity * (1.0 - rgba[0])) * 255)
        g = int((1.0 - max_intensity * (1.0 - rgba[1])) * 255)
        b = int((1.0 - max_intensity * (1.0 - rgba[2])) * 255)
        return f"\\cellcolor[RGB]{{{r},{g},{b}}}"

    # Group benchmark metrics by category
    cat_to_metrics = {}
    for bm in benchmark_metrics:
        cat = category_map.get(bm, "Other")
        if cat not in cat_to_metrics:
            cat_to_metrics[cat] = []
        cat_to_metrics[cat].append(bm)

    if ordered_categories is not None:
        group_order = list(ordered_categories)
        for cat in cat_to_metrics:
            if cat not in group_order:
                group_order.append(cat)
    else:
        group_order = list(cat_to_metrics.keys())

    ordered_groups = [
        (cat, cat_to_metrics[cat]) for cat in group_order if cat in cat_to_metrics
    ]

    # Display names for corr metrics
    cm_displays = {}
    for cm in corr_metrics:
        if corr_metric_display_names and cm in corr_metric_display_names:
            cm_displays[cm] = corr_metric_display_names[cm]
        else:
            cm_displays[cm] = cm.replace("_", " ").title()

    # Caption / label defaults
    if caption is None:
        cm_list = ", ".join(
            f"\\texttt{{{cm.replace('_', chr(92) + '_')}}}" for cm in corr_metrics
        )
        caption = f"Spearman correlations between benchmark metrics and {cm_list} across environments."
    if label is None:
        label = "tab:corr_combined_" + "_".join(corr_metrics)

    # ---- Build LaTeX ----
    lines = []
    lines.append("\\begin{table*}[ht]")
    lines.append("\\centering")
    lines.append("\\small")

    # Column spec: l | ccc | ccc  (one group of c's per corr_metric)
    col_groups = " | ".join(["c" * n_envs for _ in corr_metrics])
    lines.append(f"\\begin{{tabular}}{{l | {col_groups}}}")
    lines.append("\\toprule")

    # --- Header row 1: corr_metric names spanning their env columns ---
    header1_parts = [""]  # empty cell for benchmark metric column
    for cm in corr_metrics:
        header1_parts.append(
            f"\\multicolumn{{{n_envs}}}{{c}}"
            f"{{\\textbf{{\\textit{{{cm_displays[cm]}}}}}}}"
        )
    lines.append(" & ".join(header1_parts) + " \\\\")

    # cmidrules under each corr_metric group
    col_idx = 2  # 1-indexed; col 1 is the metric name
    cmidrules = []
    for _ in corr_metrics:
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_envs - 1}}}")
        col_idx += n_envs
    lines.append(" ".join(cmidrules))

    # --- Header row 2: env names repeated under each corr_metric ---
    header2_parts = ["Benchmark Metric"]
    for _ in corr_metrics:
        header2_parts.extend(env_headers)
    lines.append(" & ".join(header2_parts) + " \\\\")
    lines.append("\\midrule")

    # --- Data rows grouped by category ---
    for group_idx, (cat_name, group_metrics) in enumerate(ordered_groups):
        use_row_bg = (group_idx % 2 == 0)
        row_bg = "\\rowcolor[gray]{0.92}" if use_row_bg else ""

        for bm in group_metrics:
            bm_display = bm.replace("_", " ").title()
            cells = [bm_display]

            # For each corr_metric block, add its env columns
            for cm in corr_metrics:
                for env_name in env_names:
                    key = (cm, env_name, bm)
                    if key in corr_lookup:
                        rho, pval = corr_lookup[key]
                        stars = '**' if pval < 0.01 else ('*' if pval < 0.05 else '')
                        color_cmd = rho_to_latex_color(rho)
                        text = f"{rho:+.2f}{stars}"
                        if pval < 0.05:
                            text = f"\\textbf{{{text}}}"
                        cells.append(f"{color_cmd} {text}")
                    else:
                        cells.append("--")

            row_line = " & ".join(cells) + " \\\\"
            if use_row_bg:
                row_line = row_bg + " " + row_line
            lines.append(row_line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")

    latex_str = "\n".join(lines)
    print(latex_str)
    return latex_str
