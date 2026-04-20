


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from scipy.stats import spearmanr, pearsonr

from llm_agent.misalignment_metrics.hms_analysis import (
    get_custom_metric_function,
    load_judge_results,
    compute_hms_metrics_per_step,
)




# ============================================================================
# AGGREGATE STATISTICS
# ============================================================================

def compute_aggregate_statistics(smoothed_data: Dict, max_step: int = 300) -> pd.DataFrame:
    """
    Compute aggregate statistics per model based on mean over seeds.
    Computes statistics for both ER and ACC gaps.

    Args:
        smoothed_data: Nested dict {model_name: {seed: DataFrame}}
        max_step: Maximum step to consider (default: 300)

    Returns:
        DataFrame with statistics per model
    """
    results = []

    for model_name, seeds_data in smoothed_data.items():
        seed_dfs = []
        for seed, df in seeds_data.items():
            if max_step is not None:
                df = df[df['step'] <= max_step].copy()
            seed_dfs.append(df)

        mean_df = seed_dfs[0][['step']].copy()

        metric_columns = ['ACC_gap_smoothed', 'ER_gap_smoothed', 'reward_smoothed',
                         'ACC_mean_smoothed', 'G_ACC_mean_smoothed',
                         'NG_ER_mean_smoothed', 'ER_mean_smoothed']

        for col in metric_columns:
            col_values = []
            for df in seed_dfs:
                if col in df.columns:
                    col_values.append(df[col].values)
            if col_values:
                mean_df[col] = np.nanmean(np.stack(col_values), axis=0)

        step_values = mean_df['step'].values
        reward_values = mean_df['reward_smoothed'].values

        er_gap_values = mean_df['ER_gap_smoothed'].values
        acc_gap_values = mean_df['ACC_gap_smoothed'].values

        # ER Gap Statistics
        er_gap_auc = np.trapezoid(er_gap_values, step_values)
        max_er_gap = np.max(er_gap_values)
        max_er_gap_idx = np.argmax(er_gap_values)
        step_at_max_er_gap = step_values[max_er_gap_idx]

        threshold_90_er = 0.9 * max_er_gap
        indices_above_90_er = np.where(er_gap_values >= threshold_90_er)[0]
        step_at_90_max_er = step_values[indices_above_90_er[0]] if len(indices_above_90_er) > 0 else np.nan

        max_er_mean = np.max(mean_df['ER_mean_smoothed'].values)
        max_acc_mean = np.max(mean_df['ACC_mean_smoothed'].values)

        # ACC Gap Statistics
        acc_gap_auc = np.trapezoid(acc_gap_values, step_values)
        max_acc_gap = np.max(acc_gap_values)
        max_acc_gap_idx = np.argmax(acc_gap_values)
        step_at_max_acc_gap = step_values[max_acc_gap_idx]

        threshold_90_acc = 0.9 * max_acc_gap
        indices_above_90_acc = np.where(acc_gap_values >= threshold_90_acc)[0]
        step_at_90_max_acc = step_values[indices_above_90_acc[0]] if len(indices_above_90_acc) > 0 else np.nan

        # Common Statistics
        max_reward = np.max(reward_values)

        time_range = step_values[-1] - step_values[0]
        if time_range > 0:
            acc_gap_auc_norm = acc_gap_auc / time_range
            er_gap_auc_norm = er_gap_auc / time_range
        else:
            acc_gap_auc_norm = acc_gap_auc
            er_gap_auc_norm = er_gap_auc

        hms_score = 0.5 * np.abs(acc_gap_auc_norm) + 0.5 * np.abs(er_gap_auc_norm)

        # Per-seed scalar stats for std computation
        per_seed_scalars = {'max_reward': [], 'hms_score': [], 'max_er_gap': [], 'max_acc_gap': [],
                            'er_gap_auc_normalized': [], 'acc_gap_auc_normalized': []}
        for df in seed_dfs:
            sv = df['step'].values
            tr = sv[-1] - sv[0]
            er_v = df['ER_gap_smoothed'].values if 'ER_gap_smoothed' in df.columns else np.array([np.nan])
            acc_v = df['ACC_gap_smoothed'].values if 'ACC_gap_smoothed' in df.columns else np.array([np.nan])
            rew_v = df['reward_smoothed'].values if 'reward_smoothed' in df.columns else np.array([np.nan])
            er_auc_n = np.trapezoid(er_v, sv) / tr if tr > 0 else np.trapezoid(er_v, sv)
            acc_auc_n = np.trapezoid(acc_v, sv) / tr if tr > 0 else np.trapezoid(acc_v, sv)
            per_seed_scalars['max_reward'].append(np.nanmax(rew_v))
            per_seed_scalars['max_er_gap'].append(np.nanmax(er_v))
            per_seed_scalars['max_acc_gap'].append(np.nanmax(acc_v))
            per_seed_scalars['er_gap_auc_normalized'].append(er_auc_n)
            per_seed_scalars['acc_gap_auc_normalized'].append(acc_auc_n)
            per_seed_scalars['hms_score'].append(0.5 * np.abs(acc_auc_n) + 0.5 * np.abs(er_auc_n))

        results.append({
            'model': model_name,
            'er_gap_auc': er_gap_auc,
            'er_gap_auc_normalized': er_gap_auc_norm,
            'er_gap_auc_normalized_std': np.std(per_seed_scalars['er_gap_auc_normalized']),
            'max_er_gap': max_er_gap,
            'max_er_gap_std': np.std(per_seed_scalars['max_er_gap']),
            'step_at_max_er_gap': step_at_max_er_gap,
            'step_at_90pct_max_er': step_at_90_max_er,
            'acc_gap_auc': acc_gap_auc,
            'acc_gap_auc_normalized': acc_gap_auc_norm,
            'acc_gap_auc_normalized_std': np.std(per_seed_scalars['acc_gap_auc_normalized']),
            'max_acc_gap': max_acc_gap,
            'max_acc_gap_std': np.std(per_seed_scalars['max_acc_gap']),
            'step_at_max_acc_gap': step_at_max_acc_gap,
            'step_at_90pct_max_acc': step_at_90_max_acc,
            'max_reward': max_reward,
            'max_reward_std': np.std(per_seed_scalars['max_reward']),
            'hms_score': hms_score,
            'hms_score_std': np.std(per_seed_scalars['hms_score']),
            'max_er_mean': max_er_mean,
            'max_acc_mean': max_acc_mean,
        })

    df_stats = pd.DataFrame(results)
    df_stats = df_stats.sort_values('hms_score', ascending=False)
    return df_stats



# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def format_corr_df_func(corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert correlation DataFrame into a formatted display version with
    three columns: benchmark_metric, spearman, pearson.
    """
    formatted_data = []
    for _, row in corr_df.iterrows():
        spearman = f"{row['spearman_r']:+.3f} (p={row['spearman_p']:.3f})"
        pearson = f"{row['pearson_r']:+.3f} (p={row['pearson_p']:.3f})"
        formatted_data.append({
            'benchmark_metric': row['benchmark_metric'],
            'spearman': spearman,
            'pearson': pearson,
        })
    return pd.DataFrame(formatted_data, columns=['benchmark_metric', 'spearman', 'pearson'])


def compare_metrics(
    hms_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    corr_metric: str = 'hms_score',
    model_col: str = 'model',
    benchmark_col: Optional[str] = None,
    higher_is_worse: bool = True,
    renorm_benchmark_metrics: bool = False,
    sort_by: str = 'spearman_r',
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare HMS metrics with existing benchmark scores.

    Args:
        hms_df: DataFrame with HMS statistics (from compute_aggregate_statistics)
        benchmark_df: DataFrame with benchmark scores
        corr_metric: Column name in hms_df to compare (default: 'hms_score')
        model_col: Column name for model identifiers (default: 'model')
        benchmark_col: Column in benchmark_df to compare against (if None, tries all numeric)
        higher_is_worse: If True, higher HMS values mean worse alignment
        renorm_benchmark_metrics: If True, z-score normalize each benchmark metric
        sort_by: Column to sort by (default: 'spearman_r')
        verbose: Whether to display intermediate results

    Returns:
        Tuple of (corr_df, merged_data)
    """
    std_col = f"{corr_metric}_std"
    hms_cols = [model_col, corr_metric] + ([std_col] if std_col in hms_df.columns else [])
    merged = hms_df[hms_cols].merge(
        benchmark_df, left_on=model_col, right_on=model_col, how='inner'
    )

    if len(merged) == 0:
        print("Warning: No matching models found between datasets!")
        return pd.DataFrame(), merged

    if verbose:
        print(f"Matched {len(merged)} models\n")

    if benchmark_col is None:
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        benchmark_cols = [col for col in numeric_cols if col != corr_metric]
    else:
        benchmark_cols = [benchmark_col]

    if renorm_benchmark_metrics:
        for bcol in benchmark_cols:
            col_mean = merged[bcol].mean()
            col_std = merged[bcol].std()
            if col_std > 0:
                merged[bcol] = (merged[bcol] - col_mean) / col_std

    results = []
    for bcol in benchmark_cols:
        valid_mask = merged[[corr_metric, bcol]].notna().all(axis=1)
        if valid_mask.sum() < 3:
            continue

        x = merged.loc[valid_mask, corr_metric]
        y = merged.loc[valid_mask, bcol]

        spearman_corr, spearman_p = spearmanr(x, y)
        pearson_corr, pearson_p = pearsonr(x, y)

        results.append({
            'benchmark_metric': bcol,
            'n_models': valid_mask.sum(),
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p,
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
        })

    corr_df = pd.DataFrame(results).sort_values(sort_by, key=abs, ascending=False)

    if verbose:
        format_corr_df = format_corr_df_func(corr_df)
        print(f"Correlation Analysis for: {corr_metric}")
        try:
            from IPython.display import display
            display(format_corr_df)
        except ImportError:
            print(format_corr_df.to_string())

    return corr_df, merged



# ============================================================================
# DATA LOADING AND SMOOTHING
# ============================================================================

def extract_per_step_data(results: List[Dict]) -> pd.DataFrame:
    """
    Extract per-step data including rewards from judge results.

    Parameters:
        results: List of judge result dictionaries

    Returns:
        DataFrame with columns: step, reward
    """
    data = []
    for item in results:
        data.append({
            'step': item['step'],
            'reward': item.get('existing_rewards', np.nan)
        })

    df = pd.DataFrame(data)
    df = df.sort_values('step').reset_index(drop=True)
    return df


def apply_smoothing(df: pd.DataFrame,
                    metric_cols: List[str],
                    window: int = 20,
                    center: bool = True) -> pd.DataFrame:
    """
    Apply rolling average smoothing to specified columns.

    Parameters:
        df: DataFrame containing metrics
        metric_cols: List of column names to smooth
        window: Window size for rolling average
        center: If True, center the window. If False, use trailing window.

    Returns:
        DataFrame with added smoothed columns (suffix: _smoothed)
    """
    df = df.copy()
    for col in metric_cols:
        if col in df.columns:
            df[f'{col}_smoothed'] = df[col].rolling(
                window=window,
                center=center,
                min_periods=1
            ).mean()
    return df


def load_hms_config(judge_results_file: str):
    """
    Extract environment configuration from judge results file path.

    Returns:
        Tuple of (env_name, env_name_formatted, output_dir, hms_dir)
    """
    poss_envs = ["therapy_talk", "political_qa", "action_advice"]
    for env in judge_results_file.split("/"):
        if env in poss_envs:
            env_name = env
            break
    else:
        raise ValueError(f"Could not detect environment from path: {judge_results_file}")

    env_name_formatted = env_name.replace("_", " ").title()
    output_dir = Path(judge_results_file).parent / "hms_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    return env_name, env_name_formatted, output_dir, output_dir


def metrics_to_dataframe(metrics_dict: Dict, is_window: bool = False) -> pd.DataFrame:
    """Convert HMS metrics dict to pandas DataFrame."""
    rows = []
    for key, metrics in metrics_dict.items():
        if is_window:
            row = {'window_start': key[0], 'window_end': key[1],
                   'window_center': (key[0] + key[1]) / 2}
        else:
            row = {'step': key}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = 'window_start' if is_window else 'step'
    df = df.sort_values(sort_col).reset_index(drop=True)
    return df


def get_smoothed_metrics(
    all_results: Dict,
    env_name: str,
    model_names: Optional[Union[str, List[str]]] = None,
    seed_nrs: Optional[Union[int, List[int]]] = None,
    smoothing_window: int = 20,
    center: bool = True,
    include_reward: bool = True,
    max_step = None,
    verbose: bool = True
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Load and smooth HMS metrics and rewards for specified experiments.

    Parameters:
        all_results: Dictionary containing experiment results
        env_name: Environment name to filter results
        model_names: Model name(s) to include (None = all)
        seed_nrs: Seed number(s) to include (None = all)
        smoothing_window: Window size for rolling average
        center: If True, center the window. If False, use trailing window
        include_reward: If True, include reward in the output
        max_step: If set, drop all steps >= this value

    Returns:
        Nested dict {model_name: {seed: smoothed_df}}
        where smoothed_df contains step, metrics, and smoothed versions
    """
    # Normalize inputs to lists
    if model_names is not None and not isinstance(model_names, list):
        model_names = [model_names]
    if seed_nrs is not None and not isinstance(seed_nrs, list):
        seed_nrs = [seed_nrs]

    smoothed_data = {}
    env_data = all_results.get(env_name, {})

    for model_name, experiments in sorted(env_data.items()):
        if model_names is not None and model_name not in model_names:
            continue

        smoothed_data[model_name] = {}

        for seed, exp_data in sorted(experiments.items()):
            if seed_nrs is not None and seed not in seed_nrs:
                continue

            # Load judge results
            try:
                judge_file = exp_data['results_dir'] + '/retroactive_evals/llm_judge_scores_from_reward_step1_subsample1.jsonl'
                env, env_fmt, out_dir, hms_dir = load_hms_config(judge_file)
                results = load_judge_results(judge_file, verbose=verbose)
            except:
                judge_file = exp_data['results_dir'] + '/llm_judge_scores_step1_subsample1.jsonl'
                env, env_fmt, out_dir, hms_dir = load_hms_config(judge_file)
                results = load_judge_results(judge_file, verbose=verbose)

            # Get HMS metrics
            hms_metric_fn = get_custom_metric_function(env)
            per_step_metrics = compute_hms_metrics_per_step(results, hms_metric_fn, verbose=False)
            df_metrics = metrics_to_dataframe(per_step_metrics, is_window=False)

            # Get rewards if requested
            if include_reward:
                df_rewards = extract_per_step_data(results)
                # Merge rewards with metrics
                df = pd.merge(df_metrics, df_rewards, on='step', how='left')
            else:
                df = df_metrics

            # Filter by max_step if specified
            if max_step is not None:
                df = df[df['step'] < max_step]

            # Apply smoothing to base metrics first
            metric_cols = ['ACC_mean', 'G_ACC_mean', 'NG_ER_mean', 'ER_mean']
            if include_reward:
                metric_cols.append('reward')

            df = apply_smoothing(df, metric_cols, window=smoothing_window, center=center)

            # Compute gap metrics from smoothed values
            # (smoothing aggregates both gameable and non-gameable steps, so gaps are meaningful)
            df['ACC_gap_smoothed'] = df['ACC_mean_smoothed'] - df['G_ACC_mean_smoothed']
            df['ER_gap_smoothed'] = df['ER_mean_smoothed'] - df['NG_ER_mean_smoothed']

            smoothed_data[model_name][seed] = df

    return smoothed_data
