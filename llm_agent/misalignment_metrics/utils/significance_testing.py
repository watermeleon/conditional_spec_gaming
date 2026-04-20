import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from scipy.stats import spearmanr, pearsonr



def compute_per_seed_scores(smoothed_data: Dict, max_step: int = 300) -> pd.DataFrame:
    """
    Compute scalar HMS metrics for each individual seed (no averaging).

    Returns a long-format DataFrame with one row per (model, seed), containing
    the same scalar metrics used in compute_aggregate_statistics. Suitable as
    input to bootstrap_seed_sensitivity.

    Args:
        smoothed_data: Nested dict {model_name: {seed: DataFrame}}
        max_step: Maximum step to consider (default: 300)

    Returns:
        DataFrame with columns [model, seed, hms_score, er_gap_auc_normalized,
                                acc_gap_auc_normalized, max_er_gap, max_acc_gap, max_reward]
    """
    records = []
    for model_name, seeds_data in smoothed_data.items():
        for seed, df in seeds_data.items():
            if max_step is not None:
                df = df[df['step'] <= max_step]
            sv = df['step'].values
            tr = sv[-1] - sv[0]
            er_v = df['ER_gap_smoothed'].values if 'ER_gap_smoothed' in df.columns else np.array([np.nan])
            acc_v = df['ACC_gap_smoothed'].values if 'ACC_gap_smoothed' in df.columns else np.array([np.nan])
            rew_v = df['reward_smoothed'].values if 'reward_smoothed' in df.columns else np.array([np.nan])
            er_auc_n = np.trapezoid(er_v, sv) / tr if tr > 0 else np.trapezoid(er_v, sv)
            acc_auc_n = np.trapezoid(acc_v, sv) / tr if tr > 0 else np.trapezoid(acc_v, sv)
            records.append({
                'model': model_name,
                'seed': seed,
                'hms_score': 0.5 * np.abs(acc_auc_n) + 0.5 * np.abs(er_auc_n),
                'er_gap_auc_normalized': er_auc_n,
                'acc_gap_auc_normalized': acc_auc_n,
                'max_er_gap': np.nanmax(er_v),
                'max_acc_gap': np.nanmax(acc_v),
                'max_reward': np.nanmax(rew_v),
            })
    return pd.DataFrame(records)



def loo_sensitivity(
    merged: pd.DataFrame,
    corr_metric: str,
    benchmark_cols: Optional[List[str]] = None,
    model_col: str = 'model',
    corr_type: str = 'spearman',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Leave-one-out sensitivity analysis for correlations.

    For each benchmark metric, drops one model at a time and recomputes ρ,
    reporting the range (min, max) of ρ values across all leave-one-out runs.

    Args:
        merged: Merged DataFrame (output of compare_metrics)
        corr_metric: Column name of the HMS metric
        benchmark_cols: List of benchmark columns to analyse (if None, uses all numeric except corr_metric)
        model_col: Column name for model identifiers
        corr_type: 'spearman' or 'pearson'
        verbose: Whether to print results

    Returns:
        DataFrame with columns [benchmark_metric, full_r, loo_min, loo_max, loo_range, loo_std]
    """
    corr_fn = spearmanr if corr_type == 'spearman' else pearsonr
    r_key = f'{corr_type}_r'

    if benchmark_cols is None:
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        benchmark_cols = [col for col in numeric_cols if col != corr_metric]

    results = []
    for bcol in benchmark_cols:
        valid = merged[[model_col, corr_metric, bcol]].dropna()
        if len(valid) < 4:  # need at least 3 after dropping one
            continue

        full_r, _ = corr_fn(valid[corr_metric], valid[bcol])

        loo_rs = []
        for idx in valid.index:
            subset = valid.drop(idx)
            r, _ = corr_fn(subset[corr_metric], subset[bcol])
            loo_rs.append(r)

        loo_rs = np.array(loo_rs)
        results.append({
            'benchmark_metric': bcol,
            f'full_{r_key}': full_r,
            'loo_min': loo_rs.min(),
            'loo_max': loo_rs.max(),
            'loo_range': loo_rs.max() - loo_rs.min(),
            'loo_std': loo_rs.std(),
        })

    loo_df = pd.DataFrame(results).sort_values('loo_range', ascending=False)

    if verbose:
        print(f"LOO Sensitivity Analysis ({corr_type})")
        print(loo_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return loo_df


def bootstrap_seed_sensitivity(
    hms_seeds_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    corr_metric: str = 'hms_score',
    model_col: str = 'model',
    seed_col: str = 'seed',
    benchmark_col: Optional[str] = None,
    corr_type: str = 'spearman',
    n_iterations: int = 1000,
    ci: float = 0.95,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Bootstrap seed sensitivity analysis for correlations.

    For each iteration, samples one seed per model (with replacement from that
    model's seeds) to obtain one HMS value per model, then computes ρ against
    each benchmark metric. Reports the CI of ρ across iterations.

    Args:
        hms_seeds_df: Long-format DataFrame with columns [model_col, seed_col, corr_metric]
                      (one row per model-seed combination)
        benchmark_df: DataFrame with benchmark scores (one row per model)
        corr_metric: Column in hms_seeds_df with the HMS score
        model_col: Column name for model identifiers
        seed_col: Column name for seed identifiers (excluded from benchmark cols)
        benchmark_col: Single benchmark column to analyse (if None, uses all numeric)
        corr_type: 'spearman' or 'pearson'
        n_iterations: Number of bootstrap iterations
        ci: Confidence interval level (default: 0.95)
        verbose: Whether to print results

    Returns:
        DataFrame with columns [benchmark_metric, full_r, mean_r, ci_low, ci_high, ci_width]
    """
    from tqdm import tqdm

    corr_fn = spearmanr if corr_type == 'spearman' else pearsonr

    # Merge benchmark values into the per-seed frame (benchmark cols repeat per seed)
    merged = hms_seeds_df.merge(benchmark_df, on=model_col, how='inner')

    if benchmark_col is None:
        numeric_cols = benchmark_df.select_dtypes(include=[np.number]).columns
        benchmark_cols = [col for col in numeric_cols if col != model_col]
    else:
        benchmark_cols = [benchmark_col]

    # Compute full_r using per-model means (comparable to compare_metrics output)
    model_means = merged.groupby(model_col)[corr_metric].mean().reset_index()
    mean_merged = model_means.merge(benchmark_df, on=model_col, how='inner')

    full_rs = {}
    for bcol in benchmark_cols:
        valid = mean_merged[[corr_metric, bcol]].dropna()
        if len(valid) >= 3:
            full_rs[bcol], _ = corr_fn(valid[corr_metric], valid[bcol])

    # Index seeds by model for fast sampling
    model_seed_indices = {
        model: group.index.tolist()
        for model, group in merged.groupby(model_col)
    }
    models = list(model_seed_indices.keys())

    boot_results: dict = {bcol: [] for bcol in benchmark_cols}

    for _ in tqdm(range(n_iterations), desc='Bootstrap'):
        # Sample one seed per model with replacement within each model's seeds
        sampled_idx = [np.random.choice(model_seed_indices[m]) for m in models]
        sample = merged.loc[sampled_idx]

        for bcol in benchmark_cols:
            valid = sample[[corr_metric, bcol]].dropna()
            if len(valid) < 3:
                continue
            r, _ = corr_fn(valid[corr_metric], valid[bcol])
            boot_results[bcol].append(r)

    alpha = 1 - ci
    results = []
    for bcol in benchmark_cols:
        rs = np.array(boot_results[bcol])
        if len(rs) == 0:
            continue
        results.append({
            'benchmark_metric': bcol,
            'full_r': full_rs.get(bcol, np.nan),
            'mean_r': rs.mean(),
            'ci_low': np.percentile(rs, 100 * alpha / 2),
            'ci_high': np.percentile(rs, 100 * (1 - alpha / 2)),
            'ci_width': np.percentile(rs, 100 * (1 - alpha / 2)) - np.percentile(rs, 100 * alpha / 2),
        })

    boot_df = pd.DataFrame(results).sort_values('ci_width', ascending=False)

    if verbose:
        print(f"Bootstrap Seed Sensitivity ({corr_type}, {n_iterations} iters, {int(ci * 100)}% CI, n={len(models)} models)")
        print(boot_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return boot_df


def per_seed_correlation(
    hms_seeds_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    corr_metric: str = 'hms_score',
    model_col: str = 'model',
    seed_col: str = 'seed',
    benchmark_col: Optional[str] = None,
    corr_type: str = 'spearman',
    min_models: int = 3,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman (or Pearson) correlation for each individual seed.

    For each seed present in hms_seeds_df, selects that seed's HMS score for
    every model that has it, then correlates against benchmark scores.  Returns
    both the per-seed results and a per-benchmark summary across seeds.

    Args:
        hms_seeds_df: Long-format DataFrame with columns [model_col, seed_col, corr_metric]
                      (one row per model-seed combination, e.g. output of compute_per_seed_scores)
        benchmark_df: DataFrame with benchmark scores (one row per model)
        corr_metric: Column in hms_seeds_df with the HMS score
        model_col: Column name for model identifiers
        seed_col: Column name for seed identifiers
        benchmark_col: Single benchmark column to analyse (if None, uses all numeric columns)
        corr_type: 'spearman' or 'pearson'
        min_models: Minimum number of models required to compute a correlation for a seed
        verbose: Whether to print the summary table

    Returns:
        Tuple of two DataFrames:
          - per_seed_df: one row per (seed, benchmark_metric) with columns
                         [seed, benchmark_metric, r, p_value, n_models]
          - summary_df:  one row per benchmark_metric with columns
                         [benchmark_metric, mean_r, std_r, min_r, max_r, range_r, n_seeds]
    """
    corr_fn = spearmanr if corr_type == 'spearman' else pearsonr

    merged = hms_seeds_df.merge(benchmark_df, on=model_col, how='inner')

    if benchmark_col is None:
        numeric_cols = benchmark_df.select_dtypes(include=[np.number]).columns
        benchmark_cols = [col for col in numeric_cols if col != model_col]
    else:
        benchmark_cols = [benchmark_col]

    seeds = sorted(merged[seed_col].unique())

    per_seed_records = []
    for seed in seeds:
        seed_df = merged[merged[seed_col] == seed]
        for bcol in benchmark_cols:
            valid = seed_df[[corr_metric, bcol]].dropna()
            if len(valid) < min_models:
                continue
            r, p = corr_fn(valid[corr_metric], valid[bcol])
            per_seed_records.append({
                seed_col: seed,
                'benchmark_metric': bcol,
                'r': r,
                'p_value': p,
                'n_models': len(valid),
            })

    per_seed_df = pd.DataFrame(per_seed_records)

    # Build wide summary: one row per benchmark_metric, one column per seed
    if not per_seed_df.empty:
        per_seed_df['label'] = per_seed_df.apply(
            lambda row: f"{row['r']:.3f} ({row['p_value']:.3f})", axis=1
        )
        summary_df = per_seed_df.pivot(
            index='benchmark_metric', columns=seed_col, values='label'
        )
        summary_df.columns = [f"corr_{s}" for s in summary_df.columns]
        summary_df = summary_df.reset_index()
    else:
        summary_df = pd.DataFrame(columns=['benchmark_metric'])

    if verbose:
        print(f"Per-Seed Correlation ({corr_type}, {len(seeds)} seeds)")
        print(summary_df.to_string(index=False))

    return per_seed_df, summary_df


def per_seed_correlation_from_smoothed_data(
    smoothed_data: Dict,
    benchmark_df: pd.DataFrame,
    corr_metric: str = 'hms_score',
    max_step: int = 300,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: builds per-seed scores from smoothed_data and runs
    per_seed_correlation in one call.

    Args:
        smoothed_data: Nested dict {model_name: {seed: DataFrame}}
        benchmark_df: DataFrame with benchmark scores (one row per model)
        corr_metric: HMS metric to correlate (must be a column in compute_per_seed_scores output)
        max_step: Maximum step to consider when computing per-seed scores
        **kwargs: Forwarded to per_seed_correlation
                  (corr_type, benchmark_col, min_models, verbose, ...)

    Returns:
        Tuple (per_seed_df, summary_df) — see per_seed_correlation
    """
    hms_seeds_df = compute_per_seed_scores(smoothed_data, max_step=max_step)
    return per_seed_correlation(hms_seeds_df, benchmark_df, corr_metric=corr_metric, **kwargs)


def bootstrap_from_smoothed_data(
    smoothed_data: Dict,
    benchmark_df: pd.DataFrame,
    corr_metric: str = 'hms_score',
    max_step: int = 300,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper: builds per-seed scores from smoothed_data and runs
    bootstrap_seed_sensitivity in one call.

    Args:
        smoothed_data: Nested dict {model_name: {seed: DataFrame}}
        benchmark_df: DataFrame with benchmark scores (one row per model)
        corr_metric: HMS metric to correlate (must be a column in compute_per_seed_scores output)
        max_step: Maximum step to consider when computing per-seed scores
        **kwargs: Forwarded to bootstrap_seed_sensitivity
                  (corr_type, n_iterations, ci, benchmark_col, verbose, ...)

    Returns:
        Bootstrap results DataFrame (see bootstrap_seed_sensitivity)
    """
    hms_seeds_df = compute_per_seed_scores(smoothed_data, max_step=max_step)
    return bootstrap_seed_sensitivity(hms_seeds_df, benchmark_df, corr_metric=corr_metric, **kwargs)

