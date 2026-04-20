"""
Alignment utility functions for the metric alignment ablation.

Import in a notebook:

    from llm_agent.ablations.metric_alignment.alignment_utils import (
        list_available_combos,
        list_judge_models,
        load_alignment_data,
        compute_correlations,
        compute_agreement_rate,
        summarize_alignment,
        plot_score_comparison,
        plot_alignment_grid,
    )

Terminology
-----------
  *_orig  — original scores: reward from Llama-3.1-8B-Instruct (training judge),
            er_metric / acc_metric from gpt-4.1-mini (retroactive eval).
  *_new   — scores re-computed by the new judge model via logprobs.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

ALIGNMENT_BASE = Path("./results_ablation/metric_alignment")
# ALIGNMENT_BASE = Path("./results_ablation/v1_5samp_metric_alignment")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def list_available_combos(
    base_dir: Optional[Path] = None,
) -> list[tuple[str, str]]:
    """
    Return all (env_name, training_model_slug) combos that have a
    selected_samples.jsonl file.
    """
    base_dir = base_dir or ALIGNMENT_BASE
    combos = []
    for path in sorted(base_dir.rglob("selected_samples.jsonl")):
        model_slug = path.parent.name
        env_name = path.parent.parent.name
        combos.append((env_name, model_slug))
    return combos


def list_judge_models(
    env_name: str,
    training_model_slug: str,
    base_dir: Optional[Path] = None,
) -> list[str]:
    """
    Return the judge model slugs for which score files exist in a combo dir.
    """
    base_dir = base_dir or ALIGNMENT_BASE
    combo_dir = base_dir / env_name / training_model_slug
    return [
        p.stem[: -len("_scores")]  # strip "_scores" suffix
        for p in sorted(combo_dir.glob("*_scores.jsonl"))
    ]


def load_alignment_data(
    env_name: str,
    training_model_slug: str,
    judge_model_slug: str,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load a DataFrame pairing original and new judge scores for one combo.

    Columns
    -------
    is_gameable   : bool  — whether the sample is from the harmful/gameable dist.
    reward_orig   : float — training reward (Llama-3.1-8B logprobs, stored in completions).
    er_orig       : int   — ER score from gpt-4.1-mini retroactive eval.
    acc_orig      : int   — ACC score from gpt-4.1-mini retroactive eval.
    reward_new    : float — reward score from the new judge (logprobs weighted avg).
    er_new        : float — ER score from the new judge (logprobs weighted avg).
    acc_new       : float — ACC score from the new judge (logprobs weighted avg).
    env_name      : str
    training_model: str
    judge_model   : str
    """
    base_dir = base_dir or ALIGNMENT_BASE
    combo_dir = base_dir / env_name / training_model_slug

    samples = _load_jsonl(combo_dir / "selected_samples.jsonl")
    new_scores = _load_jsonl(combo_dir / f"{judge_model_slug}_scores.jsonl")

    rows = []
    for sample, new in zip(samples, new_scores):
        rows.append(
            {
                "is_gameable":    sample.get("is_gameable"),
                "reward_orig":    sample.get("reward"),
                "er_orig":        sample.get("er_metric"),
                "acc_orig":       sample.get("acc_metric"),
                "reward_new":     new.get("reward"),
                "er_new":         new.get("er_metric"),
                "acc_new":        new.get("acc_metric"),
                "env_name":       env_name,
                "training_model": training_model_slug,
                "judge_model":    judge_model_slug,
            }
        )

    return pd.DataFrame(rows)


def load_all_alignment_data(
    judge_model_slug: str,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load and concatenate alignment data for ALL (env, training_model) combos
    that have scores for the given judge model slug.
    """
    base_dir = base_dir or ALIGNMENT_BASE
    frames = []
    for env_name, training_model_slug in list_available_combos(base_dir):
        score_path = base_dir / env_name / training_model_slug / f"{judge_model_slug}_scores.jsonl"
        if score_path.exists():
            frames.append(
                load_alignment_data(env_name, training_model_slug, judge_model_slug, base_dir)
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Alignment metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlations(
    col_a: pd.Series,
    col_b: pd.Series,
) -> dict[str, float]:
    """
    Compute Pearson and Spearman correlations between two score series.

    Returns a dict with keys: pearson_r, spearman_r, kendall_tau, n.
    Each correlation value is a string formatted as "correlation (p-value)".
    NaN values in either series are excluded pairwise.
    """
    mask = col_a.notna() & col_b.notna()
    a, b = col_a[mask].to_numpy(dtype=float), col_b[mask].to_numpy(dtype=float)
    if len(a) < 3:
        return {"pearson_r": "nan", "spearman_r": "nan", "kendall_tau": "nan", "n": int(len(a))}

    pearson_r, pearson_p = stats.pearsonr(a, b)
    spearman_r, spearman_p = stats.spearmanr(a, b)
    kendall_tau, kendall_p = stats.kendalltau(a, b)
    
    return {
        "pearson_r": f"{pearson_r:.2f} ({pearson_p:.2e})",
        "spearman_r": f"{spearman_r:.2f} ({spearman_p:.2e})",
        "kendall_tau": f"{kendall_tau:.2f} ({kendall_p:.2e})",
        "n": int(len(a))
    }

def compute_agreement_rate(
    col_a: pd.Series,
    col_b: pd.Series,
    tolerance: float = 1.0,
) -> float:
    """
    Fraction of samples where |score_a - score_b| <= tolerance.
    NaN values are excluded.
    """
    mask = col_a.notna() & col_b.notna()
    a, b = col_a[mask].to_numpy(dtype=float), col_b[mask].to_numpy(dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b) <= tolerance))


def summarize_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise correlations and agreement rates for a standard set of
    metric pairs in the alignment DataFrame.

    Pairs examined
    --------------
    - reward_orig ↔ reward_new     (same metric, different judge)
    - er_orig     ↔ er_new         (same metric, different judge)
    - acc_orig    ↔ acc_new        (same metric, different judge)
    - reward_orig ↔ er_orig        (different metric, original judge)
    - reward_new  ↔ er_new         (different metric, new judge)
    - er_orig     ↔ acc_orig       (different metric, original judge)
    - er_new      ↔ acc_new        (different metric, new judge)
    - reward_orig ↔ er_new         (cross-judge: orig reward vs new ER)

    Returns a DataFrame with columns: pair, pearson_r, spearman_r, kendall_tau, n, agreement_±1
    """
    pairs = [
        ("reward_orig", "reward_new",  "reward: orig vs new judge"),
        ("er_orig",     "er_new",      "er:     orig vs new judge"),
        ("acc_orig",    "acc_new",     "acc:    orig vs new judge"),
        # ("reward_orig", "er_orig",     "reward vs er  (orig judge)"),
        # ("reward_new",  "er_new",      "reward vs er  (new judge)"),
        # ("er_orig",     "acc_orig",    "er vs acc     (orig judge)"),
        # ("er_new",      "acc_new",     "er vs acc     (new judge)"),
        # ("reward_orig", "er_new",      "reward_orig vs er_new (cross-judge)"),
    ]

    rows = []
    for col_a, col_b, label in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        corr = compute_correlations(df[col_a], df[col_b])
        agree = compute_agreement_rate(df[col_a], df[col_b])
        rows.append({"pair": label, **corr, "agreement_±1": agree})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_comparison(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color_by_gameable: bool = True,
    ax=None,
    title: Optional[str] = None,
):
    """
    Scatter plot of col_a vs col_b with an OLS regression line.

    If color_by_gameable=True and 'is_gameable' is in df, harmful samples
    (is_gameable=True) are plotted in orange and harmless in blue.

    Returns the matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    mask = df[col_a].notna() & df[col_b].notna()
    sub = df[mask].copy()
    a = sub[col_a].to_numpy(dtype=float)
    b = sub[col_b].to_numpy(dtype=float)

    if color_by_gameable and "is_gameable" in sub.columns:
        gameable = sub["is_gameable"].astype(bool)
        ax.scatter(a[gameable],  b[gameable],  alpha=0.4, s=18, color="tab:orange", label="harmful")
        ax.scatter(a[~gameable], b[~gameable], alpha=0.4, s=18, color="tab:blue",   label="harmless")
        ax.legend(fontsize=8, markerscale=1.2)
    else:
        ax.scatter(a, b, alpha=0.4, s=18, color="tab:blue")

    if len(a) >= 2:
        m, c = np.polyfit(a, b, 1)
        x_line = np.linspace(a.min(), a.max(), 200)
        ax.plot(x_line, m * x_line + c, "r--", linewidth=1.5, zorder=5)

    corr = compute_correlations(pd.Series(a), pd.Series(b))
    subtitle = f"r={corr['pearson_r']:.2f}  ρ={corr['spearman_r']:.2f}  τ={corr['kendall_tau']:.2f}  n={corr['n']}"
    ax.set_title((title or f"{col_a} vs {col_b}") + f"\n{subtitle}", fontsize=9)
    ax.set_xlabel(xlabel or col_a, fontsize=9)
    ax.set_ylabel(ylabel or col_b, fontsize=9)
    return ax


def plot_alignment_grid(
    df: pd.DataFrame,
    pairs: Optional[list[tuple[str, str, str, str]]] = None,
    figsize: Optional[tuple[float, float]] = None,
    color_by_gameable: bool = True,
):
    """
    Plot a grid of scatter plots for multiple score pairs in one figure.

    Parameters
    ----------
    df : alignment DataFrame from load_alignment_data()
    pairs : list of (col_a, col_b, xlabel, ylabel).
            Defaults to the three cross-judge same-metric pairs.
    figsize : figure size; auto-sized if None.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if pairs is None:
        pairs = [
            ("reward_orig", "reward_new", "reward (orig judge)", "reward (new judge)"),
            ("er_orig",     "er_new",     "ER (orig judge)",     "ER (new judge)"),
            ("acc_orig",    "acc_new",    "ACC (orig judge)",    "ACC (new judge)"),
        ]

    ncols = len(pairs)
    if figsize is None:
        figsize = (4.5 * ncols, 4.5)

    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    for ax, (col_a, col_b, xlabel, ylabel) in zip(axes, pairs):
        plot_score_comparison(
            df, col_a, col_b,
            xlabel=xlabel, ylabel=ylabel,
            color_by_gameable=color_by_gameable,
            ax=ax,
        )

    fig.tight_layout()
    return fig
