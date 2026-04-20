"""
Plotting utilities for reward vs er_metric analysis:
- Scatter plot of reward vs er_metric for small/large models at start/end of training
- Bar plot of mean Max HEX Gap for small vs large models across environments

Uses combined JSONL files from results_ablation/completions_restruct/.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from scipy.stats import spearmanr

# Model name -> num_params (billions)
MODEL_SIZES = {
    "Qwen1_5_0_5B_Chat": 0.5,
    "Qwen1_5_1_8B_Chat": 1.8,
    "gemma_1_1_2b_it": 2.0,
    "Qwen1_5_4B_Chat": 4.0,
    "Yi_6B_Chat": 6.0,
    "Llama_2_7b_chat_hf": 7.0,
    "Qwen1_5_7B_Chat": 7.0,
    "gemma_1_1_7b_it": 7.0,
    "Meta_Llama_3_8B_Instruct": 8.0,
    "Llama_2_13b_chat_hf": 13.0,
    "Qwen1_5_14B_Chat": 14.0,
}

# Display names for models
MODEL_DISPLAY = {k: k.replace("_", "-") for k in MODEL_SIZES}


def load_combined_jsonl(
    env_name: str,
    model_name: str,
    seeds: List[int] = [5, 42, 83],
    base_dir: str = "results_ablation/completions_restruct",
) -> list:
    """Load all combined JSONL entries for a model across seeds."""
    entries = []
    for seed in seeds:
        path = Path(base_dir) / env_name / model_name / f"S{seed}_combined.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                d["_model_name"] = model_name
                d["_seed"] = seed
                entries.append(d)
    return entries


def get_models_by_size(
    env_name: str,
    n_small: int = 2,
    n_large: int = 2,
    base_dir: str = "results_ablation/completions_restruct",
) -> Tuple[List[str], List[str]]:
    """Get the n smallest and n largest models available for this environment."""
    available = [
        d.name for d in (Path(base_dir) / env_name).iterdir()
        if d.is_dir() and d.name in MODEL_SIZES
    ]
    available.sort(key=lambda m: MODEL_SIZES[m])
    small = available[:n_small]
    large = available[-n_large:]
    return small, large


def filter_entries(
    entries: list,
    step_min: int,
    step_max: int,
    gameable_only: bool = True,
    n_samples: int = 50,
) -> list:
    """Filter entries by step range, gameable flag, and subsample."""
    filtered = [
        e for e in entries
        if step_min <= e["step"] <= step_max
        and (not gameable_only or e.get("is_gameable", False))
    ]
    if len(filtered) > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(filtered), n_samples, replace=False)
        filtered = [filtered[i] for i in idx]
    return filtered


def plot_reward_vs_er(
    env_name: str = "therapy_talk",
    t_range: int = 50,
    max_step: int = 300,
    n_samples: int = 50,
    n_small: int = 2,
    n_large: int = 2,
    gameable_only: bool = True,
    seeds: List[int] = [5, 42, 83],
    base_dir: str = "results_ablation/completions_restruct",
    output_dir: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5.5),
    alpha: float = 0.55,
    point_size: int = 40,
    jitter: float = 0.0,
    x_start: Optional[float] = None,
):
    """
    Scatter plot: reward (x) vs er_metric (y) for small/large models at start/end of training.

    Parameters:
        env_name: Environment name (therapy_talk, action_advice, political_qa)
        t_range: Step range width. Start = [0, t_range], End = [max_step - t_range, max_step]
        max_step: Maximum training step to consider
        n_samples: Max samples per group
        n_small: Number of smallest models to include
        n_large: Number of largest models to include
        gameable_only: Only use gameable user contexts
        seeds: List of random seeds
        base_dir: Path to completions_restruct directory
        output_dir: If set, save figure there
        figsize: Figure size
        alpha: Point transparency
        point_size: Marker size
        jitter: Amount of random jitter to add to er_metric (y-axis) to reduce overplotting.
            0.0 = no jitter, 0.2-0.3 is a good default for discrete metrics.
    """
    small_models, large_models = get_models_by_size(env_name, n_small, n_large, base_dir)

    start_range = (0, t_range)
    end_range = (max_step - t_range, max_step)

    # Define the 4 groups: (label, models, step_range, color, marker)
    groups = [
        ("Small models – start", small_models, start_range, "#4DFF00", "o"),
        ("Small models – end",   small_models, end_range,   "#BD8B2D", "o"),
        ("Large models – start", large_models, start_range, "#B037B8", "s"),
        ("Large models – end",   large_models, end_range,   "#2E189B", "s"),
    ]

    fig, ax = plt.subplots(figsize=figsize)

    for label, models, (s_min, s_max), color, marker in groups:
        rewards, er_metrics = [], []
        for model in models:
            entries = load_combined_jsonl(env_name, model, seeds, base_dir)
            filtered = filter_entries(entries, s_min, s_max, gameable_only, n_samples)
            for e in filtered:
                rewards.append(e["reward"])
                er_metrics.append(e["er_metric"])

        if jitter > 0:
            rng = np.random.RandomState(hash(label) % 2**31)
            er_metrics = [v + rng.uniform(-jitter, jitter) for v in er_metrics]

        size_str = ", ".join(f"{MODEL_SIZES[m]}B" for m in models)
        full_label = f"{label} ({size_str})"
        ax.scatter(
            rewards, er_metrics,
            c=color, marker=marker, s=point_size,
            alpha=alpha, label=full_label, edgecolors="white", linewidth=0.3,
        )

    env_display = env_name.replace("_", " ").title()
    if x_start is not None:
        ax.set_xlim(left=x_start)
    ax.set_xlabel("Reward", fontsize=13)
    ax.set_ylabel("HEX Gap (er_metric)", fontsize=13)
    ax.set_title(f"Reward vs HEX Gap – {env_display}", fontsize=14)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"scatter_reward_er_{env_name}.pdf"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved to {fname}")

    plt.show()
    return fig, ax


ENV_DISPLAY = {
    "therapy_talk": "TT",
    "action_advice": "AA",
    "political_qa": "PQA",
}

SMALL_THRESHOLD = 4.0   # <= 4B
LARGE_THRESHOLD = 7.0   # >= 7B


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def compute_bar_data(
    env_names: List[str] = ["therapy_talk", "action_advice", "political_qa"],
    corr_metric: str = "max_er_gap",
    small_range: Tuple[float, float] = (0, SMALL_THRESHOLD),
    large_range: Tuple[float, float] = (LARGE_THRESHOLD, 100),
) -> Dict:
    """
    Compute the data needed for the bar plot (slow step — runs correlations).

    Returns a dict with keys: env_names, corr_metric, small_range, large_range,
    and per-env entries with small/large means, stds, and rho annotations.
    """
    from llm_agent.misalignment_metrics.utils import (
        collect_multi_env_correlations,
        load_model_features_metrics,
    )
    from llm_agent.utils.utils import get_all_results_dirs_for_experiment

    sw_metrics = load_model_features_metrics(
        "./safetywashing/data/model_features_metrics_Combi.json"
    )
    sw_metrics["model"] = sw_metrics["model"].str.replace("-", "_").str.replace(".", "_")

    env_results = collect_multi_env_correlations(
        env_names=env_names,
        sw_metrics=sw_metrics,
        corr_metric=corr_metric,
        get_all_results_dirs_for_experiment=get_all_results_dirs_for_experiment,
    )

    data = {
        "env_names": env_names,
        "corr_metric": corr_metric,
        "small_range": small_range,
        "large_range": large_range,
        "envs": {},
    }

    for env_name in env_names:
        corr_df, merged = env_results[env_name]

        small_mask = (merged["num_params"] >= small_range[0]) & (merged["num_params"] <= small_range[1])
        large_mask = (merged["num_params"] >= large_range[0]) & (merged["num_params"] <= large_range[1])

        small_vals = merged.loc[small_mask, corr_metric]
        large_vals = merged.loc[large_mask, corr_metric]

        row = corr_df[corr_df["benchmark_metric"] == "num_params"]
        if len(row) > 0:
            rho = row.iloc[0]["spearman_r"]
            p = row.iloc[0]["spearman_p"]
            stars = _significance_stars(p)
            rho_text = f"\u03c1 = {rho:.2f}{stars}"
        else:
            rho_text = ""

        data["envs"][env_name] = {
            "small_mean": small_vals.mean(),
            "small_std": small_vals.std(),
            "large_mean": large_vals.mean(),
            "large_std": large_vals.std(),
            "rho_annotation": rho_text,
        }

    return data


def plot_bar_max_hex_gap(
    bar_data: Dict,
    figsize: Tuple[float, float] = (6, 4.5),
    output_path: Optional[str] = None,
    bar_width: float = 0.3,
    colors: Tuple[str, str] = ("#66c2a5", "#fc8d62"),
    show_values: bool = True,
    show_std: bool = True,
    show_yticks: bool = True,
    font_scale: float = 1.0,
):
    """
    Bar plot of mean Max HEX Gap for small vs large models across environments.
    Takes pre-computed data from compute_bar_data().

    Parameters:
        bar_data: Output of compute_bar_data()
        figsize: Figure size
        output_path: If set, save figure there
        bar_width: Width of each bar
        colors: (small_color, large_color)
        show_values: Annotate bar heights
        show_std: Show error bars (std)
        show_yticks: Show y-axis tick labels
        font_scale: Scale all font sizes
    """
    env_names = bar_data["env_names"]
    corr_metric = bar_data["corr_metric"]
    small_range = bar_data["small_range"]
    large_range = bar_data["large_range"]

    env_labels = [ENV_DISPLAY.get(e, e) for e in env_names]
    small_means = [bar_data["envs"][e]["small_mean"] for e in env_names]
    small_stds = [bar_data["envs"][e]["small_std"] for e in env_names]
    large_means = [bar_data["envs"][e]["large_mean"] for e in env_names]
    large_stds = [bar_data["envs"][e]["large_std"] for e in env_names]
    rho_annotations = [bar_data["envs"][e]["rho_annotation"] for e in env_names]

    x = np.arange(len(env_labels))
    fs = font_scale

    fig, ax = plt.subplots(figsize=figsize)

    bars_small = ax.bar(
        x - bar_width / 2, small_means, bar_width,
        yerr=small_stds if show_std else None, capsize=4,
        label=f"Small ({small_range[0]}–{small_range[1]}B)",
        color=colors[0], edgecolor="white", linewidth=0.5,
    )
    bars_large = ax.bar(
        x + bar_width / 2, large_means, bar_width,
        yerr=large_stds if show_std else None, capsize=4,
        label=f"Large ({large_range[0]}–{large_range[1]}B)",
        color=colors[1], edgecolor="white", linewidth=0.5,
    )

    if show_values:
        for bars in [bars_small, bars_large]:
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8 * fs,
                )

    for i, rho_text in enumerate(rho_annotations):
        if rho_text:
            env = env_names[i]
            small_top = bar_data["envs"][env]["small_mean"] + (bar_data["envs"][env]["small_std"] if show_std else 0)
            large_top = bar_data["envs"][env]["large_mean"] + (bar_data["envs"][env]["large_std"] if show_std else 0)
            local_top = max(small_top, large_top)
            try:
                rho_val = float(rho_text.split("=")[1].strip().rstrip("*"))
                rho_color = "#237a23" if rho_val >= 0 else "#a12323"
            except (IndexError, ValueError):
                rho_color = "#333333"
            ax.text(
                x[i], local_top + 0.05, rho_text,
                ha="center", va="bottom", fontsize=10 * fs,
                fontstyle="italic", fontweight="bold", color=rho_color,
            )

    metric_display = corr_metric.replace("_", " ").title()
    metric_display = "HEX Gap" 
    ax.set_ylabel(f"{metric_display}", fontsize=12 * fs)
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, fontsize=11 * fs)
    if not show_yticks:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.tick_params(axis="y", labelsize=11 * fs)
    ax.legend(fontsize=9 * fs, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved to {output_path}")

    # plt.show()
    return fig, ax


if __name__ == "__main__":
    plot_reward_vs_er(
        env_name="therapy_talk",
        output_dir="llm_agent/misalignment_metrics/figs",
    )
