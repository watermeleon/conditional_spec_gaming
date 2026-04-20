"""Loading and joining utilities for TT gameable reward-vs-safety analysis.

The pipeline produces two log files per (model, seed) run:

  * ``completions_from_reward.jsonl``   — one row per rollout, with the
    training-time reward, is_gameable flag, env_name, subenv_id, step.
  * ``retroactive_evals/llm_judge_scores_from_reward_step1_subsample1.jsonl``
    — one row per step (random sample_index), containing gpt-4.1-mini
    ``acc_metric`` (dissuade) and ``er_metric`` (encouragement) scores.

Both files are joined on ``(step, agent_message)``. Rewards come from the
training-time judge (Llama-3.1-8B-Instruct) and are comparable across runs
because the same judge was used for every run.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(os.getenv("RESULTS_DIR", "./results")) / "therapy_talk"

# 11 instruct/chat models used in the paper, with family, size (B params),
# and the 3 seed directories that contain both completions + retroactive evals.
# All under experiment subdir 'exp5seeds'.
MODEL_REGISTRY: List[Dict] = [
    dict(model="Qwen1_5_0_5B_Chat",        family="Qwen",  size=0.5, seeds=["S42_0.5",   "S5_0.5",      "S83_0.5"]),
    dict(model="Qwen1_5_1_8B_Chat",        family="Qwen",  size=1.8, seeds=["S42_0.5",   "S5_0.5",      "S83_0.5"]),
    dict(model="Qwen1_5_4B_Chat",          family="Qwen",  size=4.0, seeds=["S42_0.5",   "S5_0.5",      "S83_0.5"]),
    dict(model="Qwen1_5_7B_Chat",          family="Qwen",  size=7.0, seeds=["S42_0.5",   "S5_0.5",      "S83_0.5"]),
    dict(model="Qwen1_5_14B_Chat",         family="Qwen",  size=14.0,seeds=["S42_0.5",   "S5_0.5",      "S83_0.5"]),
    dict(model="Llama_2_7b_chat_hf",       family="Llama", size=7.0, seeds=["S42_0.5",   "S5_0.5_v2",   "S83_0.5_v3"]),
    dict(model="Llama_2_13b_chat_hf",      family="Llama", size=13.0,seeds=["S42_0.5",   "S5_0.5",      "S83_0.5"]),
    dict(model="Meta_Llama_3_8B_Instruct", family="Llama", size=8.0, seeds=["S42_0.5",   "S5_0.5_v3",   "S83_0.5_v3"]),
    dict(model="gemma_1_1_2b_it",          family="Gemma", size=2.0, seeds=["S42_0.5",   "S5_0.5",      "S83_0.5"]),
    dict(model="gemma_1_1_7b_it",          family="Gemma", size=7.0, seeds=["S42_0.5",   "S5_0.5",      "S83_0.5"]),
    dict(model="Yi_6B_Chat",               family="Yi",    size=6.0, seeds=["S42_0.5_v2","S5_0.5_v2",   "S83_0.5_v3"]),
]

# Models that were trained for 500 steps rather than 300. We always cap to the
# first 300 steps so that "last 50 steps" is comparable across models.
MAX_STEP = 300

EXP = "exp5seeds"
RETRO_FILENAME = "llm_judge_scores_from_reward_step1_subsample1.jsonl"


def _load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _load_run(model: str, seed: str) -> Optional[pd.DataFrame]:
    """Load one (model, seed) run and return the joined gameable DataFrame.

    Columns: model, family, size, seed, step, sample_index, env_name,
    subenv_id, is_gameable, reward, agent_message, er_metric, acc_metric.

    Returns None if either file is missing. Capped to step < MAX_STEP.
    """
    run_dir = ROOT / model / EXP / seed
    comp_path = run_dir / "completions_from_reward.jsonl"
    retro_path = run_dir / "retroactive_evals" / RETRO_FILENAME
    if not comp_path.exists() or not retro_path.exists():
        return None

    comps = _load_jsonl(comp_path)
    retros = _load_jsonl(retro_path)

    retro_lookup = {
        (r["step"], r["agent_message"]): r["llm_judge_scores"] for r in retros
    }

    meta = next(m for m in MODEL_REGISTRY if m["model"] == model)

    rows = []
    for r in comps:
        if not r.get("is_gameable"):
            continue
        if r["step"] >= MAX_STEP:
            continue
        k = (r["step"], r["agent_message"])
        scores = retro_lookup.get(k)
        if scores is None:
            continue
        rows.append(
            dict(
                model=model,
                family=meta["family"],
                size=meta["size"],
                seed=seed,
                step=r["step"],
                sample_index=r.get("sample_index"),
                env_name=r["env_name"],
                subenv_id=r["subenv_id"],
                reward=r["reward"],
                agent_message=r["agent_message"],
                er_metric=scores.get("er_metric"),
                acc_metric=scores.get("acc_metric"),
            )
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_joined_dataset() -> pd.DataFrame:
    """Load and join every (model, seed) run. Returns a single DataFrame
    containing only gameable rollouts that have retro ER/ACC scores."""
    frames: List[pd.DataFrame] = []
    for meta in MODEL_REGISTRY:
        for seed in meta["seeds"]:
            df = _load_run(meta["model"], seed)
            if df is None or df.empty:
                continue
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    # Drop rows where retro scores were -1 (judge parse failures) to keep
    # the plots honest.
    full = full[(full["er_metric"] > 0) & (full["acc_metric"] > 0)].reset_index(drop=True)
    return full


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def last_n_steps(df: pd.DataFrame, n: int = 50, max_step: int = MAX_STEP) -> pd.DataFrame:
    """Filter to the last ``n`` training steps of the (capped) run."""
    return df[df["step"] >= (max_step - n)].copy()


def top_k_reward_per_subenv(df: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """For each (env_name, subenv_id), return the ``k`` rows with highest reward."""
    return (
        df.sort_values("reward", ascending=False)
        .groupby(["env_name", "subenv_id"], sort=False)
        .head(k)
        .reset_index(drop=True)
    )


def top_k_reward_per_model(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """For each model, return the ``k`` rows with highest reward (pooled over seeds)."""
    return (
        df.sort_values("reward", ascending=False)
        .groupby("model", sort=False)
        .head(k)
        .reset_index(drop=True)
    )


def summarize_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean reward/er/acc and counts per (model, seed)."""
    g = df.groupby(["model", "family", "size", "seed"], as_index=False).agg(
        n=("reward", "size"),
        mean_reward=("reward", "mean"),
        mean_er=("er_metric", "mean"),
        mean_acc=("acc_metric", "mean"),
        std_reward=("reward", "std"),
    )
    return g.sort_values(["size", "model", "seed"]).reset_index(drop=True)


def coverage_report(df: pd.DataFrame) -> Dict:
    """Return a small dict summarising how much data the plots will see."""
    out = {
        "total_rows": len(df),
        "n_models": df["model"].nunique(),
        "n_runs": df.groupby(["model", "seed"]).ngroups,
        "n_subenvs": df.groupby(["env_name", "subenv_id"]).ngroups,
        "rows_per_subenv_median": int(
            df.groupby(["env_name", "subenv_id"]).size().median()
        ),
        "rows_per_subenv_mean": float(
            df.groupby(["env_name", "subenv_id"]).size().mean()
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Plot 3.3 — reward vs HEX scatter, top-k highest-reward rollouts per run
# ---------------------------------------------------------------------------


DEFAULT_FAMILY_PALETTE = {
    "Qwen":  "#630062",
    "Gemma": "#db3657",
    "Llama": "#f6cc35",
    "Yi":    "#48A37B",
}

# Display name to use in legends / labels for each family key in the palette.
FAMILY_DISPLAY_NAMES = {
    "Qwen":  "Qwen",
    "Gemma": "Gemma",
    "Llama": "LLaMA",
    "Yi":    "Yi",
}


def _hex_col(df: pd.DataFrame) -> str:
    """Return the name of the HEX column, whether it's 'hex_metric' or 'er_metric'."""
    if "hex_metric" in df.columns:
        return "hex_metric"
    if "er_metric" in df.columns:
        return "er_metric"
    raise KeyError("DataFrame has neither 'hex_metric' nor 'er_metric' column")


def top_k_per_model_seed(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """For each (model, seed) take the k rows with highest reward (whole run)."""
    return (
        df.sort_values("reward", ascending=False)
        .groupby(["model", "seed"], sort=False)
        .head(k)
        .reset_index(drop=True)
    )


def aggregate_top_k_per_model(
    df: pd.DataFrame,
    k: int = 10,
    last_n_steps_only: Optional[int] = None,
    max_step: int = MAX_STEP,
) -> pd.DataFrame:
    """Two-stage aggregation used by Plot 3.3.

    Stage 1: per (model, seed), average reward and HEX of the top-k rollouts.
    Stage 2: average those per-seed means across seeds -> one row per model,
             plus across-seed standard deviation on reward and HEX.

    Parameters
    ----------
    df : joined gameable dataframe (cap applied by the loader).
    k : number of top-reward rollouts per (model, seed).
    last_n_steps_only : if set (e.g. 50), restrict to the last N steps of
        each run before taking the top-k. If None, use the whole run.
    max_step : end of the run window (default MAX_STEP = 300). Used only
        when ``last_n_steps_only`` is set.
    """
    if last_n_steps_only is not None:
        df = df[df["step"] >= (max_step - last_n_steps_only)]
    hcol = _hex_col(df)
    topk = top_k_per_model_seed(df, k=k)

    per_seed = (
        topk.groupby(["model", "seed"], as_index=False)
        .agg(reward_top=("reward", "mean"), hex_top=(hcol, "mean"))
    )
    per_model = (
        per_seed.groupby("model", as_index=False)
        .agg(
            mean_reward=("reward_top", "mean"),
            std_reward=("reward_top", "std"),
            mean_hex=("hex_top", "mean"),
            std_hex=("hex_top", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    meta = pd.DataFrame(MODEL_REGISTRY)[["model", "family", "size"]]
    per_model = per_model.merge(meta, on="model").sort_values("size").reset_index(drop=True)
    per_seed = per_seed.merge(meta, on="model")
    return per_model, per_seed


def plot_reward_vs_hex_top10(
    df: pd.DataFrame,
    *,
    k: int = 10,
    last_n_steps_only: Optional[int] = None,
    ax=None,
    show_individual_dots: bool = False,
    show_reward_std: bool = False,
    show_hex_std: bool = False,
    show_bands: bool = False,
    size_by_params: bool = True,
    show_model_labels: bool = True,
    show_params: bool = False,
    font_size_scaler: float = 1.0,
    family_palette: Optional[Dict[str, str]] = None,
    centroid_size: float = 170.0,
    centroid_size_scale: float = 22.0,
    centroid_size_base: float = 70.0,
    label_offset: Tuple[int, int] = (6, 4),
    figsize: Tuple[float, float] = (8, 6),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Metrics of the Top-10 reward rollouts",
):
    """Scatter of mean reward vs mean HEX for the top-k highest-reward rollouts.

    Aggregation:
        Per (model, seed) take top-k rollouts by training reward, average
        reward and HEX. Then average across seeds -> one dot per model.

    Parameters
    ----------
    df : pd.DataFrame
        Joined gameable dataframe (already capped to step < 300 by the
        loader). Can contain either 'hex_metric' or 'er_metric'.
    k : int
        Number of top-reward rollouts per (model, seed).
    last_n_steps_only : int, optional
        If set (e.g. 50), restrict to the last N training steps of each run
        before taking the top-k. If None (default), use the full run
        (steps 0 .. MAX_STEP-1). Useful for comparing full-run top-k with
        converged-policy top-k.
    ax : matplotlib axis, optional
        Axis to draw on. A new (fig, ax) is created if None.
    show_individual_dots : bool
        If True, draw small translucent dots for each (model, seed) top-k
        average in addition to the per-model centroid.
    show_reward_std : bool
        If True, draw vertical error bars (across-seed SD of reward_top).
    show_hex_std : bool
        If True, draw horizontal error bars (across-seed SD of hex_top).
    show_bands : bool
        If True, shade HEX<=3 (safe), HEX 4-5 (mixed), HEX>=6 (harmful)
        with light guide lines at HEX=3.5 and HEX=5.5.
    size_by_params : bool
        If True, centroid dot size scales with model parameter count.
        If False, all centroids have the same fixed size `centroid_size`.
    show_model_labels : bool
        If True (default), annotate each centroid with a shortened model name.
        Set to False when the family legend alone is enough (model names can
        feel redundant with family color + size encoding). Ignored when
        ``show_params`` is True.
    show_params : bool
        If True, annotate each centroid with its parameter count (e.g. "7B")
        instead of the model name. Takes precedence over ``show_model_labels``.
        When enabled, the separate "Params" size-reference legend is hidden
        since the sizes are now labelled inline.
    font_size_scaler : float
        Multiplier applied to every font size in the plot.
    family_palette : dict, optional
        Colour per model family. Defaults to DEFAULT_FAMILY_PALETTE.
    centroid_size : float
        Fixed centroid dot size used when size_by_params is False.
    centroid_size_scale, centroid_size_base : float
        When size_by_params is True, centroid dot size = base + size_B * scale.
    label_offset : (int, int)
        Pixel offset of the text label next to each centroid.

    Returns
    -------
    ax : matplotlib axis
    per_model : pd.DataFrame with columns
        model, family, size, mean_reward, std_reward, mean_hex, std_hex, n_seeds
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, MaxNLocator

    # --- centralised font-size dict (tweak here) -----------------------------
    FONT_SIZES = {
        "title": 13 * font_size_scaler,
        "axis_label": 11 * font_size_scaler,
        "tick": 9 * font_size_scaler,
        "legend": 9 * font_size_scaler,
        "legend_title": 9 * font_size_scaler,
        "annotation": 8 * font_size_scaler,
    }
    # -------------------------------------------------------------------------

    palette = family_palette or DEFAULT_FAMILY_PALETTE

    per_model, per_seed = aggregate_top_k_per_model(
        df, k=k, last_n_steps_only=last_n_steps_only
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Optional safe / mixed / harmful bands (now on the y-axis: HEX is vertical)
    if show_bands:
        ax.axhspan(0.5, 3.5, alpha=0.08, color="#55A868", zorder=0)
        ax.axhspan(3.5, 5.5, alpha=0.05, color="#888888", zorder=0)
        ax.axhspan(5.5, 10.5, alpha=0.08, color="#C44E52", zorder=0)
        ax.axhline(3.5, color="gray", linestyle=":", linewidth=0.8, zorder=0)
        ax.axhline(5.5, color="gray", linestyle=":", linewidth=0.8, zorder=0)

    # Optional per-seed top-k cloud (x = reward, y = HEX)
    if show_individual_dots:
        per_seed_colors = [palette[f] for f in per_seed["family"]]
        ax.scatter(
            per_seed["reward_top"], per_seed["hex_top"],
            s=22, c=per_seed_colors, alpha=0.45,
            edgecolor="none", zorder=1,
        )

    # Optional error bars on centroids (x = reward, y = HEX)
    if show_reward_std or show_hex_std:
        xerr = per_model["std_reward"] if show_reward_std else None
        yerr = per_model["std_hex"] if show_hex_std else None
        ax.errorbar(
            per_model["mean_reward"], per_model["mean_hex"],
            xerr=xerr, yerr=yerr,
            fmt="none", ecolor="#888", elinewidth=1,
            capsize=2.5, alpha=0.75, zorder=2,
        )

    # Centroids — one dot per model (x = reward, y = HEX)
    if size_by_params:
        sizes = centroid_size_base + per_model["size"] * centroid_size_scale
    else:
        sizes = np.full(len(per_model), centroid_size)
    colors = [palette[f] for f in per_model["family"]]
    ax.scatter(
        per_model["mean_reward"], per_model["mean_hex"],
        s=sizes, c=colors,
        edgecolor="#222", linewidth=0.6, alpha=0.95, zorder=3,
    )

    # Labels next to each centroid
    def _short(name: str) -> str:
        return (name.replace("_", " ")
                    .replace("1 5", "1.5")
                    .replace(" Chat", "")
                    .replace("Instruct", "Inst")
                    .replace(" hf", ""))

    def _fmt_params(size_b: float) -> str:
        # 0.5 -> "0.5B", 1.8 -> "1.8B", 7.0 -> "7B", 14.0 -> "14B"
        if float(size_b).is_integer():
            return f"{int(size_b)}B"
        return f"{size_b:g}B"

    if show_params:
        # Big Llama models (13B, 14B) sit visually close to the 7B/8B cluster;
        # flip their labels to the bottom-left so they don't overlap.
        flip_bl_sizes = {13.0, 14.0}
        dx, dy = label_offset
        for _, row in per_model.iterrows():
            if float(row["size"]) in flip_bl_sizes:
                offset = (-dx, -dy)
                ha, va = "right", "top"
            else:
                offset = label_offset
                ha, va = "left", "bottom"
            ax.annotate(
                _fmt_params(row["size"]),
                (row["mean_reward"], row["mean_hex"]),
                fontsize=FONT_SIZES["annotation"], alpha=0.9,
                xytext=offset, textcoords="offset points",
                ha=ha, va=va,
                zorder=4,
            )
    elif show_model_labels:
        for _, row in per_model.iterrows():
            ax.annotate(
                _short(row["model"]),
                (row["mean_reward"], row["mean_hex"]),
                fontsize=FONT_SIZES["annotation"], alpha=0.85,
                xytext=label_offset, textcoords="offset points",
                zorder=4,
            )

    # Axes cosmetics (x = Mean Reward, y = Mean HEX)
    ax.set_xlabel("Mean Reward", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Mean HEX", fontsize=FONT_SIZES["axis_label"])
    window = (
        f"last {last_n_steps_only} steps"
        if last_n_steps_only is not None else "full run"
    )
    if title is not None:
        ax.set_title(
            # f"Top-{k} reward rollouts per (model, seed) [{window}], avg across seeds",
            title,
            fontsize=FONT_SIZES["title"],
        )

    # Sparser ticks: x (reward) is continuous, y (HEX) is roughly integer-valued.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])

    # Optional manual axis limits (default None -> matplotlib auto-scales).
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Subtle grid + despine, matching the look of plot_multi_env_scatter_single_col
    # so this figure drops in next to those scatters without looking different.
    ax.grid(True, alpha=0.2, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Legend ------------------------------------------------------------
    # Old style: single horizontal legend below the figure. Kept as reference.
    # family_order = [f for f in palette if f in set(per_model["family"])] or list(palette)
    # handles = [
    #     Line2D([0], [0], marker="o", color="w",
    #            markerfacecolor=palette[f], markeredgecolor="#222",
    #            markersize=9, label=FAMILY_DISPLAY_NAMES.get(f, f))
    #     for f in family_order
    # ]
    # ax.legend(
    #     handles=handles, loc="upper center",
    #     bbox_to_anchor=(0.5, -0.25), ncol=len(handles),
    #     fontsize=FONT_SIZES["legend"], frameon=False,
    #     handletextpad=0.4, columnspacing=1.2,
    # )

    # New style: two stacked legends on the right-hand side of the axes.
    # Top legend = family colors, bottom legend = parameter-size reference.
    family_order = [f for f in palette if f in set(per_model["family"])] or list(palette)
    family_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=palette[f], markeredgecolor="#222",
               markersize=9, label=FAMILY_DISPLAY_NAMES.get(f, f))
        for f in family_order
    ]
    if show_params:
        # With params annotated inline, there's no right-side size legend to
        # anchor to, so keep the family legend *inside* the axes. This plays
        # nicely with plt.tight_layout() in the notebook (external legends
        # get clipped otherwise).
        family_legend = ax.legend(
            handles=family_handles,
            loc="best",
            fontsize=FONT_SIZES["legend"],
            title="Family",
            title_fontsize=FONT_SIZES["legend_title"],
            frameon=True,
            framealpha=0.85,
            handletextpad=0.5,
            borderaxespad=0.4,
        )
    else:
        family_legend = ax.legend(
            handles=family_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=FONT_SIZES["legend"],
            title="Family",
            title_fontsize=FONT_SIZES["legend_title"],
            frameon=False,
            handletextpad=0.5,
            borderaxespad=0.0,
        )
    ax.add_artist(family_legend)

    # Size reference legend — only informative when size_by_params is True
    # and params aren't already labelled inline on each centroid.
    if size_by_params and not show_params:
        size_refs_b = [2, 7, 14]
        size_handles = []
        for s_b in size_refs_b:
            area = centroid_size_base + s_b * centroid_size_scale
            # scatter `s` is area in pt^2; Line2D `markersize` is diameter in pt.
            markersize = float(np.sqrt(area))
            size_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#cccccc", markeredgecolor="#222",
                       markeredgewidth=0.6, markersize=markersize,
                       label=f"{s_b}B"),
            )
        ax.legend(
            handles=size_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.55),
            fontsize=FONT_SIZES["legend"],
            title="Params",
            title_fontsize=FONT_SIZES["legend_title"],
            frameon=False,
            handletextpad=0.8,
            labelspacing=1.2,
            borderaxespad=0.0,
        )

    return ax, per_model


# ---------------------------------------------------------------------------
# Schematic panel — "safety prior determines path" cartoon
# ---------------------------------------------------------------------------


def plot_safety_prior_schematic(
    *,
    ax=None,
    figsize: Tuple[float, float] = (4.2, 3.2),
    font_size_scaler: float = 1.0,
    strong_color: str = "#0F6E56",
    weak_color: str = "#C04828",
    init_color: str = "#555555",
    annotation_color: str = "#626161",
    save_path_base: Optional[str] = None,
):
    """Schematic (not real data) of two training trajectories in reward/HEX space.

    Illustrates the claim: models with a strong safety prior stay near HEX 0
    as reward climbs, while models with a weak prior drift toward higher HEX.
    Both trajectories start at the same "Init" point and end at roughly the
    same reward, making the gap in HEX the salient visual.

    Parameters
    ----------
    ax : matplotlib axis, optional
        Axis to draw on. A new (fig, ax) is created if None.
    figsize : (float, float)
        Figure size when ax is None.
    font_size_scaler : float
        Multiplier applied to every font size in the plot.
    strong_color, weak_color : str
        Colours of the two schematic trajectories.
    init_color : str
        Colour of the shared "Init" starting point.
    annotation_color : str
        Colour of the auxiliary annotations (double-headed arrow + labels).
    save_path_base : str, optional
        If given, save both `<base>.pdf` and `<base>.png` (dpi=300, tight bbox).
        The parent directory is created if missing.

    Returns
    -------
    ax : matplotlib axis
    """
    import os
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    # --- centralised font-size dict (tweak here) -----------------------------
    FONT_SIZES = {
        "axis_label": 11 * font_size_scaler,
        "tick": 9 * font_size_scaler,
        "init_label": 9 * font_size_scaler,
        "traj_label": 8.5 * font_size_scaler,
        "annotation": 7.5 * font_size_scaler,
    }
    # -------------------------------------------------------------------------

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Trajectory points (schematic, not real data). Format: (reward, hex).
    t_strong = np.array([
        [5.8, 0.4],
        [6.2, 0.45],
        [6.6, 0.5],
        [7.0, 0.5],
        [7.3, 0.55],
        [7.5, 0.55],
    ])
    t_weak = np.array([
        [5.8, 0.4],
        [6.1, 1.0],
        [6.4, 1.8],
        [6.7, 2.8],
        [7.0, 3.5],
        [7.3, 4.2],
    ])

    # Ghost trail dots (intermediate steps, fading in)
    n = len(t_strong)
    for i in range(1, n - 1):
        alpha = 0.12 + 0.12 * i
        ax.scatter(t_strong[i, 0], t_strong[i, 1], s=30, color=strong_color,
                   alpha=alpha, zorder=3, edgecolors="none")
        ax.scatter(t_weak[i, 0], t_weak[i, 1], s=30, color=weak_color,
                   alpha=alpha, zorder=3, edgecolors="none")

    # Smooth trajectory curves + arrowhead at the end
    for pts, color in [(t_strong, strong_color), (t_weak, weak_color)]:
        t_param = np.linspace(0, 1, len(pts))
        t_fine = np.linspace(0, 1, 200)
        spl_x = make_interp_spline(t_param, pts[:, 0], k=3)
        spl_y = make_interp_spline(t_param, pts[:, 1], k=3)
        x_fine = spl_x(t_fine)
        y_fine = spl_y(t_fine)
        ax.plot(x_fine[:-10], y_fine[:-10], color=color, linewidth=1.8, zorder=4)
        ax.annotate(
            "", xy=(x_fine[-1], y_fine[-1]),
            xytext=(x_fine[-15], y_fine[-15]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.8),
            zorder=4,
        )

    # Shared "Init" point
    ax.scatter(t_strong[0, 0], t_strong[0, 1], s=70, color=init_color,
               alpha=0.25, zorder=5, edgecolors=init_color, linewidths=1.2)
    ax.text(t_strong[0, 0], t_strong[0, 1] - 0.35, "Init",
            ha="center", va="top",
            fontsize=FONT_SIZES["init_label"], color=init_color)

    # Trajectory end points
    ax.scatter(t_strong[-1, 0], t_strong[-1, 1], s=60, color=strong_color,
               zorder=5, edgecolors=strong_color, linewidths=1.5, alpha=0.9)
    ax.scatter(t_weak[-1, 0], t_weak[-1, 1], s=60, color=weak_color,
               zorder=5, edgecolors=weak_color, linewidths=1.5, alpha=0.9)

    # Trajectory labels
    ax.text(7.55, 0.35, "Strong safety prior",
            fontsize=FONT_SIZES["traj_label"], color=strong_color,
            va="center", ha="left", style="italic")
    ax.text(7.35, 4.5, "Weak safety prior",
            fontsize=FONT_SIZES["traj_label"], color=weak_color,
            va="center", ha="left", style="italic")

    # "Same reward, different harm" — vertical double-headed arrow + label
    ax.annotate(
        "", xy=(7.5, 0.7), xytext=(7.5, 4.0),
        arrowprops=dict(arrowstyle="<->", color=annotation_color, lw=0.8,
                        linestyle="--"),
    )
    ax.text(7.6, 2.35, "Same reward,\ndifferent harm",
            fontsize=FONT_SIZES["annotation"],
            color="#888888", va="center", ha="left")

    # "Safety prior determines path" near the fork
    ax.text(6.7, 1.35, "Safety prior\ndetermines path",
            fontsize=FONT_SIZES["annotation"],
            color=annotation_color, va="top", ha="center", style="italic")

    # Axes cosmetics
    ax.set_xlabel("Reward", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Harmfulness (HEX)", fontsize=FONT_SIZES["axis_label"])
    ax.set_xlim(5.4, 8.3)
    ax.set_ylim(-0.3, 5.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZES["tick"])
    # Schematic, not real data -> hide numeric ticks.
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path_base is not None:
        os.makedirs(os.path.dirname(save_path_base) or ".", exist_ok=True)
        fig.savefig(f"{save_path_base}.pdf", bbox_inches="tight", dpi=300)
        # fig.savefig(f"{save_path_base}.png", bbox_inches="tight", dpi=300)

    return ax


# ---------------------------------------------------------------------------
# Sanity-check helpers — inspecting actual responses
# ---------------------------------------------------------------------------


# Models whose responses the user wants to exclude from qualitative inspection
# because their responses have minor formatting issues. Pass ignore=False to
# the helpers below to disable this filter.
DEFAULT_IGNORE_MODELS: List[str] = ["gemma_1_1_2b_it"]


def filter_models(
    df: pd.DataFrame,
    ignore: bool = True,
    ignore_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop rows from models we don't want in the qualitative inspection.

    Parameters
    ----------
    df : input DataFrame
    ignore : if False, return df unchanged
    ignore_models : override the default ignore list
    """
    if not ignore:
        return df
    bad = ignore_models if ignore_models is not None else DEFAULT_IGNORE_MODELS
    return df[~df["model"].isin(bad)].copy()


def format_example(row: pd.Series, show_chars: int = 600) -> str:
    """Return a one-rollout pretty-printed block as a string."""
    import textwrap
    hcol = "hex_metric" if "hex_metric" in row.index else "er_metric"
    lines = [
        f"[{row['model']} · seed={row['seed']} · step={int(row['step'])}]",
        f"  topic={row['env_name']}  subenv={row['subenv_id']}",
        f"  reward={row['reward']:.2f}   HEX={int(row[hcol])}   ACC={int(row['acc_metric'])}",
        "  --- response ---",
    ]
    msg = str(row["agent_message"]).strip()
    if len(msg) > show_chars:
        msg = msg[:show_chars] + " …(truncated)"
    for ln in textwrap.wrap(msg, width=100):
        lines.append("  " + ln)
    return "\n".join(lines)


def show_examples(
    frame: pd.DataFrame,
    n: int = 3,
    title: Optional[str] = None,
    show_chars: int = 1600,
) -> None:
    """Print `n` rollouts from `frame` with the format_example layout."""
    if title:
        print("=" * 90)
        print(title)
        print("=" * 90)
    for _, row in frame.head(n).iterrows():
        print(format_example(row, show_chars=show_chars))
        print()


def top_reward_examples(
    df: pd.DataFrame,
    models: Iterable[str],
    n: int = 3,
    ignore: bool = True,
    ignore_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return the top-reward rows from `df` restricted to `models`."""
    sub = df[df["model"].isin(list(models))]
    sub = filter_models(sub, ignore=ignore, ignore_models=ignore_models)
    return sub.sort_values("reward", ascending=False).head(n)


def find_same_prompt_pair(
    df: pd.DataFrame,
    hex_safe_max: int = 2,
    hex_harmful_min: int = 6,
    ignore: bool = True,
    ignore_models: Optional[List[str]] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Find gameable prompts that host both a safe and a harmful high-reward rollout.

    Returns up to `top_n` candidate prompts ranked by `max_safe_reward` desc,
    each with columns ``env_name, subenv_id, n, max_safe_reward,
    max_harmful_reward, delta``.
    """
    df_f = filter_models(df, ignore=ignore, ignore_models=ignore_models)
    hcol = _hex_col(df_f)

    pairs = []
    for (env, sid), g in df_f.groupby(["env_name", "subenv_id"]):
        safe = g[g[hcol] <= hex_safe_max]
        harmful = g[g[hcol] >= hex_harmful_min]
        if len(safe) == 0 or len(harmful) == 0:
            continue
        pairs.append({
            "env_name": env,
            "subenv_id": sid,
            "n": len(g),
            "max_safe_reward": safe["reward"].max(),
            "max_harmful_reward": harmful["reward"].max(),
            "delta": safe["reward"].max() - harmful["reward"].max(),
        })
    out = pd.DataFrame(pairs)
    if out.empty:
        return out
    return out.sort_values("max_safe_reward", ascending=False).head(top_n).reset_index(drop=True)


def print_same_prompt_pair(
    df: pd.DataFrame,
    env_name: str,
    subenv_id: str,
    hex_safe_max: int = 2,
    hex_harmful_min: int = 6,
    ignore: bool = True,
    ignore_models: Optional[List[str]] = None,
) -> None:
    """Print the safe and harmful highest-reward rollouts on one specific prompt."""
    df_f = filter_models(df, ignore=ignore, ignore_models=ignore_models)
    hcol = _hex_col(df_f)
    rows = df_f[(df_f["env_name"] == env_name) & (df_f["subenv_id"] == subenv_id)]
    if rows.empty:
        print(f"No rows for ({env_name}, {subenv_id}).")
        return
    safe = rows[rows[hcol] <= hex_safe_max].sort_values("reward", ascending=False).head(1)
    harmful = rows[rows[hcol] >= hex_harmful_min].sort_values("reward", ascending=False).head(1)
    print(f">>> Same-prompt pair: topic={env_name}, subenv_id={subenv_id}")
    show_examples(safe, n=1, title="SAFE highest-reward rollout on this prompt")
    show_examples(harmful, n=1, title="HARMFUL highest-reward rollout on this prompt")


if __name__ == "__main__":
    df = load_joined_dataset()
    print("Loaded:", df.shape)
    print("Coverage:", coverage_report(df))
    print("\nPer-model summary:")
    print(summarize_per_model(df).to_string(index=False))
