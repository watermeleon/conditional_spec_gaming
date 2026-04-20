"""
Plot before/after gameability distributions for appendix figures.

For each environment (action_advice, political_qa), shows how the
gameability score distributions changed after dataset modification:
- "Before": original scored distributions (top/bottom 10% selected)
- "After": rescored distributions of the rewritten samples

Usage:
    python -m llm_agent.data_preprocess.plot_gameability_before_after \
        --env political_qa --save_dir figures/

    python -m llm_agent.data_preprocess.plot_gameability_before_after \
        --env action_advice --save_dir figures/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from llm_agent.data_preprocess.env_configs import load_env_config
from llm_agent.data_preprocess.gameability_scoring import load_all_scores, load_scores


# ── Styling ──────────────────────────────────────────────────────────────────

COLORS = {
    "original": "#bdbdbd",       # light grey — full original pool
    "selected_g": "#ef8a62",     # warm orange — selected gameable (before)
    "selected_ng": "#67a9cf",    # cool blue — selected non-gameable (before)
    "after_g": "#b2182b",        # dark red — gameable after rewrite
    "after_ng": "#2166ac",       # dark blue — non-gameable after rewrite
}

NICE_ENV_NAMES = {
    "action_advice": "Action Advice",
    "political_qa": "Political QA",
}

# Which side was rewritten per subenv in political_qa.
# "vuln" = gameable side rewritten to be more entrenched
# "normal" = non-gameable side rewritten to be more open-minded
POLITICAL_QA_REWRITTEN_SIDE = {
    "liberal": "vuln",
    "socialist": "vuln",
    "alt-right": "normal",
    "conservative": "normal",
}

NICE_SUBENV_NAMES = {
    # action_advice
    "drinking": "Drinking",
    "extreme_sports": "Extreme Sports",
    "gambling": "Gambling",
    "impulsive-decision": "Impulsive Decision",
    "neglecting-medication": "Neglecting Medication",
    "over-spending": "Over-spending",
    "smoking": "Smoking",
    # political_qa
    "alt-right": "Alt-right",
    "conservative": "Conservative",
    "liberal": "Liberal",
    "socialist": "Socialist",
}


def _load_history_ids(json_path: Path) -> set:
    """Load history IDs from a subenv JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    histories = data.get("histories", {})
    if isinstance(histories, dict):
        return set(histories.keys())
    # list of dicts with 'id' key
    return {str(h["id"]) for h in histories}


def _extract_selected_scores(
    all_scores_df: pd.DataFrame,
    subenv: str,
    gameable_ids: set,
    non_gameable_ids: set,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract scores of selected gameable/non-gameable samples from the full pool."""
    env_df = all_scores_df[all_scores_df["env_name"] == subenv].copy()
    g_df = env_df[env_df["history_id"].astype(str).isin(gameable_ids)]
    ng_df = env_df[env_df["history_id"].astype(str).isin(non_gameable_ids)]
    return g_df, ng_df


def plot_before_after(
    env: str,
    save_dir: Optional[str] = None,
    # figsize_per_subenv: tuple = (4.5, 3.0),
    figsize_per_subenv: tuple = (2.5, 2.2),
    bins: int = 25,
    score_range: tuple = (1, 10),
    font_scaler: float = 1.1,
) -> plt.Figure:
    """
    Create a figure with one row per subenvironment showing before/after
    gameability distributions side by side.

    Left column: "Before" — full original pool (grey) with selected
                 gameable (orange) and non-gameable (blue) highlighted.
    Right column: "After" — rescored gameable (red) and non-gameable (blue)
                  distributions after rewriting.

    Returns the Figure object.
    """
    # ── Font sizes (scaled by font_scaler) ───────────────────────────────
    _FS = {
        "suptitle": 12,
        "title":    11,
        "row_label": 10,
        "xlabel":     9,
        "legend":     7,
        "ticks":      8,
    }
    FS = {k: v * font_scaler for k, v in _FS.items()}

    cfg = load_env_config(env)
    subenvs = cfg.ALL_SUBENVS
    n_subenvs = len(subenvs)

    # ── Load original scores (full pool, excluding vuln_/normal_ files) ──
    scores_path = Path(cfg.SCORES_DIR)
    original_records = []
    for jsonl_file in sorted(scores_path.glob("*_scores.jsonl")):
        if jsonl_file.name.startswith(("vuln_", "normal_")):
            continue
        with open(jsonl_file) as f:
            for line in f:
                original_records.append(json.loads(line))
    df_original = pd.DataFrame(original_records)

    # ── Check if after-rewrite scores exist ──────────────────────────────
    scores_dir = Path(cfg.SCORES_DIR)
    has_after_scores = any(scores_dir.glob("vuln_*_scores.jsonl"))

    if has_after_scores:
        df_after_g = load_all_scores(scores_dir=cfg.SCORES_DIR, env=env, file_prefix="vuln_")
        df_after_ng = load_all_scores(scores_dir=cfg.SCORES_DIR, env=env, file_prefix="normal_")
    else:
        df_after_g = None
        df_after_ng = None

    # ── Identify which samples were selected (by matching history IDs) ───
    # Determine the directory where gameable/non-gameable JSONs live
    if env == "action_advice":
        g_dir = Path(cfg.GAMEABLE_V2_DIR)
        ng_dir = Path("llm_agent/environments/action_advice/chat_subenvs/non_gameable")
    else:
        g_dir = Path(cfg.GAMEABLE_V2_DIR)
        ng_dir = Path("llm_agent/environments/political_qa/non_gameable_v2")

    # ── Create figure ────────────────────────────────────────────────────
    ncols = 2 if has_after_scores else 1
    fig, axes = plt.subplots(
        n_subenvs, ncols,
        figsize=(figsize_per_subenv[0] * ncols, figsize_per_subenv[1] * n_subenvs),
        squeeze=False,
    )

    bin_edges = np.linspace(score_range[0], score_range[1], bins + 1)

    for row_idx, subenv in enumerate(subenvs):
        nice_name = NICE_SUBENV_NAMES.get(subenv, subenv)
        env_scores = df_original[df_original["env_name"] == subenv]["weighted_avg_score"]

        # Load selected IDs
        g_json = g_dir / f"vuln_{subenv}.json"
        ng_json = ng_dir / f"normal_{subenv}.json"
        g_ids = _load_history_ids(g_json) if g_json.exists() else set()
        ng_ids = _load_history_ids(ng_json) if ng_json.exists() else set()
        before_g, before_ng = _extract_selected_scores(df_original, subenv, g_ids, ng_ids)

        # ── Left panel: Before ───────────────────────────────────────────
        ax_before = axes[row_idx, 0]

        # Full pool (background)
        ax_before.hist(
            env_scores, bins=bin_edges, color=COLORS["original"],
            edgecolor="white", linewidth=0.3, alpha=0.6,
            label=f"Full pool (n={len(env_scores)})",
        )
        # Selected gameable (before rewrite)
        if len(before_g) > 0:
            ax_before.hist(
                before_g["weighted_avg_score"], bins=bin_edges,
                color=COLORS["selected_g"], edgecolor="white", linewidth=0.3,
                alpha=0.8,
                label=f"G (n={len(before_g)}, μ={before_g['weighted_avg_score'].mean():.1f})",
            )
        # Selected non-gameable (before rewrite)
        if len(before_ng) > 0:
            ax_before.hist(
                before_ng["weighted_avg_score"], bins=bin_edges,
                color=COLORS["selected_ng"], edgecolor="white", linewidth=0.3,
                alpha=0.8,
                label=f"NG (n={len(before_ng)}, μ={before_ng['weighted_avg_score'].mean():.1f})",
            )

        ax_before.set_ylabel(nice_name, fontsize=FS["row_label"], fontweight="bold")
        ax_before.set_yticks([]) 
        ax_before.set_xlim(score_range)
        ax_before.legend(fontsize=FS["legend"], loc="upper right")
        ax_before.tick_params(labelsize=FS["ticks"])
        if row_idx == 0:
            ax_before.set_title("Before rewriting", fontsize=FS["title"], fontweight="bold")
        if row_idx == n_subenvs - 1:
            ax_before.set_xlabel("Gameability score", fontsize=FS["xlabel"])

        # ── Right panel: After (if scores available) ─────────────────────
        if has_after_scores:
            ax_after = axes[row_idx, 1]

            # For political_qa, only one side was rewritten per subenv.
            # Use rescored data for the rewritten side, original scores
            # for the unchanged side.
            rewritten_side = POLITICAL_QA_REWRITTEN_SIDE.get(subenv) if env == "political_qa" else None

            g_rescored = df_after_g[df_after_g["env_name"] == subenv]["weighted_avg_score"]
            ng_rescored = df_after_ng[df_after_ng["env_name"] == subenv]["weighted_avg_score"]

            if rewritten_side == "normal":
                # Gameable side was NOT rewritten — use original scores
                g_scores_after = before_g["weighted_avg_score"] if len(before_g) > 0 else g_rescored
                ng_scores_after = ng_rescored
                g_label_suffix = ""
                ng_label_suffix = "_new"
            elif rewritten_side == "vuln":
                # Non-gameable side was NOT rewritten — use original scores
                g_scores_after = g_rescored
                ng_scores_after = before_ng["weighted_avg_score"] if len(before_ng) > 0 else ng_rescored
                g_label_suffix = "_new"
                ng_label_suffix = ""
            else:
                # action_advice or unknown — use rescored for both
                g_scores_after = g_rescored
                ng_scores_after = ng_rescored
                g_label_suffix = "_new"
                ng_label_suffix = "_new"

            if len(g_scores_after) > 0:
                ax_after.hist(
                    g_scores_after, bins=bin_edges,
                    color=COLORS["after_g"], edgecolor="white", linewidth=0.3,
                    alpha=0.7,
                    label=f"G{g_label_suffix} (n={len(g_scores_after)}, μ={g_scores_after.mean():.1f})",
                )
            if len(ng_scores_after) > 0:
                ax_after.hist(
                    ng_scores_after, bins=bin_edges,
                    color=COLORS["after_ng"], edgecolor="white", linewidth=0.3,
                    alpha=0.7,
                    label=f"NG{ng_label_suffix} (n={len(ng_scores_after)}, μ={ng_scores_after.mean():.1f})",
                )

            ax_after.set_xlim(score_range)
            ax_after.legend(fontsize=FS["legend"], loc="upper right")
            ax_after.tick_params(labelsize=FS["ticks"])
            ax_after.set_yticks([]) 
            if row_idx == 0:
                ax_after.set_title("After rewriting", fontsize=FS["title"], fontweight="bold")
            if row_idx == n_subenvs - 1:
                ax_after.set_xlabel("Gameability score", fontsize=FS["xlabel"])

    env_title = NICE_ENV_NAMES.get(env, env)
    fig.suptitle(
        f"{env_title} — Gameability distributions\n before & after dataset modification",
        fontsize=FS["suptitle"], fontweight="bold", y=.98,
    )
    fig.tight_layout()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fname = save_path / f"gameability_before_after_{env}.pdf"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved: {fname}")

    return fig


def plot_combined_overview(
    env: str,
    save_dir: Optional[str] = None,
    figsize: tuple = (7, 4),
    bins: int = 30,
    score_range: tuple = (1, 10),
) -> plt.Figure:
    """
    Single-panel overview: aggregate before/after across all subenvironments.
    Shows the combined gameable and non-gameable distributions as KDEs.
    """
    cfg = load_env_config(env)
    scores_dir = Path(cfg.SCORES_DIR)
    has_after = any(scores_dir.glob("vuln_*_scores.jsonl"))

    scores_path = Path(cfg.SCORES_DIR)
    original_records = []
    for jsonl_file in sorted(scores_path.glob("*_scores.jsonl")):
        if jsonl_file.name.startswith(("vuln_", "normal_")):
            continue
        with open(jsonl_file) as f:
            for line in f:
                original_records.append(json.loads(line))
    df_original = pd.DataFrame(original_records)

    fig, axes = plt.subplots(1, 2 if has_after else 1, figsize=figsize, squeeze=False)
    bin_edges = np.linspace(score_range[0], score_range[1], bins + 1)

    # Before: full pool
    ax = axes[0, 0]
    ax.hist(
        df_original["weighted_avg_score"], bins=bin_edges,
        color=COLORS["original"], edgecolor="white", linewidth=0.3,
        alpha=0.7, density=True,
        label=f"All samples (n={len(df_original)})",
    )
    ax.set_title("Before rewriting", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gameability score", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(score_range)

    if has_after:
        df_g = load_all_scores(scores_dir=cfg.SCORES_DIR, env=env, file_prefix="vuln_")
        df_ng = load_all_scores(scores_dir=cfg.SCORES_DIR, env=env, file_prefix="normal_")

        ax2 = axes[0, 1]
        ax2.hist(
            df_g["weighted_avg_score"], bins=bin_edges,
            color=COLORS["after_g"], edgecolor="white", linewidth=0.3,
            alpha=0.6, density=True,
            label=f"Gameable (n={len(df_g)}, μ={df_g['weighted_avg_score'].mean():.1f})",
        )
        ax2.hist(
            df_ng["weighted_avg_score"], bins=bin_edges,
            color=COLORS["after_ng"], edgecolor="white", linewidth=0.3,
            alpha=0.6, density=True,
            label=f"Non-gameable (n={len(df_ng)}, μ={df_ng['weighted_avg_score'].mean():.1f})",
        )
        ax2.set_title("After rewriting", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Gameability score", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.set_xlim(score_range)

    env_title = NICE_ENV_NAMES.get(env, env)
    fig.suptitle(f"{env_title} — Aggregate gameability", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fname = save_path / f"gameability_overview_{env}.pdf"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved: {fname}")

    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot before/after gameability distributions")
    parser.add_argument("--env", required=True, choices=["action_advice", "political_qa"])
    parser.add_argument("--save_dir", default="./figs", help="Directory to save PDF figures")
    parser.add_argument("--overview", action="store_true", help="Also plot combined overview")
    args = parser.parse_args()

    plot_before_after(args.env, save_dir=args.save_dir)
    if args.overview:
        plot_combined_overview(args.env, save_dir=args.save_dir)

    if not args.save_dir:
        plt.show()
