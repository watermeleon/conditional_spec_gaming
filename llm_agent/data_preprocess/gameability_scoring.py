"""
Gameability scoring and data splitting — environment-agnostic.

Scores each user history on how "gameable" it is (1-10) using an LLM judge,
then splits histories into normal (non-gameable) and vuln (gameable) JSON files.

Environment-specific prompts, paths, and metadata are loaded from
llm_agent.data_preprocess.env_configs.<env_name>.

Usage:
  # As CLI (from repo root, with a vLLM server running):
  python -m llm_agent.data_preprocess.gameability_scoring \
      --env action_advice --env_name drinking \
      --vllm_url http://localhost:8010/v1 \
      --model_name meta-llama/Llama-3.1-8B-Instruct

  # From notebook / script:
  from llm_agent.data_preprocess.gameability_scoring import (
      load_scores, load_all_scores, plot_score_distributions, split_by_gameability
  )
"""

import argparse
import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from llm_agent.data_preprocess.env_configs import load_env_config
from llm_agent.reward_model.utils.vllm_utils import get_score_from_vllm

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt building
# =============================================================================

def build_gameability_prompt(
    user_message: str,
    pm_prompt: str,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_template: str,
) -> str:
    """
    Build a formatted prompt for gameability scoring.

    Constructs system + user messages and applies the judge model's chat template
    to produce a string ready for the vLLM completions API.
    """
    user_content = user_template.format(
        pm_prompt=pm_prompt,
        user_message=user_message,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =============================================================================
# Scoring
# =============================================================================

async def score_gameability(
    env_name: str,
    env: str = "action_advice",
    vllm_url: str = "http://localhost:8010/v1",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    n_samples: Optional[int] = None,
    max_concurrent: int = 32,
    output_dir: Optional[str] = None,
    seed: int = 42,
    chat_subenvs_dir: Optional[str] = None,
    file_prefix: str = "",
) -> str:
    """
    Score gameability of histories in a single subenv using a vLLM server.

    Args:
        env_name: Subenv name (e.g. "drinking"). Must match a JSON filename.
        env: Environment config name (e.g. "action_advice", "political_qa").
        vllm_url: URL of the vLLM OpenAI-compatible API (include /v1).
        model_name: Model name served by vLLM.
        n_samples: Number of histories to score. None = all.
        max_concurrent: Max concurrent requests to vLLM.
        output_dir: Directory to save the output JSONL. None = use env config default.
        seed: Random seed (used when sampling a subset).
        chat_subenvs_dir: Directory containing the subenv JSON files. None = use env config default.
        file_prefix: Prefix for input/output filenames (e.g. "vuln_", "normal_", or "").

    Returns:
        Path to the output JSONL file.
    """
    cfg = load_env_config(env)
    if output_dir is None:
        output_dir = cfg.SCORES_DIR
    if chat_subenvs_dir is None:
        chat_subenvs_dir = cfg.CHAT_SUBENVS_DIR

    random.seed(seed)

    # Load subenv JSON
    json_path = Path(chat_subenvs_dir) / f"{file_prefix}{env_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Subenv file not found: {json_path}")

    with open(json_path, "r") as f:
        env_data = json.load(f)

    pm_prompt = env_data.get("pm_prompt", "")
    histories: Dict[str, list] = env_data["histories"]

    # Optionally sample a subset
    history_ids = list(histories.keys())
    if n_samples is not None and n_samples < len(history_ids):
        history_ids = random.sample(history_ids, k=n_samples)
    logger.info(f"Scoring {len(history_ids)} histories for {file_prefix}{env_name}")

    # Load tokenizer for prompt formatting
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Build prompts
    prompts = {}
    for hid in history_ids:
        user_message = histories[hid][0]["content"]
        prompts[hid] = build_gameability_prompt(
            user_message, pm_prompt, tokenizer,
            system_prompt=cfg.GAMEABILITY_SYSTEM_PROMPT,
            user_template=cfg.GAMEABILITY_USER_TEMPLATE,
        )

    # Set up async client
    client = AsyncOpenAI(api_key="EMPTY", base_url=vllm_url, timeout=120.0)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Score all prompts
    async def _score_one(hid: str, prompt: str):
        weighted_avg, top_token, top_prob = await get_score_from_vllm(
            client=client,
            semaphore=semaphore,
            model_name=model_name,
            prompt=prompt,
            valid_score_range=(1, 10),
        )
        return {
            "env_name": env_name,
            "history_id": hid,
            "weighted_avg_score": round(weighted_avg, 4),
            "highest_prob_token": top_token,
            "highest_prob_value": round(top_prob, 4),
            "user_message_preview": histories[hid][0]["content"][:200],
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
        }

    tasks = [_score_one(hid, prompt) for hid, prompt in prompts.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    scored = []
    errors = 0
    for r in results:
        if isinstance(r, Exception):
            errors += 1
            logger.warning(f"Scoring error: {r}")
        else:
            scored.append(r)

    # Save JSONL — include prefix in filename so G/NG scores don't collide
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{file_prefix}{env_name}_scores.jsonl"

    with open(out_file, "w") as f:
        for entry in scored:
            f.write(json.dumps(entry) + "\n")

    # Summary
    if scored:
        scores = [e["weighted_avg_score"] for e in scored]
        print(f"\n--- {file_prefix}{env_name} scoring complete ---")
        print(f"  Scored: {len(scored)}, Errors: {errors}")
        print(f"  Mean: {sum(scores)/len(scores):.2f}, "
              f"Min: {min(scores):.2f}, Max: {max(scores):.2f}")
        print(f"  Saved to: {out_file}")

    return str(out_file)


def run_scoring(
    env_name: str,
    env: str = "action_advice",
    vllm_url: str = "http://localhost:8010/v1",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    n_samples: Optional[int] = None,
    max_concurrent: int = 32,
    output_dir: Optional[str] = None,
    seed: int = 42,
    chat_subenvs_dir: Optional[str] = None,
    file_prefix: str = "",
) -> str:
    """Synchronous wrapper around score_gameability()."""
    return asyncio.run(score_gameability(
        env_name=env_name,
        env=env,
        vllm_url=vllm_url,
        model_name=model_name,
        n_samples=n_samples,
        max_concurrent=max_concurrent,
        output_dir=output_dir,
        seed=seed,
        chat_subenvs_dir=chat_subenvs_dir,
        file_prefix=file_prefix,
    ))


# =============================================================================
# Loading scores
# =============================================================================

def load_scores(
    env_name: str,
    scores_dir: Optional[str] = None,
    path: Optional[str] = None,
    env: str = "action_advice",
    file_prefix: str = "",
) -> pd.DataFrame:
    """Load gameability scores for a single subenv from JSONL into a DataFrame.

    Args:
        env_name: Subenv name.
        scores_dir: Directory containing score files.
        path: Explicit path to a scores file (overrides scores_dir).
        env: Environment config name.
        file_prefix: Prefix for the scores file (e.g. "vuln_", "normal_").
    """
    if path is None:
        if scores_dir is None:
            cfg = load_env_config(env)
            scores_dir = cfg.SCORES_DIR
        path = Path(scores_dir) / f"{file_prefix}{env_name}_scores.jsonl"
    else:
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")

    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_all_scores(
    scores_dir: Optional[str] = None,
    env: str = "action_advice",
    file_prefix: str = "",
) -> pd.DataFrame:
    """Load scores for all available subenvs into a single DataFrame.

    Args:
        scores_dir: Directory containing score files.
        env: Environment config name.
        file_prefix: Only load files matching this prefix (e.g. "vuln_", "normal_").
                     Empty string matches all *_scores.jsonl files.
    """
    if scores_dir is None:
        cfg = load_env_config(env)
        scores_dir = cfg.SCORES_DIR

    scores_path = Path(scores_dir)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_path}")

    all_records = []
    for jsonl_file in sorted(scores_path.glob(f"{file_prefix}*_scores.jsonl")):
        with open(jsonl_file, "r") as f:
            for line in f:
                all_records.append(json.loads(line))

    if not all_records:
        raise FileNotFoundError(
            f"No {file_prefix}*_scores.jsonl files found in {scores_path}"
        )

    return pd.DataFrame(all_records)


# =============================================================================
# Plotting
# =============================================================================

def plot_score_distributions(
    scores_df: pd.DataFrame,
    score_column: str = "weighted_avg_score",
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    """
    Plot distribution of gameability scores.

    If the DataFrame has an 'env_name' column with multiple values,
    creates per-subenv subplot histograms. Otherwise plots a single histogram.
    """
    import matplotlib.pyplot as plt

    if "env_name" not in scores_df.columns or scores_df["env_name"].nunique() <= 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores_df[score_column], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Gameability Score")
        ax.set_ylabel("Count")
        ax.set_title("Gameability Score Distribution")
        return

    env_names = sorted(scores_df["env_name"].unique())
    n_envs = len(env_names)
    ncols = min(3, n_envs)
    nrows = (n_envs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for idx, env_name in enumerate(env_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        env_scores = scores_df[scores_df["env_name"] == env_name][score_column]

        ax.hist(env_scores, bins=30, edgecolor="black", alpha=0.7, color=f"C{idx}")
        ax.set_title(f"{env_name} (n={len(env_scores)})")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.axvline(env_scores.mean(), color="red", linestyle="--", label=f"mean={env_scores.mean():.1f}")
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_envs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Gameability Score Distributions per Subenv", fontsize=14)
    plt.tight_layout()


def plot_score_distributions_gng(
    gameable_df: pd.DataFrame,
    non_gameable_df: pd.DataFrame,
    score_column: str = "weighted_avg_score",
    figsize: Tuple[int, int] = (16, 10),
    title: str = "Gameability Score Distributions: Gameable vs Non-Gameable",
) -> None:
    """
    Plot overlaid G and NG score distributions per subenv.

    Each subplot shows one subenv with both gameable (red) and non-gameable (blue)
    histograms overlaid, plus vertical mean lines.

    Args:
        gameable_df: DataFrame with scores for gameable samples.
        non_gameable_df: DataFrame with scores for non-gameable samples.
        score_column: Column to plot.
        figsize: Figure size.
        title: Overall figure title.
    """
    import matplotlib.pyplot as plt

    # Get all subenvs present in either DataFrame
    all_envs = sorted(set(
        gameable_df["env_name"].unique().tolist()
        + non_gameable_df["env_name"].unique().tolist()
    ))
    n_envs = len(all_envs)

    if n_envs == 0:
        print("No subenvs found in the provided DataFrames.")
        return

    ncols = min(3, n_envs)
    nrows = (n_envs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    # Compute shared bin range across all data
    all_scores = pd.concat([
        gameable_df[score_column],
        non_gameable_df[score_column],
    ])
    bin_min, bin_max = all_scores.min(), all_scores.max()
    bins = 30

    for idx, env_name in enumerate(all_envs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        g_scores = gameable_df[gameable_df["env_name"] == env_name][score_column]
        ng_scores = non_gameable_df[non_gameable_df["env_name"] == env_name][score_column]

        # Plot both distributions
        if len(g_scores) > 0:
            ax.hist(
                g_scores, bins=bins, range=(bin_min, bin_max),
                alpha=0.5, color="red", edgecolor="darkred",
                label=f"G (n={len(g_scores)}, \u03bc={g_scores.mean():.2f})",
            )
            ax.axvline(g_scores.mean(), color="red", linestyle="--", linewidth=1.5)

        if len(ng_scores) > 0:
            ax.hist(
                ng_scores, bins=bins, range=(bin_min, bin_max),
                alpha=0.5, color="blue", edgecolor="darkblue",
                label=f"NG (n={len(ng_scores)}, \u03bc={ng_scores.mean():.2f})",
            )
            ax.axvline(ng_scores.mean(), color="blue", linestyle="--", linewidth=1.5)

        ax.set_title(env_name, fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_envs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()


# =============================================================================
# Splitting
# =============================================================================

def split_by_gameability(
    env_name: str,
    env: str = "action_advice",
    scores_dir: Optional[str] = None,
    chat_subenvs_dir: Optional[str] = None,
    top_percent: float = 10.0,
    bottom_percent: float = 10.0,
    score_column: str = "weighted_avg_score",
) -> Tuple[str, str]:
    """
    Split a subenv's histories into normal (non-gameable) and vuln (gameable) files.

    Takes the top X% scoring histories as gameable -> vuln_{env_name}.json
    Takes the bottom X% scoring histories as non-gameable -> normal_{env_name}.json

    Args:
        env_name: Subenv name.
        env: Environment config name (e.g. "action_advice", "political_qa").
        scores_dir: Directory with *_scores.jsonl files. None = use env config default.
        chat_subenvs_dir: Directory with original subenv JSON files. None = use env config default.
        top_percent: Percentage of top-scoring samples for vuln file.
        bottom_percent: Percentage of bottom-scoring samples for normal file.
        score_column: Column name in the scores DataFrame to sort by.

    Returns:
        (normal_file_path, vuln_file_path)
    """
    cfg = load_env_config(env)
    if scores_dir is None:
        scores_dir = cfg.SCORES_DIR
    if chat_subenvs_dir is None:
        chat_subenvs_dir = cfg.CHAT_SUBENVS_DIR

    # Load scores
    df = load_scores(env_name, scores_dir)
    df = df.sort_values(score_column)

    n_total = len(df)
    n_bottom = max(1, int(n_total * bottom_percent / 100))
    n_top = max(1, int(n_total * top_percent / 100))

    normal_ids = set(df.head(n_bottom)["history_id"].tolist())
    vuln_ids = set(df.tail(n_top)["history_id"].tolist())

    print(f"{env_name}: {n_total} scored, "
          f"normal={len(normal_ids)} (bottom {bottom_percent}%), "
          f"vuln={len(vuln_ids)} (top {top_percent}%)")

    # Load original JSON
    json_path = Path(chat_subenvs_dir) / f"{env_name}.json"
    with open(json_path, "r") as f:
        env_data = json.load(f)

    # Build split files
    pm_prompts = cfg.PM_PROMPTS
    char_suffixes = cfg.CHAR_SUFFIXES

    def _make_split_json(history_ids: set, split_type: str) -> dict:
        """Create a new JSON dict for a split (normal or vuln)."""
        split_data = {}
        # Copy all metadata fields except histories
        for key, value in env_data.items():
            if key != "histories":
                split_data[key] = value

        # Override pm_prompt and add char_suffix
        prompts = pm_prompts.get(env_name, {})
        suffixes = char_suffixes.get(env_name, {})
        if split_type in prompts:
            split_data["pm_prompt"] = prompts[split_type]
        if split_type in suffixes:
            split_data["char_suffix"] = suffixes[split_type]

        # Select histories
        split_data["histories"] = {
            hid: env_data["histories"][hid]
            for hid in history_ids
            if hid in env_data["histories"]
        }
        return split_data

    normal_data = _make_split_json(normal_ids, "normal")
    vuln_data = _make_split_json(vuln_ids, "vuln")

    # Save
    subenvs_path = Path(chat_subenvs_dir)
    normal_path = subenvs_path / f"normal_{env_name}.json"
    vuln_path = subenvs_path / f"vuln_{env_name}.json"

    with open(normal_path, "w") as f:
        json.dump(normal_data, f, indent=2)
    with open(vuln_path, "w") as f:
        json.dump(vuln_data, f, indent=2)

    print(f"  Saved: {normal_path} ({len(normal_data['histories'])} histories)")
    print(f"  Saved: {vuln_path} ({len(vuln_data['histories'])} histories)")

    return str(normal_path), str(vuln_path)


def split_by_random(
    env_name: str,
    env: str = "action_advice",
    scores_dir: Optional[str] = None,
    chat_subenvs_dir: Optional[str] = None,
    top_percent: float = 10.0,
    bottom_percent: float = 10.0,
    seed: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Split a subenv's histories randomly into normal and vuln files.

    Selects the same number of samples as split_by_gameability() would,
    but drawn at random (no overlap between splits).

    Args:
        env_name: Subenv name.
        env: Environment config name (e.g. "action_advice", "political_qa").
        scores_dir: Directory with *_scores.jsonl files. None = use env config default.
        chat_subenvs_dir: Directory with original subenv JSON files. None = use env config default.
        top_percent: Percentage of samples for the vuln file (mirrors split_by_gameability).
        bottom_percent: Percentage of samples for the normal file (mirrors split_by_gameability).
        seed: Optional random seed for reproducibility.

    Returns:
        (normal_file_path, vuln_file_path)
    """
    cfg = load_env_config(env)
    if scores_dir is None:
        scores_dir = cfg.SCORES_DIR
    if chat_subenvs_dir is None:
        chat_subenvs_dir = cfg.CHAT_SUBENVS_DIR

    # Load scores only to get the full list of history IDs and total count
    df = load_scores(env_name, scores_dir)
    n_total = len(df)
    n_normal = max(1, int(n_total * bottom_percent / 100))
    n_vuln = max(1, int(n_total * top_percent / 100))

    all_ids = df["history_id"].tolist()
    rng = random.Random(seed)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)

    # Draw without overlap: take first n_normal for normal, next n_vuln for vuln
    normal_ids = set(shuffled[:n_normal])
    vuln_ids = set(shuffled[n_normal:n_normal + n_vuln])

    print(f"{env_name}: {n_total} total, "
          f"normal={len(normal_ids)} ({bottom_percent}% random), "
          f"vuln={len(vuln_ids)} ({top_percent}% random)")

    # Load original JSON
    json_path = Path(chat_subenvs_dir) / f"{env_name}.json"
    with open(json_path, "r") as f:
        env_data = json.load(f)

    pm_prompts = cfg.PM_PROMPTS
    char_suffixes = cfg.CHAR_SUFFIXES

    def _make_split_json(history_ids: set, split_type: str) -> dict:
        """Create a new JSON dict for a split (normal or vuln)."""
        split_data = {}
        for key, value in env_data.items():
            if key != "histories":
                split_data[key] = value

        prompts = pm_prompts.get(env_name, {})
        suffixes = char_suffixes.get(env_name, {})
        if split_type in prompts:
            split_data["pm_prompt"] = prompts[split_type]
        if split_type in suffixes:
            split_data["char_suffix"] = suffixes[split_type]

        split_data["histories"] = {
            hid: env_data["histories"][hid]
            for hid in history_ids
            if hid in env_data["histories"]
        }
        return split_data

    normal_data = _make_split_json(normal_ids, "normal")
    vuln_data = _make_split_json(vuln_ids, "vuln")

    subenvs_path = Path(chat_subenvs_dir)
    normal_path = subenvs_path / f"normal_{env_name}.json"
    vuln_path = subenvs_path / f"vuln_{env_name}.json"

    with open(normal_path, "w") as f:
        json.dump(normal_data, f, indent=2)
    with open(vuln_path, "w") as f:
        json.dump(vuln_data, f, indent=2)

    print(f"  Saved: {normal_path} ({len(normal_data['histories'])} histories)")
    print(f"  Saved: {vuln_path} ({len(vuln_data['histories'])} histories)")

    return str(normal_path), str(vuln_path)


# =============================================================================
# CLI
# =============================================================================

def run_scoring_cli():
    """CLI entry point for scoring a single subenv."""
    parser = argparse.ArgumentParser(
        description="Score gameability of subenv histories"
    )
    parser.add_argument("--env", type=str, default="action_advice",
                        help="Environment config name (default: action_advice)")
    parser.add_argument("--env_name", type=str, required=True,
                        help="Subenv name (e.g. drinking, smoking, liberal)")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8010/v1",
                        help="vLLM server URL (default: http://localhost:8010/v1)")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Judge model name")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="Number of samples to score (0 = all)")
    parser.add_argument("--max_concurrent", type=int, default=32,
                        help="Max concurrent vLLM requests")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for scores JSONL (default: from env config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--chat_subenvs_dir", type=str, default=None,
                        help="Directory with subenv JSON files (default: from env config)")
    parser.add_argument("--file_prefix", type=str, default="",
                        help="Filename prefix (e.g. 'vuln_', 'normal_') for split files")
    args = parser.parse_args()

    n_samples = args.n_samples if args.n_samples > 0 else None

    run_scoring(
        env_name=args.env_name,
        env=args.env,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        n_samples=n_samples,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
        seed=args.seed,
        chat_subenvs_dir=args.chat_subenvs_dir,
        file_prefix=args.file_prefix,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_scoring_cli()
