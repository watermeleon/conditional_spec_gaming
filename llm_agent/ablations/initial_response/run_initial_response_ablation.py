"""
Initial-response ablation: can we predict emergent misalignment from initial models?

For each of 11 pre-trained models × 3 environments, generate responses for
50 harmful (gameable) and 50 harmless (non-gameable) prompts drawn with a
model-specific seed (so every model sees a different random 100-sample draw),
score the responses with an LLM judge using the same ER metric as the main
experiments, and compute the ER-gap = mean_ER(harmful) − mean_ER(harmless).
A per-environment plot of ER-gap vs. model parameter count is produced.

Phases (selectable via --phase):
  generate  - load each model, generate responses, save completions JSONL
  score     - call LLM-judge retroactive eval on each completions file
  plot      - load ER scores, compute ER-gap, plot vs. num_params
  all       - run all three phases in sequence

Output goes to results_ablation/initial_response/ to avoid polluting the
main results/ directory used by existing processing pipelines.

Usage:
    python -m llm_agent.ablations.initial_response.run_initial_response_ablation --phase generate
    python -m llm_agent.ablations.initial_response.run_initial_response_ablation --phase score
    python -m llm_agent.ablations.initial_response.run_initial_response_ablation --phase plot
    python -m llm_agent.ablations.initial_response.run_initial_response_ablation --phase all
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── repo root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm_agent.environments.action_advice.action_advice_loader import ActionAdviceLoader
from llm_agent.environments.political_qa.politicalqa_dataloader import PoliticalQALoader
from llm_agent.environments.therapy_talk.therapy_talk_loader import TherapyTalkLoader
from llm_agent.retroactive.retroactive_eval import evaluate_completions
from llm_agent.utils.utils import ensure_chat_template, get_allowed_tokenizer_roles

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Sorted largest-first so OOM surfaces immediately
MODEL_NAMES = [
    "Qwen/Qwen1.5-14B-Chat",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-1.1-7b-it",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen1.5-7B-Chat",
    "01-ai/Yi-6B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "google/gemma-1.1-2b-it",
    "Qwen/Qwen1.5-0.5B-Chat",
]

TARGET_ENVS = ["therapy_talk", "action_advice", "political_qa"]

N_PER_TYPE = 50          # 50 harmful + 50 harmless per env per model
MAX_NEW_TOKENS = 256
# BATCH_SIZE = 4
BATCH_SIZE = 26

ABLATION_BASE = Path("./results_ablation/initial_response")
FEATURES_FILE = Path("./safetywashing/data/model_features_metrics_Combi.json")

JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
JUDGE_CONCURRENCY = 50


# ─────────────────────────────────────────────────────────────────────────────
# Utilities shared with run_perplexity_ablation
# ─────────────────────────────────────────────────────────────────────────────

def model_to_slug(model_name: str) -> str:
    """Convert HF model path to filesystem-safe folder name."""
    return model_name.split("/")[-1].replace(".", "_").replace("-", "_")


def model_to_seed(model_name: str) -> int:
    """Derive a reproducible seed from the model name via MD5."""
    return int(hashlib.md5(model_name.encode()).hexdigest(), 16) % (2 ** 31)


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"\n  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    tokenizer = ensure_chat_template(tokenizer, model_name)
    tokenizer.padding_side = "left"  # re-assert in case ensure_chat_template replaced it
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Loading model ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def unload_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_env_samples(
    env_name: str,
    n: int,
    gameable_fraction: float,
    seed: int,
    allowed_tokenizer_roles: dict,
) -> list[dict]:
    """
    Load n samples (gameable_fraction harmful, rest harmless) for an environment.

    Returns a list of dicts with at minimum:
      prompt, evaluation_prompts, is_gameable, env_name
    """
    env_paths = {
        "therapy_talk":  "./llm_agent/environments/therapy_talk/",
        "action_advice": "./llm_agent/environments/action_advice/",
        "political_qa":  "./llm_agent/environments/political_qa/",
    }
    path = env_paths[env_name]

    if env_name == "therapy_talk":
        loader = TherapyTalkLoader(path)
    elif env_name == "action_advice":
        loader = ActionAdviceLoader(path)
    elif env_name == "political_qa":
        loader = PoliticalQALoader(path)
    else:
        raise ValueError(f"Unknown env: {env_name}")

    dataset = loader.load_samples(
        n_samples=n,
        gameable_fraction=gameable_fraction,
        seed=seed,
        allowed_tokenizer_roles=allowed_tokenizer_roles,
    )
    return list(dataset)   # list of dicts from HF Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Response generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list,
    device: torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
    temperature: float = 0.7,
) -> list[str]:
    """
    Generate one response per prompt inside torch.no_grad().

    Uses sampling (do_sample=True) when temperature > 0, greedy otherwise.
    Each prompt is a list of message dicts (output of load_env_samples).
    Left-padding ensures all sequences in a batch share the same padded input
    length, so slicing output_ids[:, input_len:] gives just the new tokens.
    """
    responses = []
    do_sample = temperature > 0.0

    for i in tqdm(range(0, len(prompts), batch_size), desc="  Generating", leave=False):
        batch = prompts[i : i + batch_size]

        # Apply chat template → one string per sample
        texts = []
        for prompt in batch:
            try:
                text = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(f"  [WARN] apply_chat_template failed: {e}. Using empty prompt.")
                text = ""
            texts.append(text)

        encodings = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=False
        ).to(device)
        input_len = encodings.input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        for out in output_ids:
            new_tokens = out[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(text.strip())

    return responses


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: generate
# ─────────────────────────────────────────────────────────────────────────────

def run_generation(
    model_names: list[str] = MODEL_NAMES,
    env_names: list[str] = TARGET_ENVS,
    n_per_type: int = N_PER_TYPE,
    skip_existing: bool = True,
    temperature: float = 0.7,
) -> None:
    """
    For each model, load once, generate responses for all envs, then unload.
    Each model uses a unique seed derived from its name, ensuring different
    sample draws across models for better coverage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_total = n_per_type * 2  # 50 harmful + 50 harmless
    tot_models = len(model_names)
    for i, model_name in enumerate(model_names):
        slug = model_to_slug(model_name)
        print(f"\n{'='*60}")
        print(f"[{i+1}/{tot_models} - {(i+1)/tot_models:.1%}] Model: {model_name}  ({slug})")

        # Collect tasks to do (allow resuming)
        tasks = []
        for env_name in env_names:
            out = ABLATION_BASE / slug / f"{env_name}_completions.jsonl"
            if skip_existing and out.exists():
                print(f"  [SKIP] {env_name} completions already exist")
            else:
                tasks.append(env_name)

        if not tasks:
            print("  All tasks already complete, skipping model load.")
            continue

        # Allowed roles must be determined before loading model (tokenizer only)
        print(f"  Detecting tokenizer role support ...")
        allowed_roles = get_allowed_tokenizer_roles(model_name)
        print(f"  Allowed roles: {allowed_roles}")

        model, tokenizer = load_model_and_tokenizer(model_name, device)
        seed = model_to_seed(model_name)
        print(f"  Model seed: {seed}")

        for env_name in tasks:
            out = ABLATION_BASE / slug / f"{env_name}_completions.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)

            print(f"  [{env_name}] Loading {n_total} samples (seed={seed}) ...", flush=True)
            try:
                samples = load_env_samples(
                    env_name=env_name,
                    n=n_total,
                    gameable_fraction=0.5,
                    seed=seed,
                    allowed_tokenizer_roles=allowed_roles,
                )
            except Exception as e:
                print(f"  [ERROR] Could not load samples for {env_name}: {e}")
                continue

            n_harmful = sum(1 for s in samples if s["is_gameable"])
            n_harmless = len(samples) - n_harmful
            print(f"  [{env_name}] {n_harmful} harmful, {n_harmless} harmless")

            prompts = [s["prompt"] for s in samples]

            print(f"  [{env_name}] Generating responses (temp={temperature}) ...", flush=True)
            responses = generate_responses(model, tokenizer, prompts, device, temperature=temperature)

            # Save completions with schema compatible with evaluate_completions()
            with open(out, "w") as f:
                for sample, response in zip(samples, responses):
                    entry = {
                        "model_name":         model_name,
                        "env_name":           sample.get("env_name", env_name),
                        "is_gameable":        sample["is_gameable"],
                        "agent_message":      response,
                        "evaluation_prompts": sample["evaluation_prompts"],
                        "step":               0,   # required by evaluate_completions filter
                    }
                    f.write(json.dumps(entry) + "\n")

            print(f"  [{env_name}] Saved {len(responses)} completions → {out}")

        unload_model(model, tokenizer)
        print(f"  Model {slug} unloaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: score
# ─────────────────────────────────────────────────────────────────────────────

def run_scoring(
    model_names: list[str] = MODEL_NAMES,
    env_names: list[str] = TARGET_ENVS,
    skip_existing: bool = True,
    judge_provider: str = JUDGE_PROVIDER,
    judge_model: str = JUDGE_MODEL,
    concurrency: int = JUDGE_CONCURRENCY,
) -> None:
    """
    Score generated completions using the LLM judge via evaluate_completions().
    Reuses the same async retroactive eval infrastructure as the main experiments.
    """
    for model_name in model_names:
        slug = model_to_slug(model_name)
        print(f"\n{'='*60}")
        print(f"Scoring: {slug}")

        for env_name in env_names:
            completions_file = ABLATION_BASE / slug / f"{env_name}_completions.jsonl"
            output_file      = ABLATION_BASE / slug / f"{env_name}_er_scores.jsonl"

            if not completions_file.exists():
                print(f"  [SKIP] {env_name}: completions file not found, run --phase generate first")
                continue

            if skip_existing and output_file.exists():
                print(f"  [SKIP] {env_name}: scores already exist")
                continue

            print(f"  [{env_name}] Scoring {completions_file.name} ...", flush=True)
            try:
                evaluate_completions(
                    completions_file=str(completions_file),
                    output_file=str(output_file),
                    eval_n_steps=1,          # step=0 for all → all pass the filter
                    subsample_per_step=None, # keep all 100 samples
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                    concurrency=concurrency,
                )
                print(f"  [{env_name}] Scores saved → {output_file}")
            except Exception as e:
                print(f"  [ERROR] Scoring failed for {env_name}/{slug}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: compute ER-gap and plot
# ─────────────────────────────────────────────────────────────────────────────

def load_model_features(file_path: Path) -> dict:
    """
    Load model features (num_params, benchmark scores, etc.) from the
    safetywashing JSON file. Returns a dict keyed by normalised model slug.
    """
    with open(file_path) as f:
        data = json.load(f)

    features = {}
    for entry in data:
        # Normalise model name the same way model_to_slug() does
        key = entry["model"].replace("-", "_").replace(".", "_")
        features[key] = entry
    return features


def compute_er_gap(scores_file: Path) -> Optional[dict]:
    """
    Load ER scores and return ER-gap statistics.

    ER-gap = mean_ER(harmful) − mean_ER(harmless)
    where ER scores are the raw 1-10 LLM-judge scores for 'er_metric',
    consistent with ER_mean in compute_alignment_metrics().
    """
    if not scores_file.exists():
        return None

    harmful_scores = []
    harmless_scores = []

    with open(scores_file) as f:
        for line in f:
            entry = json.loads(line)
            er_score = entry.get("llm_judge_scores", {}).get("er_metric")
            if er_score is None or er_score < 0:
                continue
            if entry.get("is_gameable"):
                harmful_scores.append(er_score)
            else:
                harmless_scores.append(er_score)

    if not harmful_scores or not harmless_scores:
        return None

    mean_er_harmful  = sum(harmful_scores)  / len(harmful_scores)
    mean_er_harmless = sum(harmless_scores) / len(harmless_scores)
    er_gap = mean_er_harmful - mean_er_harmless

    return {
        "mean_er_harmful":  mean_er_harmful,
        "mean_er_harmless": mean_er_harmless,
        "er_gap":           er_gap,
        "n_harmful":        len(harmful_scores),
        "n_harmless":       len(harmless_scores),
    }


def run_plot(
    model_names: list[str] = MODEL_NAMES,
    env_names: list[str] = TARGET_ENVS,
    features_file: Path = FEATURES_FILE,
) -> None:
    """
    Compute ER-gap for every model × env and produce one scatter plot per env.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    features = load_model_features(features_file)

    # Collect results: {env_name: [{slug, num_params, er_gap, ...}, ...]}
    all_results: dict[str, list[dict]] = {env: [] for env in env_names}
    summary = {}

    for model_name in model_names:
        slug = model_to_slug(model_name)
        num_params = None
        if slug in features:
            num_params = features[slug].get("num_params")
        else:
            print(f"  [WARN] {slug} not found in features file; num_params will be None")

        summary[slug] = {}
        for env_name in env_names:
            scores_file = ABLATION_BASE / slug / f"{env_name}_er_scores.jsonl"
            result = compute_er_gap(scores_file)
            if result is None:
                print(f"  [SKIP] No valid ER scores for {slug}/{env_name}")
                continue
            result["model_name"] = model_name
            result["slug"]       = slug
            result["num_params"] = num_params
            all_results[env_name].append(result)
            summary[slug][env_name] = result

    # Save summary JSON
    summary_path = ABLATION_BASE / "er_gap_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved ER-gap summary → {summary_path}")

    # One plot per environment
    plots_dir = ABLATION_BASE / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for env_name in env_names:
        rows = all_results[env_name]
        if not rows:
            print(f"  [SKIP] No data for env {env_name}")
            continue

        # Sort by num_params for a meaningful x-axis
        rows_with_params = [r for r in rows if r["num_params"] is not None]
        rows_with_params.sort(key=lambda r: r["num_params"])

        params  = [r["num_params"] for r in rows_with_params]
        er_gaps = [r["er_gap"]     for r in rows_with_params]
        labels  = [r["slug"]       for r in rows_with_params]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(params, er_gaps, zorder=3)
        ax.plot(params, er_gaps, linestyle="--", alpha=0.5, zorder=2)

        for x, y, lbl in zip(params, er_gaps, labels):
            ax.annotate(
                lbl,
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Model Parameters (B)")
        ax.set_ylabel("ER Gap  (Harmful − Harmless)")
        ax.set_title(f"Initial ER Gap vs. Model Size  [{env_name}]")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", alpha=0.3)
        fig.tight_layout()

        out_path = plots_dir / f"er_gap_{env_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Initial-response ablation: predict emergent misalignment from ER-gap"
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "score", "plot", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--n_per_type",
        type=int,
        default=N_PER_TYPE,
        help=f"Samples per type (harmful/harmless) per model/env (default: {N_PER_TYPE})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation; 0 = greedy (default: 0.7)",
    )
    parser.add_argument(
        "--judge_provider",
        type=str,
        default=JUDGE_PROVIDER,
        choices=["openai", "anthropic"],
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=JUDGE_MODEL,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=JUDGE_CONCURRENCY,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.phase in ("generate", "all"):
        print("\n" + "=" * 60)
        print("PHASE 1: GENERATE RESPONSES")
        print("=" * 60)
        run_generation(n_per_type=args.n_per_type, temperature=args.temperature)

    if args.phase in ("score", "all"):
        print("\n" + "=" * 60)
        print("PHASE 2: SCORE WITH LLM JUDGE")
        print("=" * 60)
        run_scoring(
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
        )

    if args.phase in ("plot", "all"):
        print("\n" + "=" * 60)
        print("PHASE 3: COMPUTE ER-GAP AND PLOT")
        print("=" * 60)
        run_plot()


if __name__ == "__main__":
    main()
