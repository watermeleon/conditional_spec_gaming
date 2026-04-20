"""
Metric Alignment Ablation — validates cross-model judge score alignment.

For each (env, training_model) combination, randomly samples num_samples records
from results_ablation/completions_restruct (which already have gpt-4.1-mini ER/ACC
scores and training-time reward scores from Llama-3.1-8B-Instruct). A new judge model
then re-scores the same completions using the stored reward, ER, and ACC prompts —
either via logprobs-based weighted average (local HF models) or via text generation
+ parsing (API models such as Claude Sonnet).

The two sets of scores can be compared with the functions in alignment_utils.py.

Phases
------
  sample  —  Select num_samples records per (env, training_model) combo from
             completions_restruct, saving the minimal fields needed for scoring
             and later analysis.
  score   —  Load each judge model, compute reward / er_metric / acc_metric scores
             for every selected sample, save results.

Usage
-----
  # Sampling (fast, CPU-only):
  python -m llm_agent.ablations.metric_alignment.run_metric_alignment_ablation --phase sample

  # Scoring with local HF judge(s) (GPU required):
  python -m llm_agent.ablations.metric_alignment.run_metric_alignment_ablation --phase score

  # Scoring with API judge(s) (Anthropic API key required):
  python -m llm_agent.ablations.metric_alignment.run_metric_alignment_ablation \\
      --phase score --api_judge_models claude-sonnet-4-6

  # Both local and API judges in one run:
  python -m llm_agent.ablations.metric_alignment.run_metric_alignment_ablation \\
      --phase score \\
      --judge_models Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --api_judge_models claude-sonnet-4-6

  # Both phases in sequence:
  python -m llm_agent.ablations.metric_alignment.run_metric_alignment_ablation --phase all

Output
------
  results_ablation/metric_alignment/
    {env_name}/{training_model_slug}/
      selected_samples.jsonl           # sampled records with original scores + prompt fields
      {judge_model_slug}_scores.jsonl  # new judge scores (reward, er_metric, acc_metric)
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

# ── repo root on path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm_agent.ablations.metric_alignment.score_local import (
    compute_logprob_score_batch,
    load_model_and_tokenizer,
    unload_model,
)
from llm_agent.ablations.metric_alignment.score_api import (
    API_JUDGE_MODELS,
    score_records_api,
)

from tqdm import tqdm
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Local (HuggingFace) judge models
JUDGE_MODELS = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]

# Metrics and the field in each record that stores their prompt messages
METRICS = {
    "reward":     "judge_context_messages",   # [system, user] with {agent_message}
    "er_metric":  "er_metric",                # evaluation_prompts["er_metric"]
    "acc_metric": "acc_metric",               # evaluation_prompts["acc_metric"]
}

# Fields saved to selected_samples.jsonl (keeps files small)
SAMPLE_FIELDS = [
    "agent_message",
    "is_gameable",
    "reward",
    "retro_reward",
    "er_metric",
    "acc_metric",
    "judge_context_messages",
    "evaluation_prompts",
    "_model_name",
    "_seed",
    "_env_name",
    "step",
    "env_name",
]

COMPLETIONS_BASE = Path("./results_ablation/completions_restruct")
TARGET_ENVS = ["therapy_talk", "action_advice", "political_qa"]

ALIGNMENT_BASE = Path("./results_ablation/metric_alignment")
# ALIGNMENT_BASE = Path("./results_ablation/v1_5samp_metric_al  ignment")


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def model_to_slug(model_name: str) -> str:
    """Convert HF model path / API slug to filesystem-safe folder name."""
    return model_name.split("/")[-1].replace(".", "_").replace("-", "_")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _fill_messages(messages: list[dict], agent_message: str) -> list[dict]:
    """Replace {agent_message} placeholder in all message contents."""
    return [
        {**m, "content": m["content"].replace("{agent_message}", agent_message)}
        for m in messages
    ]


def _get_metric_messages(rec: dict, metric: str) -> Optional[list[dict]]:
    """
    Extract and fill the message template for a metric from a record.

    - 'reward'     → rec['judge_context_messages']
    - 'er_metric'  → rec['evaluation_prompts']['er_metric']
    - 'acc_metric' → rec['evaluation_prompts']['acc_metric']
    """
    agent_message = rec.get("agent_message", "")

    if metric == "reward":
        messages = rec.get("judge_context_messages")
    else:
        eval_prompts = rec.get("evaluation_prompts") or {}
        messages = eval_prompts.get(metric)

    if not messages:
        return None

    return _fill_messages(messages, agent_message)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Sample
# ─────────────────────────────────────────────────────────────────────────────

def _sample_across_seeds(
    env_name: str,
    training_model_slug: str,
    num_samples: int,
    seed: int,
) -> list[dict]:
    """
    Load all S*_combined.jsonl files for the (env, training_model) combo and
    randomly draw num_samples records across all seeds.
    Only records that have both er_metric and acc_metric are eligible.
    """
    model_dir = COMPLETIONS_BASE / env_name / training_model_slug
    if not model_dir.exists():
        return []

    all_records = []
    for f in sorted(model_dir.glob("S*_combined.jsonl")):
        for rec in load_jsonl(f):
            if rec.get("er_metric") is not None and rec.get("acc_metric") is not None:
                all_records.append(rec)

    if not all_records:
        return []

    rng = random.Random(seed)
    if len(all_records) <= num_samples:
        return all_records
    return rng.sample(all_records, num_samples)


def _filter_fields(rec: dict) -> dict:
    """Keep only SAMPLE_FIELDS to reduce file size."""
    return {k: rec[k] for k in SAMPLE_FIELDS if k in rec}


def run_sample_phase(num_samples: int = 100, seed: int = 42) -> None:
    """
    For every (env, training_model) combo found in completions_restruct,
    draw num_samples records and save them to ALIGNMENT_BASE.
    """
    print(f"\n=== Phase 1: Sampling ({num_samples} per combo, seed={seed}) ===\n")

    for env_name in TARGET_ENVS:
        env_dir = COMPLETIONS_BASE / env_name
        if not env_dir.exists():
            print(f"  [WARN] Not found: {env_dir}")
            continue

        for model_dir in sorted(env_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            training_model_slug = model_dir.name

            out_dir = ALIGNMENT_BASE / env_name / training_model_slug
            out_path = out_dir / "selected_samples.jsonl"

            if out_path.exists():
                n = sum(1 for _ in open(out_path))
                print(f"  [SKIP] {env_name}/{training_model_slug}  ({n} samples already)")
                continue

            records = _sample_across_seeds(env_name, training_model_slug, num_samples, seed)
            if not records:
                print(f"  [WARN] No valid records for {env_name}/{training_model_slug}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for rec in records:
                    f.write(json.dumps(_filter_fields(rec)) + "\n")

            print(f"  Saved {len(records):4d} samples → {out_path}")

    print("\nSampling complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2a — Score (local HuggingFace models)
# ─────────────────────────────────────────────────────────────────────────────

def run_score_phase(
    judge_models: list[str] = JUDGE_MODELS,
    skip_existing: bool = True,
    batch_size: int = 26,
) -> None:
    """
    For each local judge model, compute reward / er_metric / acc_metric scores
    for every selected_samples.jsonl found under ALIGNMENT_BASE.
    Records are processed in batches of batch_size for GPU efficiency.
    """
    if not judge_models:
        return

    print(f"\n=== Phase 2a: Local scoring with {len(judge_models)} model(s), batch_size={batch_size} ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for judge_name in judge_models:
        judge_slug = model_to_slug(judge_name)
        print(f"\n{'='*60}")
        print(f"Judge: {judge_name}  ({judge_slug})")

        tasks = []
        for samples_path in sorted(ALIGNMENT_BASE.rglob("selected_samples.jsonl")):
            training_model_slug = samples_path.parent.name
            env_name = samples_path.parent.parent.name
            out_path = samples_path.parent / f"{judge_slug}_scores.jsonl"

            if skip_existing and out_path.exists():
                print(f"  [SKIP] {env_name}/{training_model_slug}")
            else:
                tasks.append((env_name, training_model_slug, samples_path, out_path))

        if not tasks:
            print("  All outputs already exist, skipping model load.")
            continue

        model, tokenizer = load_model_and_tokenizer(judge_name, device)
        total_tasks = len(tasks)
        for i, (env_name, training_model_slug, samples_path, out_path) in enumerate(tasks):
            records = load_jsonl(samples_path)
            n = len(records)
            print(f"\n ({i+1}/{total_tasks} - {(i+1)/total_tasks:.1%}) [{env_name}/{training_model_slug}] {n} samples", flush=True)

            results: list[dict] = [{"idx": idx} for idx in range(n)]

            for metric in METRICS:
                all_messages = [_get_metric_messages(rec, metric) for rec in records]
                scores: list[Optional[float]] = [None] * n

                for start in tqdm(range(0, n, batch_size), desc=f"  {env_name}/{training_model_slug} [{metric}]", leave=False):
                    end = min(start + batch_size, n)
                    batch_msgs = all_messages[start:end]

                    valid = [(local_i, m) for local_i, m in enumerate(batch_msgs) if m is not None]
                    if not valid:
                        continue
                    local_indices, valid_msgs = zip(*valid)
                    batch_scores = compute_logprob_score_batch(model, tokenizer, list(valid_msgs), device)
                    for local_i, score in zip(local_indices, batch_scores):
                        scores[start + local_i] = score

                for idx, score in enumerate(scores):
                    results[idx][metric] = score

            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

            n_ok = sum(1 for r in results if r.get("reward") is not None)
            print(f"    → saved {len(results)} records ({n_ok} with valid reward score)")

        unload_model(model, tokenizer)
        print(f"\n  Judge {judge_slug} unloaded.")

    print("\nLocal scoring complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2b — Score (API models, e.g. Claude Sonnet)
# ─────────────────────────────────────────────────────────────────────────────

def run_score_phase_api(
    judge_models: list[str] = API_JUDGE_MODELS,
    skip_existing: bool = True,
    max_concurrent: int = 10,
) -> None:
    """
    For each API judge model, compute reward / er_metric / acc_metric scores
    for every selected_samples.jsonl found under ALIGNMENT_BASE.

    Scores are obtained via async text generation + parsing (no GPU required).
    Up to max_concurrent requests are in-flight at once.
    Requires ANTHROPIC_API_KEY to be set in the environment.
    """
    if not judge_models:
        return

    print(f"\n=== Phase 2b: API scoring with {len(judge_models)} model(s), max_concurrent={max_concurrent} ===\n")

    for judge_name in judge_models:
        judge_slug = model_to_slug(judge_name)
        print(f"\n{'='*60}")
        print(f"API Judge: {judge_name}  ({judge_slug})")

        tasks = []
        for samples_path in sorted(ALIGNMENT_BASE.rglob("selected_samples.jsonl")):
            training_model_slug = samples_path.parent.name
            env_name = samples_path.parent.parent.name
            out_path = samples_path.parent / f"{judge_slug}_scores.jsonl"

            if skip_existing and out_path.exists():
                print(f"  [SKIP] {env_name}/{training_model_slug}")
            else:
                tasks.append((env_name, training_model_slug, samples_path, out_path))

        if not tasks:
            print("  All outputs already exist, skipping.")
            continue

        total_tasks = len(tasks)
        for i, (env_name, training_model_slug, samples_path, out_path) in enumerate(tasks):
            records = load_jsonl(samples_path)
            n = len(records)
            print(f"\n ({i+1}/{total_tasks} - {(i+1)/total_tasks:.1%}) [{env_name}/{training_model_slug}] {n} samples", flush=True)

            results = score_records_api(
                records, _get_metric_messages, METRICS,
                model_slug=judge_name,
                max_concurrent=max_concurrent,
            )

            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

            n_ok = sum(1 for r in results if r.get("reward") is not None)
            print(f"    → saved {len(results)} records ({n_ok} with valid reward score)")

    print("\nAPI scoring complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Metric alignment ablation — sample and/or score."
    )
    parser.add_argument(
        "--phase",
        choices=["sample", "score", "all"],
        default="all",
        help="Which phase(s) to run (default: all).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples per (env, training_model) combo (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--judge_models",
        nargs="+",
        default=JUDGE_MODELS,
        metavar="MODEL",
        help="HuggingFace model IDs for local judge(s).",
    )
    parser.add_argument(
        "--api_judge_models",
        nargs="+",
        default=API_JUDGE_MODELS,
        metavar="MODEL",
        help="API model slugs for API judge(s) (e.g. claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for local model forward passes (default: 16).",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Max concurrent API requests during API scoring (default: 10).",
    )
    parser.add_argument(
        "--no_api",
        action="store_true",
        help="Skip API scoring even if --api_judge_models is set.",
    )
    parser.add_argument(
        "--no_local",
        action="store_true",
        help="Skip local HF model scoring.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.phase in ("sample", "all"):
        run_sample_phase(num_samples=args.num_samples, seed=args.seed)

    if args.phase in ("score", "all"):
        if not args.no_local:
            run_score_phase(judge_models=args.judge_models, batch_size=args.batch_size)
        if not args.no_api:
            run_score_phase_api(judge_models=args.api_judge_models, max_concurrent=args.max_concurrent)
