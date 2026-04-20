"""
Evaluate generated completions using external LLM API (OpenAI or Anthropic).

This script loads completions with embedded evaluation prompts from a JSONL file,
fills in the {agent_message} placeholder, sends them to an LLM judge API,
and saves the metric scores.

Key features:
- Filters evaluations by step interval (eval_n_steps)
- Subsamples responses per step to get diverse coverage
- Supports both OpenAI and Anthropic as judges

Expected JSONL entry format:
    - 'evaluation_prompts': dict of metric_name -> list of message dicts with {agent_message} placeholder
    - 'agent_message': the agent's response to fill into the placeholder
    - 'step' or 'call_count': step number for filtering

USAGE:
    python -m llm_agent.retroactive.retroactive_eval \
        --completions_file results/therapy_talk/run_name/completions.jsonl \
        --environment therapy_talk \
        --eval_n_steps 10 \
        --subsample_per_step 2 \
        --judge_provider openai \
        --judge_model gpt-4.1-mini-2025-04-14
"""

import json
import argparse
import asyncio
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from llm_agent.retroactive.retroactive_utils import (
    load_jsonl_file,
    filter_and_subsample_by_steps,
    initialize_judge_client,
    initialize_async_judge_client,
    parse_metric_response,
    fill_agent_message_placeholder
)
from llm_agent.utils.utils import get_all_results_dirs_for_experiment


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""
    eval_n_steps: int = 1
    subsample_per_step: Optional[int] = None
    judge_provider: str = 'openai'
    judge_model: str = 'gpt-4.1-mini-2025-04-14'
    judge_temperature: float = 0.0
    metrics: Optional[List[str]] = None
    rate_limit_delay: float = 0.01
    concurrency: int = 50
    api_key: Optional[str] = None
    seed: int = 42


def evaluate_single_sample(
    client,
    entry: Dict[str, Any],
    metrics_to_evaluate: Optional[List[str]] = None,
    rate_limit_delay: float = 0.0
) -> Dict[str, int]:
    """
    Evaluate a single sample for all metrics.

    Args:
        client: LLM judge client (OpenAIChatResponder or AnthropicChatResponder)
        entry: Evaluation entry with 'evaluation_prompts' dict and 'agent_message'
        metrics_to_evaluate: List of metrics to evaluate (None = all)
        rate_limit_delay: Delay between API calls (seconds)

    Returns:
        Dict mapping metric_name to score

    Raises:
        KeyError: If 'agent_message' is missing from entry
        ValueError: If evaluation prompts don't contain {agent_message} placeholder
    """
    evaluation_prompts = entry['evaluation_prompts']

    # Require agent_message to exist
    if 'agent_message' not in entry:
        raise KeyError("Entry missing required 'agent_message' field")
    agent_message = entry['agent_message']

    # Determine which metrics to evaluate
    if metrics_to_evaluate is None:
        metrics_to_evaluate = list(evaluation_prompts.keys())

    scores = {}

    for metric_name in metrics_to_evaluate:
        if metric_name not in evaluation_prompts:
            print(f"Warning: Metric '{metric_name}' not found in evaluation prompts")
            scores[metric_name] = -1
            continue

        prompt = evaluation_prompts[metric_name]

        # Fill in {agent_message} placeholder
        if isinstance(prompt, list):
            prompt = fill_agent_message_placeholder(prompt, agent_message)
        elif isinstance(prompt, str):
            if '{agent_message}' not in prompt:
                raise ValueError(f"No {{agent_message}} placeholder found in prompt for metric '{metric_name}'")
            prompt = prompt.replace('{agent_message}', agent_message)
        else:
            raise TypeError(f"Unexpected prompt type for metric '{metric_name}': {type(prompt)}")

        try:
            # Get response from LLM
            response_text = client.get_response(prompt)

            # Parse the score
            score = parse_metric_response(response_text)
            scores[metric_name] = score

            # Rate limiting
            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)

        except Exception as e:
            print(f"Error evaluating metric '{metric_name}': {e}")
            scores[metric_name] = -1

    return scores


async def evaluate_single_sample_async(
    client,
    semaphore: asyncio.Semaphore,
    entry: Dict[str, Any],
    entry_index: int,
    metrics_to_evaluate: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single sample for all metrics (async version).

    Uses semaphore to control concurrency. Returns a full result dict
    (not just scores) so results can be collected out of order.
    """
    evaluation_prompts = entry['evaluation_prompts']

    if 'agent_message' not in entry:
        raise KeyError("Entry missing required 'agent_message' field")
    agent_message = entry['agent_message']

    if metrics_to_evaluate is None:
        metrics_to_evaluate = list(evaluation_prompts.keys())

    scores = {}

    for metric_name in metrics_to_evaluate:
        if metric_name not in evaluation_prompts:
            scores[metric_name] = -1
            continue

        prompt = evaluation_prompts[metric_name]

        # Fill in {agent_message} placeholder
        if isinstance(prompt, list):
            prompt = fill_agent_message_placeholder(prompt, agent_message)
        elif isinstance(prompt, str):
            if '{agent_message}' not in prompt:
                raise ValueError(f"No {{agent_message}} placeholder found in prompt for metric '{metric_name}'")
            prompt = prompt.replace('{agent_message}', agent_message)
        else:
            raise TypeError(f"Unexpected prompt type for metric '{metric_name}': {type(prompt)}")

        try:
            async with semaphore:
                response_text = await client.get_response(prompt)
            score = parse_metric_response(response_text)
            scores[metric_name] = score
        except Exception as e:
            print(f"Error evaluating metric '{metric_name}': {e}")
            scores[metric_name] = -1

    # Build result dict
    existing_rewards = entry.get('existing_rewards', entry.get('reward'))
    result = {
        'step': entry['step'],
        'agent_message': entry['agent_message'],
        'user_context': entry.get('user_context', ''),
        'existing_rewards': existing_rewards,
        'llm_judge_scores': scores
    }
    if 'is_gameable' in entry:
        result['is_gameable'] = entry['is_gameable']
    if 'is_vulnerable' in entry:
        result['is_vulnerable'] = entry['is_vulnerable']

    return entry_index, result


def evaluate_completions(
    completions_file: str,
    output_file: Optional[str] = None,
    eval_n_steps: int = 1,
    subsample_per_step: Optional[int] = None,
    judge_provider: str = 'openai',
    judge_model: str = 'gpt-4-turbo-preview',
    judge_temperature: float = 0.0,
    metrics: Optional[List[str]] = None,
    rate_limit_delay: float = 0.1,
    api_key: Optional[str] = None,
    seed: int = 42,
    concurrency: int = 50
):
    """
    Main evaluation function. Uses asyncio for concurrent API calls.

    Args:
        completions_file: Path to completions.jsonl with embedded evaluation_prompts
        output_file: Path to save results (default: auto-generated)
        eval_n_steps: Evaluate every N steps
        subsample_per_step: Number of samples to keep per step (None = keep all)
        judge_provider: 'openai' or 'anthropic'
        judge_model: Model slug for judge
        judge_temperature: Temperature for judge
        metrics: List of metrics to evaluate (None = all)
        rate_limit_delay: Unused (kept for backwards compatibility). Use concurrency to control throughput.
        api_key: API key (optional, uses env var if None)
        seed: Random seed for subsampling
        concurrency: Max concurrent API requests (default: 50)
    """
    print("=" * 80)
    print("EVALUATING COMPLETIONS WITH LLM JUDGE")
    print("=" * 80)

    # Load entries directly from completions file
    entries = load_jsonl_file(completions_file)
    print(f"Loaded {len(entries)} evaluation prompt entries from {completions_file}")


    # Filter and subsample by eval_n_steps
    print(f"\nFiltering to every {eval_n_steps} steps with subsampling={subsample_per_step}...")
    filtered_entries = filter_and_subsample_by_steps(
        entries,
        eval_n_steps,
        subsample_per_step,
        seed
    )

    if len(filtered_entries) == 0:
        print("No entries to evaluate after filtering!")
        return

    # Auto-generate output file if not provided
    if output_file is None:
        input_path = Path(completions_file)
        suffix = f"_step{eval_n_steps}"
        if subsample_per_step:
            suffix += f"_subsample{subsample_per_step}"
        output_file = str(input_path.parent / f"llm_judge_scores{suffix}.jsonl")

    print(f"\nJudge provider: {judge_provider}")
    print(f"Judge model: {judge_model}")
    print(f"Concurrency: {concurrency}")
    print(f"Output file: {output_file}")

    # Initialize async judge client
    client = initialize_async_judge_client(
        provider=judge_provider,
        model=judge_model,
        temperature=judge_temperature,
        api_key=api_key
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_all():
        try:
            tasks = [
                evaluate_single_sample_async(
                    client=client,
                    semaphore=semaphore,
                    entry=entry,
                    entry_index=i,
                    metrics_to_evaluate=metrics,
                )
                for i, entry in enumerate(filtered_entries)
            ]

            results = [None] * len(filtered_entries)
            pbar = tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Evaluating"
            )
            for coro in pbar:
                idx, result = await coro
                results[idx] = result

            return results
        finally:
            # Close the underlying httpx client before the event loop shuts down,
            # otherwise httpx schedules aclose() as a background task and hits
            # RuntimeError('Event loop is closed') during teardown.
            await client.aclose()

    # Evaluate all entries concurrently
    print(f"\nEvaluating {len(filtered_entries)} samples...")
    results = asyncio.run(_run_all())

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')

    print(f"\n✅ Saved {len(results)} evaluated samples to: {output_file}")

    # Print summary statistics
    print_evaluation_summary(results)

    return output_file


def print_evaluation_summary(results: List[Dict[str, Any]]):
    """Print summary statistics of evaluation results."""
    if not results:
        return

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Aggregate scores by metric
    all_metrics = set()
    for result in results:
        all_metrics.update(result['llm_judge_scores'].keys())

    for metric in sorted(all_metrics):
        scores = [r['llm_judge_scores'][metric] for r in results if r['llm_judge_scores'][metric] >= 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"{metric}: {avg_score:.2f} (n={len(scores)})")

    # Show example
    print("\n" + "-" * 80)
    print("EXAMPLE OUTPUT ENTRY:")
    print("-" * 80)
    example = results[0]
    print(f"Step: {example['step']}")
    print(f"Agent message: {example['agent_message'][:100]}...")
    print(f"User context: {example.get('user_context', '')[:100]}...")
    print(f"Existing rewards: {example.get('existing_rewards')}")
    print(f"LLM judge scores: {example['llm_judge_scores']}")
    if 'is_gameable' in example:
        print(f"Is gameable: {example['is_gameable']}")


def find_completion_files(directory: str) -> List[Path]:
    """
    Find all .jsonl files containing 'completions' in their name.

    Args:
        directory: Directory path to search

    Returns:
        List of Path objects for matching files
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find all .jsonl files with 'completions' in the name
    completion_files = list(dir_path.glob("*completions*.jsonl"))

    if not completion_files:
        raise ValueError(f"No completion files found in {directory}")

    completion_files.sort()  # Sort for consistent ordering
    print(f"Found {len(completion_files)} completion file(s):")
    for f in completion_files:
        print(f"  - {f.name}")

    return completion_files


def _process_single_file(
    completions_file: str,
    output_file: str,
    config: EvalConfig
) -> tuple[str, bool, str]:
    """
    Process a single completion file. Used for parallel execution.

    Returns:
        Tuple of (filename, success, message)
    """
    try:
        evaluate_completions(
            completions_file=completions_file,
            output_file=output_file,
            eval_n_steps=config.eval_n_steps,
            subsample_per_step=config.subsample_per_step,
            judge_provider=config.judge_provider,
            judge_model=config.judge_model,
            judge_temperature=config.judge_temperature,
            metrics=config.metrics,
            rate_limit_delay=config.rate_limit_delay,
            concurrency=config.concurrency,
            api_key=config.api_key,
            seed=config.seed
        )
        return (completions_file, True, "Success")
    except Exception as e:
        import traceback
        return (completions_file, False, f"{e}\n{traceback.format_exc()}")


def _get_output_filename(completion_file: Path, config: EvalConfig) -> str:
    """Generate output filename based on input file and config."""
    suffix = f"_step{config.eval_n_steps}"
    if config.subsample_per_step:
        suffix += f"_subsample{config.subsample_per_step}"
    return completion_file.stem.replace("completions", "llm_judge_scores") + suffix + ".jsonl"


def process_from_experiment(exp_name: str, config: EvalConfig, run_parallel: bool = False):
    """
    Process all completion files for a given experiment name.

    Args:
        exp_name: Experiment name to search for
        config: Evaluation configuration
        run_parallel: Whether to process files in parallel (max 3 workers)
    """
    exp_results = get_all_results_dirs_for_experiment(exp_name)

    if not exp_results:
        print(f"No results directories found for experiment: {exp_name}")
        exit(1)

    # Collect all directories to process
    all_dirs = []
    for env_name, models in exp_results.items():
        for model_name, seeds in models.items():
            for seed, info in seeds.items():
                all_dirs.append({
                    'env': env_name,
                    'model': model_name,
                    'seed': seed,
                    'results_dir': info['results_dir']
                })

    print(f"Found {len(all_dirs)} results directories for experiment '{exp_name}'")
    for d in all_dirs:
        print(f"  - {d['env']}/{d['model']}/S{d['seed']}: {d['results_dir']}")

    # Collect all files to process, filtering out already-completed ones
    all_tasks = []  # List of (completions_file, output_file, dir_info)
    skipped_tasks = []  # Already completed
    for dir_info in all_dirs:
        results_dir = dir_info['results_dir']
        try:
            completion_files = find_completion_files(results_dir)
        except ValueError as e:
            print(f"⚠️ Skipping {results_dir}: {e}")
            continue

        output_base_dir = Path(results_dir) / "retroactive_evals"

        for completion_file in completion_files:
            output_filename = _get_output_filename(completion_file, config)
            output_file = output_base_dir / output_filename

            # Check if output already exists
            if output_file.exists():
                skipped_tasks.append((str(completion_file), str(output_file), dir_info))
            else:
                all_tasks.append((str(completion_file), str(output_file), dir_info))

    # Report skipped files
    if skipped_tasks:
        print(f"\n⏭️  Skipping {len(skipped_tasks)} file(s) with existing results:")
        for completions_file, output_file, dir_info in skipped_tasks:
            print(f"    - {Path(output_file).name} (in {dir_info['env']}/{dir_info['model']}/S{dir_info['seed']})")

    if not all_tasks:
        print("\n✅ All files already processed! Nothing to do.")
        return

    # Show files to be processed and ask for confirmation
    print("\n" + "=" * 80)
    print(f"📋 FILES TO PROCESS ({len(all_tasks)} total):")
    print("=" * 80)

    # Group by directory for cleaner output
    tasks_by_dir = {}
    for completions_file, output_file, dir_info in all_tasks:
        dir_key = f"{dir_info['env']}/{dir_info['model']}/S{dir_info['seed']}"
        if dir_key not in tasks_by_dir:
            tasks_by_dir[dir_key] = []
        tasks_by_dir[dir_key].append(Path(completions_file).name)

    for dir_key, files in tasks_by_dir.items():
        print(f"\n  {dir_key}:")
        for f in files:
            print(f"    - {f}")

    print("\n" + "-" * 80)
    print(f"Total: {len(all_tasks)} file(s) to process, {len(skipped_tasks)} skipped (already done)")
    print("-" * 80)

    # Ask for confirmation
    response = input("\nProceed with retroactive evaluation? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    # Create output directories (only after confirmation)
    for dir_info in all_dirs:
        output_base_dir = Path(dir_info['results_dir']) / "retroactive_evals"
        output_base_dir.mkdir(exist_ok=True)

    print(f"\n🚀 Starting evaluation of {len(all_tasks)} files...")

    if run_parallel:
        _process_files_parallel(all_tasks, config, max_workers=6)
    else:
        _process_files_sequential(all_tasks, config)

    print("\n" + "=" * 80)
    print(f"✅ EXPERIMENT PROCESSING COMPLETE - Processed {len(all_tasks)} files")
    print("=" * 80)


def process_from_directory(completions_dir: str, config: EvalConfig, run_parallel: bool = False):
    """
    Process all completion files in a directory.

    Args:
        completions_dir: Directory containing completion files
        config: Evaluation configuration
        run_parallel: Whether to process files in parallel (max 3 workers)
    """
    completion_files = find_completion_files(completions_dir)

    output_base_dir = Path(f"./{completions_dir}/retroactive_evals")
    output_base_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_base_dir}")
    print(f"Found {len(completion_files)} file(s) to process\n")

    # Build task list
    all_tasks = []
    for completion_file in completion_files:
        output_filename = _get_output_filename(completion_file, config)
        output_file = output_base_dir / output_filename
        all_tasks.append((str(completion_file), str(output_file), None))

    if run_parallel:
        _process_files_parallel(all_tasks, config, max_workers=3)
    else:
        _process_files_sequential(all_tasks, config)

    print("\n" + "=" * 80)
    print(f"✅ BATCH PROCESSING COMPLETE - Processed {len(completion_files)} file(s)")
    print(f"Results saved to: {output_base_dir}")
    print("=" * 80)


def _process_files_sequential(tasks: List[tuple], config: EvalConfig):
    """Process files sequentially."""
    for i, (completions_file, output_file, dir_info) in enumerate(tasks, 1):
        print("\n" + "=" * 80)
        print(f"Processing file {i}/{len(tasks)}: {Path(completions_file).name}")
        if dir_info:
            print(f"  Environment: {dir_info['env']}, Model: {dir_info['model']}, Seed: {dir_info['seed']}")
        print("=" * 80)

        filename, success, message = _process_single_file(completions_file, output_file, config)
        if success:
            print(f"✅ Successfully processed: {Path(filename).name}")
        else:
            print(f"❌ Error processing {Path(filename).name}: {message}")


def _process_files_parallel(tasks: List[tuple], config: EvalConfig, max_workers: int = 3):
    """Process files in parallel with a maximum number of workers."""
    print(f"\n🚀 Running in parallel mode with {max_workers} workers")

    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_process_single_file, completions_file, output_file, config): (completions_file, dir_info)
            for completions_file, output_file, dir_info in tasks
        }

        # Process results as they complete
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing files"):
            completions_file, dir_info = future_to_task[future]
            try:
                filename, success, message = future.result()
                if success:
                    print(f"\n✅ Successfully processed: {Path(filename).name}")
                    successful += 1
                else:
                    print(f"\n❌ Error processing {Path(filename).name}: {message}")
                    failed += 1
            except Exception as e:
                print(f"\n❌ Exception processing {Path(completions_file).name}: {e}")
                failed += 1

    print(f"\n📊 Parallel processing complete: {successful} succeeded, {failed} failed")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate completions using external LLM judge API"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--completions_file",
        type=str,
        help="Path to completions.jsonl with embedded evaluation_prompts"
    )
    input_group.add_argument(
        "--completions_dir",
        type=str,
        help="Directory containing multiple completion files (will process all *completions*.jsonl files)"
    )
    input_group.add_argument(
        "--from_exp_name",
        type=str,
        help="Experiment name to find all results directories for (processes all *completions*.jsonl files in each)"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results (default: auto-generated)"
    )
    parser.add_argument(
        "--eval_n_steps",
        type=int,
        default=1,
        help="Evaluate every N steps (default: 1 = evaluate all)"
    )
    parser.add_argument(
        "--subsample_per_step",
        type=int,
        default=None,
        help="Number of samples to keep per step (default: None = keep all). "
             "For GRPO with 10 responses per input, use 2-3 for better coverage."
    )

    # Judge configuration
    parser.add_argument(
        "--judge_provider",
        type=str,
        default='openai',
        choices=['openai', 'anthropic'],
        help="LLM provider for judge"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default='gpt-4.1-mini-2025-04-14',
        help="Model slug for judge (e.g., gpt-4-turbo-preview, claude-3-5-sonnet-20241022)"
    )
    parser.add_argument(
        "--judge_temperature",
        type=float,
        default=0.0,
        help="Temperature for judge"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to evaluate (default: all)"
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=0.01,
        help="Delay between API calls in seconds"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (optional, uses environment variable if not provided)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Max concurrent async API requests per file (default: 50)"
    )
    parser.add_argument(
        "--run_parallel",
        action="store_true",
        help="Run evaluations in parallel (max 3 files at a time)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build config from args
    config = EvalConfig(
        eval_n_steps=args.eval_n_steps,
        subsample_per_step=args.subsample_per_step,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        metrics=args.metrics,
        rate_limit_delay=args.rate_limit_delay,
        concurrency=args.concurrency,
        api_key=args.api_key,
        seed=args.seed
    )

    # Determine which mode to use
    if args.from_exp_name:
        process_from_experiment(args.from_exp_name, config, run_parallel=args.run_parallel)
    elif args.completions_dir:
        process_from_directory(args.completions_dir, config, run_parallel=args.run_parallel)
    else:
        # Single file mode
        evaluate_completions(
            completions_file=args.completions_file,
            output_file=args.output_file,
            eval_n_steps=config.eval_n_steps,
            subsample_per_step=config.subsample_per_step,
            judge_provider=config.judge_provider,
            judge_model=config.judge_model,
            judge_temperature=config.judge_temperature,
            metrics=config.metrics,
            rate_limit_delay=config.rate_limit_delay,
            concurrency=config.concurrency,
            api_key=config.api_key,
            seed=config.seed
        )


if __name__ == "__main__":
    main()
