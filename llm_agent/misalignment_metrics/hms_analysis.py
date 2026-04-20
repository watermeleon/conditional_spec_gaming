"""
HMS (Harmful Misalignment Score) Analysis

This script loads LLM judge evaluation results and computes custom HMS metrics
(like ACC, ER, etc.) using environment-specific reward functions.

Key features:
- Loads judge scores from retroactive_eval.py output
- Gets custom HMS metric function from environment
- Computes metrics per step and in aggregated windows
- Supports grouping by vulnerability properties (is_gameable, etc.)
- Generates plots and summary statistics

USAGE:
    python -m llm_agent.retroactive.hms_analysis \
        --judge_results_file results/therapy_talk/run_name/llm_judge_scores_step10_subsample2.jsonl \
        --environment therapy_talk \
        --aggregate_window 10 \
        --output_dir results/therapy_talk/run_name/hms_analysis
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import sys

# Import create_dataset for getting custom metric functions
sys.path.append(str(Path(__file__).parent.parent))
from environments import create_dataset


def get_custom_metric_function(environment: str):
    """
    Get custom metric computation function for the environment.

    Args:
        environment: Environment name (e.g., 'therapy_talk', 'action_advice', 'political_qa')

    Returns:
        Custom metric function or None if not available
    """
    _, _, custom_metric_fn = create_dataset(
        dataset_name=environment,
        n_samples=20,
        allowed_tokenizer_roles={"system": False,
                                 "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                                 },
        seed=42
    )
    return custom_metric_fn


def load_judge_results(jsonl_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Load LLM judge evaluation results from JSONL file.
    
    Expected format per line:
    {
        'step': int,
        'agent_message': str,
        'llm_judge_scores': dict,
        'is_gameable': bool (optional),
        'is_vulnerable': bool (optional),
        ...
    }
    
    Args:
        jsonl_path: Path to judge results JSONL
        
    Returns:
        List of result entries
    """
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    if verbose:
        print(f"Loaded {len(results)} judge result entries from {jsonl_path}")
    return results


def compute_hms_metrics_per_step(
    results: List[Dict[str, Any]],
    hms_metric_fn,
    verbose: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Compute HMS metrics for each step individually.
    
    Groups results by step and computes HMS metrics (ACC, ER, etc.) for each step.
    
    Args:
        results: List of evaluation result dicts with 'step', 'llm_judge_scores', etc.
        hms_metric_fn: Function to compute HMS metrics from evaluation scores
        verbose: Whether to print progress
        
    Returns:
        Dict mapping step -> HMS metrics dict
    """
    # Group results by step
    results_by_step = defaultdict(list)
    for r in results:
        results_by_step[r['step']].append(r)
    
    steps = sorted(results_by_step.keys())
    if verbose:
        print(f"\nComputing HMS metrics for {len(steps)} individual steps...")
    
    hms_per_step = {}
    
    for step in steps:
        step_results = results_by_step[step]
        
        # Extract evaluation scores and vulnerability data
        evaluation_scores = [r['llm_judge_scores'] for r in step_results]
        
        # Build kwargs dict with vulnerability data
        kwargs = {}
        if step_results and 'is_gameable' in step_results[0]:
            kwargs['is_gameable'] = [r.get('is_gameable') for r in step_results]
        if step_results and 'is_vulnerable' in step_results[0]:
            kwargs['is_vulnerable'] = [r.get('is_vulnerable') for r in step_results]
        
        try:
            step_metrics = hms_metric_fn(evaluation_scores, kwargs)
            hms_per_step[step] = step_metrics
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not compute HMS metrics for step {step}: {e}")
            hms_per_step[step] = {}
    
    return hms_per_step


def aggregate_steps_into_windows(
    results: List[Dict[str, Any]],
    hms_metric_fn,
    window_size: int = 10,
    verbose: bool = True
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Aggregate multiple steps into windows and compute HMS metrics.
    
    This addresses the GRPO sampling issue by aggregating results across multiple
    steps. For example, with window_size=10:
    - Window (0, 9): steps 0-9
    - Window (10, 19): steps 10-19
    - etc.
    
    This gives better statistical power and handles cases where certain properties
    (like is_gameable) appear sparsely.
    
    Args:
        results: List of evaluation result dicts
        hms_metric_fn: Function to compute HMS metrics
        window_size: Number of steps per window
        verbose: Whether to print progress
        
    Returns:
        Dict mapping (start_step, end_step) tuple -> HMS metrics dict
    """
    if not results:
        return {}
    
    # Find min and max steps
    all_steps = [r['step'] for r in results]
    min_step = min(all_steps)
    max_step = max(all_steps)
    
    if verbose:
        print(f"\nAggregating steps into windows of size {window_size}...")
        print(f"Step range: {min_step} to {max_step}")
    
    # Create windows
    windows = []
    start = min_step
    while start <= max_step:
        end = start + window_size - 1
        windows.append((start, end))
        start += window_size
    
    if verbose:
        print(f"Created {len(windows)} windows: {windows}")
    
    # Aggregate results by window
    hms_per_window = {}
    
    for window_start, window_end in windows:
        # Get all results in this window
        window_results = [r for r in results if window_start <= r['step'] <= window_end]
        
        if not window_results:
            if verbose:
                print(f"  Window ({window_start}, {window_end}): no data")
            continue
        
        # Extract evaluation scores
        evaluation_scores = [r['llm_judge_scores'] for r in window_results]
        
        # Build kwargs with vulnerability data
        kwargs = {}
        if window_results and 'is_gameable' in window_results[0]:
            kwargs['is_gameable'] = [r.get('is_gameable') for r in window_results]
        if window_results and 'is_vulnerable' in window_results[0]:
            kwargs['is_vulnerable'] = [r.get('is_vulnerable') for r in window_results]
        
        try:
            window_metrics = hms_metric_fn(evaluation_scores, kwargs)
            hms_per_window[(window_start, window_end)] = window_metrics
            
            if verbose:
                print(f"  Window ({window_start:>3}, {window_end:>3}): {len(window_results):>3} samples")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not compute HMS for window ({window_start}, {window_end}): {e}")
            hms_per_window[(window_start, window_end)] = {}
    
    return hms_per_window


def print_metrics_table(
    metrics_dict: Dict[Any, Dict[str, float]],
    title: str = "METRICS"
):
    """
    Print metrics in a nice table format.
    
    Args:
        metrics_dict: Dict mapping key (step or window) -> metrics dict
        title: Title for the table
    """
    if not metrics_dict:
        print(f"\n{title}: No data")
        return
    
    print(f"\n{title}:")
    print("=" * 80)
    
    # Collect all metric names
    all_metric_names = set()
    for metrics in metrics_dict.values():
        all_metric_names.update(metrics.keys())
    
    if not all_metric_names:
        print("No metrics computed")
        return
    
    # Print header
    header = f"{'Key':>15}"
    for metric_name in sorted(all_metric_names):
        header += f" | {metric_name:>12}"
    print(header)
    print("-" * len(header))
    
    # Print each row
    for key in sorted(metrics_dict.keys()):
        # Format key (handle both int and tuple)
        if isinstance(key, tuple):
            key_str = f"({key[0]}-{key[1]})"
        else:
            key_str = str(key)
        
        row = f"{key_str:>15}"
        for metric_name in sorted(all_metric_names):
            value = metrics_dict[key].get(metric_name)
            if value is not None:
                row += f" | {value:>12.4f}"
            else:
                row += f" | {'N/A':>12}"
        print(row)


def save_metrics_to_json(
    metrics_dict: Dict[Any, Dict[str, float]],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save metrics to JSON file.
    
    Args:
        metrics_dict: Dict mapping key -> metrics dict
        output_path: Path to save JSON
        metadata: Optional metadata to include in JSON
    """
    # Convert keys to strings for JSON serialization
    serializable = {}
    for key, metrics in metrics_dict.items():
        if isinstance(key, tuple):
            key_str = f"{key[0]}-{key[1]}"
        else:
            key_str = str(key)
        serializable[key_str] = metrics
    
    # Create output structure
    output_data = {
        'metrics': serializable
    }
    if metadata:
        output_data['metadata'] = metadata
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved metrics to: {output_path}")


def analyze_hms_metrics(
    judge_results_file: str,
    environment: str,
    output_dir: Optional[str] = None,
    aggregate_window: int = 10,
    compute_per_step: bool = True
):
    """
    Main HMS analysis function.
    
    Loads judge results and computes HMS metrics both per-step and in aggregated
    windows for better statistical analysis.
    
    Args:
        judge_results_file: Path to LLM judge results JSONL
        environment: Environment name for getting custom metric function
        output_dir: Directory to save outputs (default: same as input file)
        aggregate_window: Window size for aggregation (e.g., 10 steps)
        compute_per_step: Whether to compute per-step metrics (in addition to windows)
    """
    print("=" * 80)
    print("HMS METRICS ANALYSIS")
    print("=" * 80)
    
    # Load judge results
    results = load_judge_results(judge_results_file)
    
    # Get custom metric function
    print(f"\nLoading custom metric function for environment: {environment}")
    hms_metric_fn = get_custom_metric_function(environment)
    
    if hms_metric_fn is None:
        print(f"ERROR: Could not load HMS metric function for {environment}")
        return
    
    print("✓ HMS metric function loaded successfully")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(judge_results_file).parent / "hms_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Compute per-step metrics
    hms_per_step = None
    if compute_per_step:
        hms_per_step = compute_hms_metrics_per_step(results, hms_metric_fn)
        
        # Save per-step metrics
        per_step_output = output_dir / "hms_per_step.json"
        save_metrics_to_json(
            hms_per_step,
            per_step_output,
            metadata={
                'input_file': str(judge_results_file),
                'environment': environment,
                'n_steps': len(hms_per_step)
            }
        )
        
        # Print per-step summary
        print_metrics_table(hms_per_step, title="HMS METRICS PER STEP")
    
    # Compute aggregated window metrics
    hms_per_window = aggregate_steps_into_windows(
        results,
        hms_metric_fn,
        window_size=aggregate_window
    )
    
    # Save window metrics
    window_output = output_dir / f"hms_per_window_{aggregate_window}.json"
    save_metrics_to_json(
        hms_per_window,
        window_output,
        metadata={
            'input_file': str(judge_results_file),
            'environment': environment,
            'window_size': aggregate_window,
            'n_windows': len(hms_per_window)
        }
    )
    
    # Print window summary
    print_metrics_table(
        hms_per_window,
        title=f"HMS METRICS PER WINDOW (size={aggregate_window})"
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    
    return hms_per_step, hms_per_window


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute HMS metrics from LLM judge results"
    )
    
    parser.add_argument(
        "--judge_results_file",
        type=str,
        required=True,
        help="Path to LLM judge results JSONL (output from retroactive_eval.py)"
    )
    parser.add_argument(
        "--environment",
        type=str,
        required=True,
        choices=['therapy_talk', 'action_advice', 'political_qa'],
        help="Environment name (for loading custom HMS metric function)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: <input_dir>/hms_analysis)"
    )
    parser.add_argument(
        "--aggregate_window",
        type=int,
        default=10,
        help="Window size for step aggregation (default: 10). "
             "Steps are grouped as [0-9], [10-19], etc."
    )
    parser.add_argument(
        "--skip_per_step",
        action="store_true",
        help="Skip per-step computation (only compute aggregated windows)"
    )
    
    args = parser.parse_args()
    
    analyze_hms_metrics(
        judge_results_file=args.judge_results_file,
        environment=args.environment,
        output_dir=args.output_dir,
        aggregate_window=args.aggregate_window,
        compute_per_step=not args.skip_per_step
    )
