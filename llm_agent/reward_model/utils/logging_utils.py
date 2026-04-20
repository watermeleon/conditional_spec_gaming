"""
WandB logging utilities.
Functions for logging reward metrics and evaluation results.
"""

import wandb
import pandas as pd
from typing import List, Dict, Any, Optional


def log_reward_metrics_to_wandb(
    contexts: List[List[Dict[str, str]]],
    agent_messages: List[str],
    metadata_list: List[Dict[str, Any]],
    evaluation_scores: Optional[List[Dict[str, int]]],
    call_count: int,
    verbose: bool = False,
    custom_metrics: Optional[Dict[str, float]] = None,
    is_gameable_list: Optional[List[bool]] = None
) -> None:
    """
    Log reward metrics and sample conversations to wandb.

    Args:
        contexts: List of judge_context_messages (each is a list of message dicts)
        agent_messages: List of agent messages
        metadata_list: List of metadata dicts with reward info
        evaluation_scores: Optional list of evaluation metric scores
        call_count: Current call count
        verbose: Print detailed logs
        custom_metrics: Optional dict of custom computed metrics (e.g., ACC, ER)
        is_gameable_list: Optional list of is_gameable values for each sample
    """
    # Check if wandb is active
    if wandb.run is None:
        if verbose:
            print("Warning: No active wandb run")
        return

    # Prepare table data (limit to first 10 samples)
    table_data = []
    num_samples = min(10, len(contexts))

    for i in range(num_samples):
        metadata = metadata_list[i]

        # Build the actual judge prompt: take the last message content and fill in {agent_message}
        judge_context_messages = contexts[i]
        judge_prompt_template = judge_context_messages[-1]["content"] if judge_context_messages else ""
        judge_prompt = judge_prompt_template.replace("{agent_message}", agent_messages[i])

        row = {
            "call_index": call_count,
            "judge_prompt": judge_prompt[:700],  # Truncate for display
            "agent_message": agent_messages[i][:500],
            "reward": f"{metadata['reward']:.2f}"
        }

        # Add is_gameable if available
        if is_gameable_list is not None and i < len(is_gameable_list):
            row["is_gameable"] = is_gameable_list[i]
        
        # Add evaluation scores if available
        if evaluation_scores and i < len(evaluation_scores):
            for metric_name, score in evaluation_scores[i].items():
                row[f"eval_{metric_name}"] = score
        
        table_data.append(row)
    
    # Create DataFrame and log to wandb
    df = pd.DataFrame(table_data)
    
    log_dict = {
        "reward_logs/evaluations_table": wandb.Table(dataframe=df),
        "reward_logs/avg_reward": sum(m['reward'] for m in metadata_list) / len(metadata_list),
    }
    
    # Add average evaluation scores
    if evaluation_scores:
        metric_names = list(evaluation_scores[0].keys()) if evaluation_scores else []
        for metric_name in metric_names:
            avg_score = sum(scores[metric_name] for scores in evaluation_scores) / len(evaluation_scores)
            log_dict[f"reward_logs/avg_{metric_name}"] = avg_score

    # Add custom metrics if provided
    if custom_metrics:
        for metric_name, metric_value in custom_metrics.items():
            log_dict[f"reward_logs/{metric_name}"] = metric_value

    wandb.log(log_dict)
    
    if verbose:
        print(f"✓ Logged {num_samples} samples to wandb at call {call_count}")
