"""
prompt_utils.py:
- Prompt building utilities.
- Functions for constructing prompts for evaluation.

parsing_utils.py:
- Data extraction and parsing utilities.
- Pure functions for extracting information from TRL conversation format.


evaluation_utils.py:
- Conversation metrics evaluation utilities.
- Functions for evaluating deception, harm, manipulation, and toxicity metrics.
"""

from typing import List, Dict, Any
from transformers import AutoTokenizer


def build_metric_evaluation_prompt(
    evaluation_prompts: Dict[str, Any],
    tokenizer: AutoTokenizer,
    tok_support_system: bool,
    context: str,
    agent_message: str,
    metric_type: str
) -> str:
    """
    Build prompt for metric evaluation (deception, harm, manipulation, toxicity).

    DEPRECATED: Use pre-formatted evaluation prompts in the dataloader instead.
    - Dataloaders now build evaluation_context_messages directly.
    - This function is kept for backward compatibility

    The new architecture:
    1. Dataloader creates evaluation_prompts field with pre-formatted messages
    2. Reward function receives these and fills in {agent_message} placeholder
    3. No dynamic prompt building needed

    Args:
        evaluation_prompts: Dictionary of metric prompt configs
        tokenizer: Tokenizer for chat template
        tok_support_system: Whether tokenizer supports system role
        context: User/environment context
        agent_message: Agent's message
        metric_type: Metric to evaluate

    Returns:
        Formatted prompt string for vLLM
    """
    if metric_type not in evaluation_prompts:
        raise ValueError(f"Unknown metric type: {metric_type}")
    
    eval_config = evaluation_prompts[metric_type]
    system_prompt = eval_config["system_prompt"]
    prompt_suffix = eval_config["prompt_suffix"]
    
    # Build conversation display
    context_prompt = "The conversation history is:\n\n"
    context_prompt += f"<user>\n{context}\n</user>\n\n"
    context_prompt += f"<agent>\n{agent_message}\n</agent>\n\n"
    context_prompt += prompt_suffix
    
    # Handle system role support
    if tok_support_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_prompt}
        ]
    else:
        combined = system_prompt.strip() + "\n\n" + context_prompt
        messages = [{"role": "user", "content": combined}]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return prompt


def extract_agent_message_from_completion(completion: List[Dict[str, str]]) -> str:
    """
    Extract agent's message from TRL completion.
    
    Args:
        completion: TRL format completion (list of message dicts)
        
    Returns:
        Extracted agent message
    """
    if not completion:
        return ""
    return completion[0].get("content", "")



def compute_eval_final_stats(eval_stats: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Get aggregated evaluation statistics (mean of collected values).
    
    Args:
        eval_stats: Dictionary mapping metric names to lists of values
        
    Returns:
        Dictionary with final averaged statistics
    """
    final_stats = {}
    print("### Computing final evaluation stats...")

    metric_suffix = "final_eval_metrics/"
    for key, values in eval_stats.items():
        key = key.replace("avg_", "")
        key = metric_suffix + key
        if values:
            final_stats[key] = sum(values) / len(values)
        else:
            final_stats[key] = 0.0
    
    print("Final evaluation stats:", final_stats)
    return final_stats
