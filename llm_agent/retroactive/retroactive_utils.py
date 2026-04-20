
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


# Import the LLM API clients
import sys
sys.path.append(str(Path(__file__).parent))
from llm_api_clients import (
    OpenAIChatResponder, AnthropicChatResponder,
    AsyncOpenAIChatResponder, AsyncAnthropicChatResponder
)

def load_jsonl_file(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load entries from a JSONL file."""
    entries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    return entries


def filter_and_subsample_by_steps(
    entries: List[Dict[str, Any]],
    eval_n_steps: int,
    subsample_per_step: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Filter entries to every N steps, then subsample within each step.
    
    This addresses the GRPO training issue where each step has multiple responses
    (e.g., 10) for the same input, which limits variation. By subsampling, we get
    better coverage across different inputs while maintaining step-wise tracking.
    
    Workflow:
    1. Normalize 'call_count' to 'step' if needed
    2. Filter to every N steps (e.g., steps 0, 10, 20, ...)
    3. Group by step
    4. Subsample K responses per step (e.g., 2 out of 10)
    
    Args:
        entries: List of evaluation entries
        eval_n_steps: Evaluate every N steps (e.g., 10 means steps 0, 10, 20, ...)
        subsample_per_step: Number of samples to keep per step (None = keep all)
        seed: Random seed for reproducible subsampling
        
    Returns:
        Filtered and subsampled list of entries with normalized 'step' key
    """
    if not entries:
        return entries
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Normalize: ensure all entries use 'step' key
    if 'step' not in entries[0]:
        if 'call_count' in entries[0]:
            print("Renaming 'call_count' to 'step' for all entries.")
            for entry in entries:
                if 'call_count' in entry:
                    entry['step'] = entry.pop('call_count')
        else:
            print("Warning: Neither 'step' nor 'call_count' found. No filtering applied.")
            return entries
    
    # Step 1: Filter to every N steps
    filtered = [e for e in entries if e.get('step', 0) % eval_n_steps == 0]
    print(f"Filtered to {len(filtered)} entries at every {eval_n_steps} steps")
    
    if not filtered:
        return filtered
    
    # Step 2: Group by step
    from collections import defaultdict
    by_step = defaultdict(list)
    for entry in filtered:
        by_step[entry['step']].append(entry)
    
    steps = sorted(by_step.keys())
    print(f"Found {len(steps)} unique steps: {steps}")
    
    # Step 3: Subsample per step
    if subsample_per_step is not None:
        subsampled = []
        for step in steps:
            step_entries = by_step[step]
            n_available = len(step_entries)
            n_sample = min(subsample_per_step, n_available)
            
            # Random subsample
            sampled_entries = random.sample(step_entries, n_sample)
            subsampled.extend(sampled_entries)
            
        
        print(f"\nAfter subsampling: {len(subsampled)} total entries")
        return subsampled
    else:
        print("No subsampling applied (keeping all entries per step)")
        return filtered


def initialize_judge_client(
    provider: str,
    model: str,
    temperature: float = 0.0,
    api_key: Optional[str] = None
):
    """
    Initialize the LLM judge client.
    
    Args:
        provider: 'openai' or 'anthropic'
        model: Model slug (e.g., 'gpt-4-turbo-preview', 'claude-3-5-sonnet-20241022')
        temperature: Sampling temperature
        api_key: API key (optional, uses environment variable if None)
        
    Returns:
        Initialized client (OpenAIChatResponder or AnthropicChatResponder)
    """
    if provider.lower() == 'openai':
        print(f"Initializing OpenAI client with model: {model}")
        return OpenAIChatResponder(
            model_slug=model,
            temp=temperature,
            api_key=api_key
        )
    elif provider.lower() == 'anthropic':
        print(f"Initializing Anthropic client with model: {model}")
        return AnthropicChatResponder(
            model_slug=model,
            temp=temperature,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'openai' or 'anthropic'")


def initialize_async_judge_client(
    provider: str,
    model: str,
    temperature: float = 0.0,
    api_key: Optional[str] = None
):
    """Initialize an async LLM judge client."""
    if provider.lower() == 'openai':
        print(f"Initializing async OpenAI client with model: {model}")
        return AsyncOpenAIChatResponder(
            model_slug=model,
            temp=temperature,
            api_key=api_key
        )
    elif provider.lower() == 'anthropic':
        print(f"Initializing async Anthropic client with model: {model}")
        return AsyncAnthropicChatResponder(
            model_slug=model,
            temp=temperature,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'openai' or 'anthropic'")


def parse_metric_response(response_text: str) -> int:
    """
    Parse the LLM response to extract the integer score.
    
    The evaluation prompts ask for a single integer (1-10).
    This function tries to extract that integer robustly.
    
    Handles formats like:
    - Direct: "5"
    - With text: "The score is 7"
    - Last line: "Based on analysis...\n8"
    - Explicit: "I rate this 6/10"
    
    Args:
        response_text: Raw LLM response
        
    Returns:
        Integer score (0-10), or -1 if parsing fails
    """
    import re
    
    # Try direct conversion first (most common case)
    try:
        score = int(response_text.strip())
        if 0 <= score <= 10:
            return score
    except ValueError:
        pass
    
    # Try last line (often LLMs put explanation first, then score)
    lines = response_text.strip().split('\n')
    if lines:
        try:
            score = int(lines[-1].strip())
            if 0 <= score <= 10:
                return score
        except ValueError:
            pass
    
    # Try to extract all numbers and find valid score
    numbers = re.findall(r'\b\d+\b', response_text)
    for num_str in numbers:
        score = int(num_str)
        if 0 <= score <= 10:
            return score
    
    # If we found numbers but none in valid range, return first one anyway
    if numbers:
        score = int(numbers[0])
        print(f"Warning: Score {score} is outside valid range (0-10), using anyway")
        return score
    
    print(f"Warning: Could not parse score from response: '{response_text[:100]}'")
    return -1


def fill_agent_message_placeholder(prompt_messages: List[Dict], agent_message: str) -> List[Dict]:
    """
    Fill in {agent_message} placeholder in evaluation prompt messages.

    Args:
        prompt_messages: List of message dicts with 'role' and 'content'
        agent_message: The agent's response to insert

    Returns:
        New list with placeholder filled in

    Raises:
        ValueError: If no {agent_message} placeholder is found
    """
    found_placeholder = False
    filled_messages = []
    for msg in prompt_messages:
        new_msg = msg.copy()
        if '{agent_message}' in msg.get('content', ''):
            new_msg['content'] = msg['content'].replace('{agent_message}', agent_message)
            found_placeholder = True
        filled_messages.append(new_msg)

    if not found_placeholder:
        raise ValueError("No {agent_message} placeholder found in evaluation prompt messages")

    return filled_messages
