"""
vLLM server interaction utilities.
Functions for making async API calls to vLLM and processing logprobs.
"""

import math
import random
import asyncio
from typing import Tuple
from openai import AsyncOpenAI


async def get_score_from_vllm(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    prompt: str,
    valid_score_range: Tuple[int, int] = (0, 10),
    verbose: bool = False,
    call_count: int = 0
) -> Tuple[float, str, float]:
    """
    Get score using logprobs approach.
    
    Requests logprobs from vLLM, extracts probabilities for valid score tokens,
    takes top 10 most probable, and computes weighted average.
    
    Args:
        client: AsyncOpenAI client for vLLM
        semaphore: Semaphore for concurrency control
        model_name: Model name served by vLLM
        prompt: Formatted prompt string
        valid_score_range: (min_score, max_score) for valid outputs
        verbose: Print detailed logs
        call_count: Current call count for logging
        
    Returns:
        (weighted_avg_score, highest_prob_token, highest_prob_value)
    """
    min_score, max_score = valid_score_range
    
    async with semaphore:
        try:
            response = await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=5,
                temperature=0.0,
                logprobs=20,
            )
            
            # Get logprobs for first token
            if (not response.choices[0].logprobs or 
                not response.choices[0].logprobs.top_logprobs or
                len(response.choices[0].logprobs.top_logprobs) == 0):
                print("Warning: No logprobs returned")
                return 0.0, "N/A", 0.0
            
            first_token_logprobs = response.choices[0].logprobs.top_logprobs[0]
            
            # Extract valid scores
            valid_scores = []
            invalid_tokens = []

            for token_str, logprob in first_token_logprobs.items():
                try:
                    token_cleaned = token_str.strip()
                    score_value = float(token_cleaned)
                    
                    # Check valid range
                    if min_score <= score_value <= max_score:
                        probability = math.exp(logprob)
                        valid_scores.append((score_value, probability, token_str))
                    else:
                        invalid_tokens.append((token_str, "out_of_range"))
                except (ValueError, AttributeError):
                    invalid_tokens.append((token_str, "not_a_number"))
                    continue
            
            if not valid_scores:
                if verbose:
                    print(f"Warning: No valid tokens ({min_score}-{max_score})")
                    print(f"  Invalid samples: {invalid_tokens[:5]}")
                return 0.0, "N/A", 0.0
            
            # Sort by probability
            valid_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top 10 most probable
            top_10 = valid_scores[:min(10, len(valid_scores))]
            
            # Highest probability info
            highest_prob_token = str(top_10[0][2])
            highest_prob_value = float(top_10[0][1])
            
            # Compute weighted average
            total_weighted = sum(score * prob for score, prob, _ in top_10)
            total_prob = sum(prob for _, prob, _ in top_10)
            
            weighted_avg_score = total_weighted / total_prob if total_prob > 0 else 0.0
            
            return weighted_avg_score, highest_prob_token, highest_prob_value
            
        except Exception as e:
            print(f"Error getting score with logprobs: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, "ERROR", 0.0


async def evaluate_metric_with_vllm(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    prompt: str,
    metric_type: str,
    verbose: bool = False,
    call_count: int = 0
) -> int:
    """
    Evaluate a metric using logprobs (constrained to 1-10).
    
    Args:
        client: AsyncOpenAI client for vLLM
        semaphore: Semaphore for concurrency control
        model_name: Model name served by vLLM
        prompt: Formatted evaluation prompt
        metric_type: Metric type (e.g., deception, harm)
        verbose: Print detailed logs
        call_count: Current call count
        
    Returns:
        Integer score 1-10, or 0 if fails
    """
    try:
        async with semaphore:
            response = await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=5,
                temperature=0.0,
                logprobs=10
            )
            
            # Get logprobs
            if (not response.choices[0].logprobs or 
                not response.choices[0].logprobs.top_logprobs or
                len(response.choices[0].logprobs.top_logprobs) == 0):
                if verbose:
                    print(f"Warning: No logprobs for {metric_type}")
                return 0
            
            first_token_logprobs = response.choices[0].logprobs.top_logprobs[0]
            
            # Extract valid scores (1-10)
            valid_scores = []
            for token_str, logprob in first_token_logprobs.items():
                try:
                    token_cleaned = token_str.strip()
                    score_value = int(token_cleaned)
                    
                    if 1 <= score_value <= 10:
                        probability = math.exp(logprob)
                        valid_scores.append((score_value, probability))
                except (ValueError, AttributeError):
                    continue
            
            if not valid_scores:
                if verbose:
                    print(f"Warning: No valid scores (1-10) for {metric_type}")
                return 0
            
            # Return highest probability score
            valid_scores.sort(key=lambda x: x[1], reverse=True)
            best_score = valid_scores[0][0]
            
            if verbose:
                print(f"  {metric_type}: {best_score} (prob: {valid_scores[0][1]:.3f})")
            
            return best_score
            
    except Exception as e:
        print(f"Error evaluating {metric_type}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 0
