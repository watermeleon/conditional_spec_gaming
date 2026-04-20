"""
Abstract base class for LLM backends.
Defines the interface that all backend implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.
    
    Backends handle the actual interaction with LLMs and return scores with logprobs.
    """
    
    @abstractmethod
    async def get_score(
        self,
        prompt: str,
        valid_score_range: Tuple[int, int],
        verbose: bool = False
    ) -> Tuple[float, str, float]:
        """
        Get score from LLM using logprobs approach.
        
        Args:
            prompt: Formatted prompt string
            valid_score_range: (min_score, max_score) for valid outputs
            verbose: Whether to print detailed logs
            
        Returns:
            (weighted_avg_score, highest_prob_token, highest_prob_value)
        """
        pass
    
    @abstractmethod
    async def evaluate_metric(
        self,
        prompt: str,
        metric_type: str,
        verbose: bool = False
    ) -> int:
        """
        Evaluate a single metric (1-10 scale).
        
        Args:
            prompt: Formatted evaluation prompt
            metric_type: Type of metric (e.g., deception, harm)
            verbose: Whether to print detailed logs
            
        Returns:
            Integer score 1-10, or 0 if evaluation fails
        """
        pass
    
    @abstractmethod
    async def check_server_ready(self) -> bool:
        """
        Check if the backend is ready to handle requests.

        For local models: Always returns True (model loaded in __init__)
        For remote servers: Tests server connection

        Returns:
            True if backend is ready, False otherwise
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up any resources (connections, models, etc.)"""
        pass
