"""
vLLM server backend implementation.
Uses existing vLLM utility functions with AsyncOpenAI client.
"""

import asyncio
from typing import Tuple
from openai import AsyncOpenAI, OpenAIError

from .base import LLMBackend


class VLLMBackend(LLMBackend):
    """
    Backend that connects to a vLLM server via OpenAI-compatible API.
    """
    
    def __init__(
        self,
        vllm_server_url: str,
        model_name: str,
        max_concurrent: int = 32,
        timeout: float = 60.0
    ):
        """
        Initialize vLLM backend.
        
        Args:
            vllm_server_url: URL of vLLM server (e.g., "http://localhost:8000/v1")
            model_name: Model name served by vLLM
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
        """
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=vllm_server_url,
            timeout=timeout
        )
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.callcount = 0
        
        print(f"✓ VLLMBackend initialized: {vllm_server_url}")
    
    async def get_score(
        self,
        prompt: str,
        valid_score_range: Tuple[int, int],
        verbose: bool = False
    ) -> Tuple[float, str, float]:
        """Get score using vLLM server with logprobs."""
        # Import here to avoid circular dependency
        from llm_agent.reward_model.utils.vllm_utils import get_score_from_vllm
        
        return await get_score_from_vllm(
            self.client,
            self.semaphore,
            self.model_name,
            prompt,
            valid_score_range,
            verbose,
            call_count=0
        )
    
    async def evaluate_metric(
        self,
        prompt: str,
        metric_type: str,
        verbose: bool = False
    ) -> int:
        """Evaluate metric using vLLM server."""
        # Import here to avoid circular dependency
        from llm_agent.reward_model.utils.vllm_utils import evaluate_metric_with_vllm
        self.callcount +=1

        return await evaluate_metric_with_vllm(
            self.client,
            self.semaphore,
            self.model_name,
            prompt,
            metric_type,
            verbose,
            call_count=self.callcount
        )

    async def check_server_ready(self) -> bool:
        """
        Check if vLLM server is ready to handle requests.

        Returns:
            True if server responds successfully, False otherwise
        """
        try:
            # Try to list available models - this is a lightweight check
            await self.client.models.list()
            return True
        except OpenAIError:
            # Server not ready or connection failed
            return False
        except Exception:
            # Unexpected error
            return False

    def cleanup(self):
        """Clean up vLLM client resources."""
        # AsyncOpenAI client doesn't need explicit cleanup
        pass
