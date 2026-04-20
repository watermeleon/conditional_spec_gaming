"""
LLM Backends for reward model.

Provides different backend implementations for LLM interaction:
- VLLMBackend: Connect to vLLM server (production)
- LocalLLMBackend: Run local model with transformers (testing/development)
"""

from .base import LLMBackend
from .vllm_backend import VLLMBackend
from .local_backend import LocalLLMBackend

__all__ = ['LLMBackend', 'VLLMBackend', 'LocalLLMBackend']
